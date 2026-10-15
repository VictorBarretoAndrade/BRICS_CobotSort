"""PyQt6 operator GUI: manual jog, semi-automatic (module A), automatic and agent modes, E-stop.

Threading model
---------------
* The simulation and the control loop run in a worker ``QThread`` (:class:`SimWorker`). Each job
  (a jog, one pick, the automatic cycle, an agent order) runs there to completion; the GUI and the
  worker talk through Qt signals and slots, which are queued across threads.
* ``MjData`` is shared by both threads, so every physics step and every state change is done under a
  ``threading.RLock`` (``SimBackend.lock``). The GUI thread copies the state under the same lock and
  renders the copy.
* The ``mujoco.Renderer`` owns an OpenGL context, so it is created and used only in the GUI thread.
  When the worker needs a camera image (a scan), it emits a request and waits for the GUI thread.
* CPython threads share one interpreter lock (the GIL), so the worker does not make the simulation
  faster; it keeps the window responsive while long motions run.

E-stop flow (as in the camp's GUI): E-STOP -> lamp RED -> "Reset errors" -> "Engage".
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from cobotsort.agent.operator import OperatorAgent
from cobotsort.cell.sorter import SortingCell, build_cell
from cobotsort.config import CLASS_NAMES, ROOT, CellConfig
from cobotsort.kinematics.ik import Pose
from cobotsort.sim.render import CellRenderer, draw_boxes

log = logging.getLogger(__name__)

SPEEDS = {"1x": 1.0, "2x": 2.0, "4x": 4.0, "8x": 8.0, "max": 0.0}
BINS = ["A", "B", "C", "REJECT"]


# ---------------------------------------------------------------- worker side
@dataclass
class Job:
    name: str
    fn: Callable[[], str]


@dataclass
class RenderRequest:
    camera: str
    width: int
    height: int
    event: threading.Event = field(default_factory=threading.Event)
    image: np.ndarray | None = None


class SimWorker(QObject):
    """Runs jobs against the cell in its own thread."""

    log = pyqtSignal(str)
    job_done = pyqtSignal(str, bool, str)
    render_request = pyqtSignal(object)

    def __init__(self, cell: SortingCell) -> None:
        super().__init__()
        self.cell = cell
        self.speed = 4.0
        self.running = threading.Event()
        self.running.set()
        self.stop_auto = False
        self.busy = False
        self._sim0 = 0.0
        self._wall0 = 0.0
        self._n = 0
        cell.api.backend.step_callbacks.append(self._pace)
        cell.render_top = self._render_top  # scans ask the GUI thread for the image

    def _pace(self, world) -> None:
        """Real-time pacing and pause, called after every physics step."""
        self._n += 1
        if self._n % 10:
            return
        backend = self.cell.api.backend
        while not self.running.is_set() and not backend.stop_request:
            time.sleep(0.02)
            self._sim0, self._wall0 = world.time, time.perf_counter()
        if self.speed > 0:
            ahead = self._wall0 + (world.time - self._sim0) / self.speed - time.perf_counter()
            if ahead > 0:
                time.sleep(min(ahead, 0.05))

    def _render_top(self) -> np.ndarray:
        w, h = self.cell.cfg.scene.top_cam_size
        req = RenderRequest("top_cam", w, h)
        self.render_request.emit(req)
        if not req.event.wait(10.0) or req.image is None:
            raise RuntimeError("the GUI did not render the top camera")
        return req.image

    @pyqtSlot(object)
    def run(self, job: Job) -> None:
        self.busy = True
        self._sim0, self._wall0 = self.cell.world.time, time.perf_counter()
        self.log.emit(f"[job] {job.name}")
        try:
            detail = job.fn()
            ok = True
        except Exception as exc:  # noqa: BLE001 - every failure is reported to the operator
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        self.busy = False
        self.log.emit(f"[job] {job.name}: {'done' if ok else 'FAILED'} {detail}")
        self.job_done.emit(job.name, ok, detail)


# ---------------------------------------------------------------- widgets
class LampWidget(QWidget):
    """Three-light signal tower."""

    COLORS = {
        "RED": QColor(255, 40, 30),
        "YELLOW": QColor(255, 200, 20),
        "GREEN": QColor(40, 220, 60),
    }

    def __init__(self) -> None:
        super().__init__()
        self.state = "YELLOW"
        self.setFixedSize(70, 190)

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(28, 10, 14, 175, QColor(60, 60, 64))
        for i, name in enumerate(("RED", "YELLOW", "GREEN")):
            color = self.COLORS[name] if name == self.state else self.COLORS[name].darker(400)
            p.setBrush(color)
            p.setPen(QColor(20, 20, 20))
            p.drawEllipse(10, 10 + 58 * i, 50, 50)
        p.end()


def to_pixmap(rgb: np.ndarray, width: int | None = None) -> QPixmap:
    h, w, _ = rgb.shape
    img = QImage(
        np.ascontiguousarray(rgb).tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888
    ).copy()
    pix = QPixmap.fromImage(img)
    if width is not None:
        pix = pix.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
    return pix


class ClickableLabel(QLabel):
    clicked = pyqtSignal(float, float)  # normalized x, y

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        pix = self.pixmap()
        if pix is not None and not pix.isNull():
            pos = event.position()
            self.clicked.emit(pos.x() / pix.width(), pos.y() / pix.height())


# ---------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    submit = pyqtSignal(object)

    def __init__(
        self, detector: str = "yolo", seed: int = 3, cfg: CellConfig | None = None
    ) -> None:
        super().__init__()
        self.setWindowTitle("CobotSort AI - operator GUI (simulation)")
        self.cfg = cfg or CellConfig()
        self.cell = build_cell(detector, seed=seed, cfg=self.cfg)
        self.world = self.cell.world
        self.api = self.cell.api
        self.lock = threading.RLock()
        self.api.backend.lock = self.lock
        # Rendering lives in this (GUI) thread.
        self.renderer = CellRenderer(self.world.model)
        self.view = mujoco.MjData(self.world.model)
        self.selected: int | None = None
        self.agent: OperatorAgent | None = None
        self.on_job_done: list[Callable[[str, bool, str], None]] = []
        # Worker thread.
        self.thread = QThread(self)
        self.worker = SimWorker(self.cell)
        self.worker.moveToThread(self.thread)
        self.submit.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.job_done.connect(self._job_done)
        self.worker.render_request.connect(self._serve_render)
        self.thread.start()
        self.api.listeners.append(self.worker.log.emit)
        self.cell.listeners.append(self.worker.log.emit)
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)  # about 20 fps
        self.append_log(f"cell ready: seed {seed}, detector {detector}. Press Engage to start.")

    # ------------------------------------------------------------ layout
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        top = QHBoxLayout()
        root.addLayout(top, stretch=5)
        # Left: views.
        views = QVBoxLayout()
        self.persp_label = QLabel()
        self.persp_label.setMinimumSize(832, 468)
        views.addWidget(self.persp_label)
        self.top_label = ClickableLabel()
        self.top_label.setFixedSize(400, 300)
        self.top_label.clicked.connect(self._clicked_top)
        row = QHBoxLayout()
        row.addWidget(self.top_label)
        self.status_label = QLabel()
        self.status_label.setFont(QFont("Consolas", 10))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        row.addWidget(self.status_label, stretch=1)
        views.addLayout(row)
        top.addLayout(views, stretch=3)
        # Right: lamp, E-stop, tabs.
        right = QVBoxLayout()
        safety = QHBoxLayout()
        self.lamp = LampWidget()
        safety.addWidget(self.lamp)
        buttons = QVBoxLayout()
        self.estop = QPushButton("E-STOP")
        self.estop.setMinimumHeight(90)
        self.estop.setStyleSheet(
            "QPushButton {background: #d01010; color: white; font-size: 26px; font-weight: bold;"
            " border-radius: 12px;} QPushButton:pressed {background: #800000;}"
        )
        self.estop.clicked.connect(self.press_estop)
        buttons.addWidget(self.estop)
        self.reset_btn = QPushButton("Reset errors")
        self.reset_btn.clicked.connect(self.reset_errors)
        self.engage_btn = QPushButton("Engage")
        self.engage_btn.clicked.connect(self.engage)
        buttons.addWidget(self.reset_btn)
        buttons.addWidget(self.engage_btn)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Sim speed"))
        self.speed_box = QComboBox()
        self.speed_box.addItems(list(SPEEDS))
        self.speed_box.setCurrentText("4x")
        self.speed_box.currentTextChanged.connect(
            lambda t: setattr(self.worker, "speed", SPEEDS[t])
        )
        speed_row.addWidget(self.speed_box)
        buttons.addLayout(speed_row)
        safety.addLayout(buttons)
        right.addLayout(safety)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._manual_tab(), "Manual")
        self.tabs.addTab(self._semi_tab(), "Semi-auto")
        self.tabs.addTab(self._auto_tab(), "Auto")
        self.tabs.addTab(self._agent_tab(), "Agent")
        right.addWidget(self.tabs)
        top.addLayout(right, stretch=2)
        # Bottom: log.
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFont(QFont("Consolas", 9))
        root.addWidget(self.log_view, stretch=1)
        self.setCentralWidget(central)
        self.job_buttons: list[QPushButton] = [
            b
            for b in self.findChildren(QPushButton)
            if b not in (self.estop, self.reset_btn, self.engage_btn, self.pause_btn, self.stop_btn)
        ]

    def _manual_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        joints = QGroupBox("Joint jog")
        grid = QGridLayout(joints)
        self.joint_step = QComboBox()
        self.joint_step.addItems(["1", "5", "10", "30"])
        self.joint_step.setCurrentText("10")
        grid.addWidget(QLabel("step (deg)"), 0, 0)
        grid.addWidget(self.joint_step, 0, 1, 1, 2)
        for j in range(6):
            grid.addWidget(QLabel(f"J{j + 1}"), j + 1, 0)
            for col, sign in ((1, -1), (2, 1)):
                b = QPushButton("-" if sign < 0 else "+")
                b.clicked.connect(lambda _=False, j=j, s=sign: self.jog_joint(j, s))
                grid.addWidget(b, j + 1, col)
        lay.addWidget(joints)
        cart = QGroupBox("Cartesian jog (tool orientation kept)")
        grid = QGridLayout(cart)
        self.cart_step = QComboBox()
        self.cart_step.addItems(["5", "10", "50", "100"])
        self.cart_step.setCurrentText("50")
        grid.addWidget(QLabel("step (mm)"), 0, 0)
        grid.addWidget(self.cart_step, 0, 1, 1, 2)
        for i, axis in enumerate("XYZ"):
            grid.addWidget(QLabel(axis), i + 1, 0)
            for col, sign in ((1, -1), (2, 1)):
                b = QPushButton("-" if sign < 0 else "+")
                b.clicked.connect(lambda _=False, i=i, s=sign: self.jog_cart(i, s))
                grid.addWidget(b, i + 1, col)
        lay.addWidget(cart)
        row = QHBoxLayout()
        for text, fn in (
            ("Tool ON", self.tool_on),
            ("Tool OFF", self.tool_off),
            ("Home", self.home),
        ):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        lay.addStretch()
        return w

    def _semi_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self.scan)
        lay.addWidget(self.scan_btn)
        lay.addWidget(QLabel("Detections (click one here or on the top camera):"))
        self.det_list = QListWidget()
        self.det_list.currentRowChanged.connect(self.select_detection)
        lay.addWidget(self.det_list)
        row = QHBoxLayout()
        row.addWidget(QLabel("Bin"))
        self.bin_box = QComboBox()
        self.bin_box.addItems(BINS)
        row.addWidget(self.bin_box)
        self.exec_btn = QPushButton("Execute")
        self.exec_btn.clicked.connect(self.execute_semi)
        row.addWidget(self.exec_btn)
        lay.addLayout(row)
        return w

    def _auto_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_auto)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_auto)
        for b in (self.start_btn, self.pause_btn, self.stop_btn):
            row.addWidget(b)
        lay.addLayout(row)
        lay.addWidget(QLabel("Rules (class -> bin):"))
        self.rules_table = QTableWidget(len(CLASS_NAMES), 2)
        self.rules_table.setHorizontalHeaderLabels(["class", "bin"])
        for i, cls in enumerate(CLASS_NAMES):
            self.rules_table.setCellWidget(i, 0, QLabel(cls))
            box = QComboBox()
            box.addItems(BINS)
            box.setCurrentText(self.cell.rules.mapping[cls])
            box.currentTextChanged.connect(
                lambda t, c=cls: self.cell.rules.mapping.__setitem__(c, t)
            )
            self.rules_table.setCellWidget(i, 1, box)
        lay.addWidget(self.rules_table)
        self.auto_label = QLabel("")
        lay.addWidget(self.auto_label)
        lay.addStretch()
        return w

    def _agent_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Provider"))
        self.provider_box = QComboBox()
        self.provider_box.addItems(["mock", "anthropic"])
        row.addWidget(self.provider_box)
        lay.addLayout(row)
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        lay.addWidget(self.chat)
        row = QHBoxLayout()
        self.order_edit = QLineEdit()
        self.order_edit.setPlaceholderText(
            "Ex.: Separe as peças por cor e mande as defeituosas para o rejeito."
        )
        self.order_edit.returnPressed.connect(self.send_order)
        row.addWidget(self.order_edit)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_order)
        row.addWidget(self.send_btn)
        lay.addLayout(row)
        return w

    # ------------------------------------------------------------ rendering
    def _copy_state(self) -> None:
        with self.lock:
            self.view.qpos[:] = self.world.data.qpos
            self.view.mocap_pos[:] = self.world.data.mocap_pos
            self.view.time = self.world.data.time
        mujoco.mj_forward(self.world.model, self.view)

    @pyqtSlot(object)
    def _serve_render(self, req: RenderRequest) -> None:
        try:
            self._copy_state()
            req.image = self.renderer.rgb(self.view, req.camera, req.width, req.height)
        finally:
            req.event.set()

    def refresh(self) -> None:
        self._copy_state()
        persp = self.renderer.rgb(self.view, "persp_cam", 832, 468)
        self.persp_label.setPixmap(to_pixmap(persp))
        w, h = self.cfg.scene.top_cam_size
        top = self.renderer.rgb(self.view, "top_cam", w, h)
        scan = self.cell.last_scan
        if scan is not None:
            top = draw_boxes(top, scan.detections)
            if self.selected is not None and self.selected < len(scan.detections):
                x1, y1, x2, y2 = (int(v) for v in scan.detections[self.selected].bbox_xyxy)
                cv2.rectangle(top, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 230, 0), 2)
        self.top_label.setPixmap(to_pixmap(top, 400))
        api = self.api
        self.lamp.set_state(api.lamp.state.value)
        placed = sum(1 for c in self.world.cubes() if self.world.bin_of(c.pos))
        self.status_label.setText(
            f"engaged : {api.engaged}\n"
            f"E-stop  : {api.estop_active}\n"
            f"fault   : {api.fault or '-'}\n"
            f"lamp    : {api.lamp.state.value}\n"
            f"command : {api.command[:40]}\n"
            f"held    : {api.backend.held_part is not None}\n"
            f"in bins : {placed}/{len(self.world.cubes())}\n"
            f"slots   : {self.cell.slots.free_by_bin()}\n"
            f"sim t   : {self.world.time:7.1f} s\n"
            f"busy    : {self.worker.busy}"
        )
        for b in self.job_buttons:
            b.setEnabled(not self.worker.busy)
        self.reset_btn.setEnabled(not self.worker.busy)
        self.engage_btn.setEnabled(not self.worker.busy and not api.engaged)

    # ------------------------------------------------------------ actions
    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def run_job(self, name: str, fn: Callable[[], str]) -> bool:
        if self.worker.busy:
            self.append_log(f"busy: '{name}' ignored")
            return False
        self.worker.busy = True
        self.submit.emit(Job(name, fn))
        return True

    def _job_done(self, name: str, ok: bool, detail: str) -> None:
        self._fill_detections()
        if name.startswith("agent"):
            self.chat.append(f"<b>agent:</b> {detail}" if ok else f"<b>error:</b> {detail}")
        for cb in list(self.on_job_done):
            cb(name, ok, detail)

    def engage(self) -> None:
        try:
            self.api.engage()
            self.append_log("engaged")
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"engage refused: {exc}")

    def press_estop(self) -> None:
        """E-STOP: interrupt the running motion between two physics steps, lamp RED."""
        self.api.backend.stop_request = "E-stop"
        self.worker.running.set()
        self.worker.stop_auto = True
        self.api.emergency_stop()
        self.append_log("E-STOP pressed: lamp RED. Reset errors, then Engage.")

    def reset_errors(self) -> None:
        self.api.reset_errors()
        self.worker.stop_auto = False
        self.append_log("errors reset: press Engage")

    def jog_joint(self, j: int, sign: int) -> None:
        delta = np.radians(float(self.joint_step.currentText())) * sign

        def job() -> str:
            q = self.world.q
            q[j] += delta
            self.api.move_to_point_j(q, label=f"jog J{j + 1} {np.degrees(delta):+.0f} deg")
            return f"J{j + 1} = {np.degrees(self.world.q[j]):.1f} deg"

        self.run_job(f"jog J{j + 1}", job)

    def jog_cart(self, axis: int, sign: int) -> None:
        delta = float(self.cart_step.currentText()) / 1000.0 * sign

        def job() -> str:
            pose = self.api.get_tool_position()
            target = Pose(pose.pos.copy(), pose.rot)
            target.pos[axis] += delta
            self.api.move_to_point_l(target, label=f"jog {'XYZ'[axis]} {delta * 1e3:+.0f} mm")
            p = self.api.get_tool_position().pos
            return f"TCP = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m"

        self.run_job(f"jog {'XYZ'[axis]}", job)

    def tool_on(self) -> None:
        self.run_job("tool on", lambda: self.api.tool_on().message)

    def tool_off(self) -> None:
        self.run_job("tool off", lambda: self.api.tool_off().message)

    def home(self) -> None:
        self.run_job("home", lambda: (self.api.move_to_start(), "at home")[1])

    def scan(self) -> None:
        self.run_job("scan", lambda: f"{len(self.cell.scan().detections)} parts detected")

    def _fill_detections(self) -> None:
        self.det_list.clear()
        scan = self.cell.last_scan
        if scan is None:
            return
        for i, d in enumerate(scan.detections):
            self.det_list.addItem(
                f"{i}: {d.class_name}  conf {d.confidence:.2f}  "
                f"({d.world_xy[0] * 1e3:.0f}, {d.world_xy[1] * 1e3:.0f}) mm"
            )
        self.selected = None

    def select_detection(self, row: int) -> None:
        scan = self.cell.last_scan
        if scan is None or not 0 <= row < len(scan.detections):
            self.selected = None
            return
        self.selected = row
        target = self.cell.rules.bin_for(scan.detections[row])
        if target:
            self.bin_box.setCurrentText(target)

    def _clicked_top(self, u: float, v: float) -> None:
        scan = self.cell.last_scan
        if scan is None:
            return
        w, h = self.cfg.scene.top_cam_size
        x, y = u * w, v * h
        for i, d in enumerate(scan.detections):
            x1, y1, x2, y2 = d.bbox_xyxy
            if x1 - 4 <= x <= x2 + 4 and y1 - 4 <= y <= y2 + 4:
                self.det_list.setCurrentRow(i)
                return

    def execute_semi(self) -> None:
        scan = self.cell.last_scan
        if scan is None or self.selected is None:
            self.append_log("scan and select a detection first")
            return
        det, bin_name = scan.detections[self.selected], self.bin_box.currentText()
        scan_id = scan.scan_id

        def job() -> str:
            if self.cell.last_scan is None or self.cell.last_scan.scan_id != scan_id:
                raise RuntimeError("the selection belongs to an old scan: scan again")
            rec = self.cell.pick_to_bin(det, bin_name)
            return f"{rec.class_name} -> {rec.slot} ({'verified' if rec.verified else rec.reason})"

        if self.run_job("semi pick", job):
            self.selected = None

    def start_auto(self) -> None:
        self.worker.stop_auto = False

        def job() -> str:
            report = self.cell.run_auto(should_stop=lambda: self.worker.stop_auto)
            text = f"{report.placed} placed, {len(report.flagged)} flagged" + (
                f", {report.stop_reason}" if report.stop_reason else ""
            )
            return text

        self.run_job("auto", job)

    def toggle_pause(self) -> None:
        if self.worker.running.is_set():
            self.worker.running.clear()
            self.pause_btn.setText("Resume")
            self.append_log("paused")
        else:
            self.worker.running.set()
            self.pause_btn.setText("Pause")
            self.append_log("resumed")

    def stop_auto(self) -> None:
        self.worker.stop_auto = True
        self.append_log("auto: stop after the current part")

    def send_order(self) -> None:
        order = self.order_edit.text().strip()
        if not order:
            return
        provider = self.provider_box.currentText()
        self.chat.append(f"<b>you:</b> {order}")
        self.order_edit.clear()

        def job() -> str:
            agent = OperatorAgent(self.cell, self.cfg.agent, provider=provider)
            agent.listeners.append(self.worker.log.emit)
            res = agent.run(order)
            if res.fallback_reason:
                return f"[fallback: {res.fallback_reason}] {res.final_text}"
            return res.final_text

        self.run_job("agent order", job)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.api.backend.stop_request = "window closed"
        self.worker.running.set()
        self.thread.quit()
        self.thread.wait(3000)
        self.timer.stop()
        self.renderer.close()
        super().closeEvent(event)


# ---------------------------------------------------------------- self-test
class SelfTest:
    """Drives every mode through the GUI and saves screenshots (`cobotsort gui --selftest`)."""

    def __init__(self, win: MainWindow, out_dir: Path, report: Path) -> None:
        self.win = win
        self.out = out_dir
        self.report = report
        self.out.mkdir(parents=True, exist_ok=True)
        self.results: list[tuple[str, bool, str]] = []
        self.steps: list[Callable[[], None]] = [
            self.s_engage,
            self.s_jog_refused,
            self.s_jog_joint,
            self.s_jog_cart,
            self.s_home,
            self.s_scan,
            self.s_semi,
            self.s_auto,
            self.s_agent,
            self.s_estop,
            self.s_recover,
        ]
        self.pending: Callable[[str, bool, str], None] | None = None
        win.on_job_done.append(self._done)

    def start(self) -> None:
        QTimer.singleShot(500, self._next)

    def _next(self) -> None:
        if not self.steps:
            self._finish()
            return
        QTimer.singleShot(200, self.steps.pop(0))

    def _done(self, name: str, ok: bool, detail: str) -> None:
        if self.pending is not None:
            cb, self.pending = self.pending, None
            cb(name, ok, detail)

    def check(self, name: str, ok: bool, detail: str = "", shot: str | None = None) -> None:
        self.results.append((name, bool(ok), detail))
        self.win.append_log(f"[selftest] {'PASS' if ok else 'FAIL'} {name} {detail}")
        if shot:
            self.win.refresh()
            self.win.grab().save(str(self.out / shot))
        self._next()

    def after_job(self, name: str, check: Callable[[bool, str], tuple[bool, str]], shot=None):
        def cb(_n: str, ok: bool, detail: str) -> None:
            good, why = check(ok, detail)
            self.check(name, good, why, shot)

        self.pending = cb

    # ------------------------------------------------------------ steps
    def s_engage(self) -> None:
        self.win.engage()
        self.check("engage", self.win.api.engaged, f"lamp {self.win.api.lamp.state.value}")

    def s_jog_refused(self) -> None:
        # From home, J1 +10 deg swings the tool past the workspace limit x = -0.2 m.
        q0 = self.win.world.q.copy()
        self.win.tabs.setCurrentIndex(0)
        self.after_job(
            "manual: unsafe jog J1 +10 deg is refused",
            lambda ok, d: (
                not ok and "workspace" in d and np.allclose(self.win.world.q, q0, atol=1e-4),
                d,
            ),
        )
        self.win.jog_joint(0, +1)

    def s_jog_joint(self) -> None:
        q0 = self.win.world.q[0]
        self.after_job(
            "manual: jog J1 -10 deg",
            lambda ok, d: (ok and abs(self.win.world.q[0] - q0 + np.radians(10)) < 0.01, d),
        )
        self.win.jog_joint(0, -1)

    def s_jog_cart(self) -> None:
        z0 = self.win.api.get_tool_position().pos[2]
        self.after_job(
            "manual: jog Z -50 mm",
            lambda ok, d: (
                ok and abs(self.win.api.get_tool_position().pos[2] - z0 + 0.05) < 2e-3,
                d,
            ),
            shot="gui_manual.png",
        )
        self.win.jog_cart(2, -1)

    def s_home(self) -> None:
        self.after_job("manual: home", lambda ok, d: (ok, d))
        self.win.home()

    def s_scan(self) -> None:
        self.win.tabs.setCurrentIndex(1)

        def check(ok: bool, d: str) -> tuple[bool, str]:
            n = self.win.det_list.count()
            if n:
                self.win.det_list.setCurrentRow(0)
            return ok and n > 0, d

        self.after_job("semi-auto: scan", check, shot="gui_semi.png")
        self.win.scan()

    def s_semi(self) -> None:
        self.after_job("semi-auto: execute selection", lambda ok, d: (ok and "verified" in d, d))
        self.win.execute_semi()

    def s_auto(self) -> None:
        self.win.tabs.setCurrentIndex(2)
        start = sum(1 for c in self.win.world.cubes() if self.win.world.bin_of(c.pos))
        QTimer.singleShot(1500, lambda: self.win.grab().save(str(self.out / "gui_auto.png")))
        QTimer.singleShot(6000, self.win.stop_auto)  # operator presses Stop after a few seconds

        def check(ok: bool, d: str) -> tuple[bool, str]:
            now = sum(1 for c in self.win.world.cubes() if self.win.world.bin_of(c.pos))
            return ok and now > start, f"{d}; parts in bins {start} -> {now}"

        self.after_job("auto: start, then stop", check)
        self.win.start_auto()

    def s_agent(self) -> None:
        self.win.tabs.setCurrentIndex(3)
        self.win.order_edit.setText("Separe todas as peças por cor.")
        self.after_job(
            "agent: Portuguese order (mock)",
            lambda ok, d: (ok and "Movi" in d, d),
            shot="gui_agent.png",
        )
        self.win.send_order()

    def s_estop(self) -> None:
        self.win.tabs.setCurrentIndex(0)
        self.win.joint_step.setCurrentText("30")
        self.win.speed_box.setCurrentText("1x")  # real time, so the press lands mid-motion
        QTimer.singleShot(700, self.win.press_estop)

        def check(ok: bool, d: str) -> tuple[bool, str]:
            api = self.win.api
            good = (
                (not ok) and api.estop_active and api.lamp.state.value == "RED" and not api.engaged
            )
            return good, f"job failed as expected: {d}"

        self.after_job("E-stop during a motion", check, shot="gui_estop.png")
        self.win.jog_joint(0, -1)

    def s_recover(self) -> None:
        self.win.speed_box.setCurrentText("max")
        self.win.reset_errors()
        self.win.engage()
        self.after_job(
            "reset errors -> engage -> home",
            lambda ok, d: (ok and self.win.api.lamp.state.value == "YELLOW", d),
        )
        self.win.home()

    def _finish(self) -> None:
        ok = all(r[1] for r in self.results)
        lines = [
            "# GUI self-test",
            "",
            "Command: `cobotsort gui --selftest`",
            "",
            "| Step | Result | Detail |",
            "|---|---|---|",
        ]
        lines += [f"| {n} | {'pass' if g else 'FAIL'} | {d} |" for n, g, d in self.results]
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for n, g, d in self.results:
            print(f"{'PASS' if g else 'FAIL'}  {n}  {d}")
        print(f"GUI self-test: {sum(r[1] for r in self.results)}/{len(self.results)} passed")
        self.win.close()
        QApplication.instance().exit(0 if ok else 1)


def run_gui(
    detector: str = "yolo",
    seed: int = 3,
    selftest: bool = False,
    out_dir: Path | None = None,
    report: Path | None = None,
) -> int:
    """Open the GUI. With ``selftest`` it drives every mode, saves screenshots and exits."""
    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Segoe UI", 9))  # falls back to the platform default elsewhere
    win = MainWindow(detector, seed)
    win.resize(1500, 950)
    win.show()
    if selftest:
        win.speed_box.setCurrentText("max")
        report = report or ROOT / "results" / "gui_selftest.md"
        tester = SelfTest(win, out_dir or ROOT / "media", report)
        tester.start()
    return app.exec()
