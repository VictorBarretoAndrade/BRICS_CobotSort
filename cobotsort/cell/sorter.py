"""The sorting cell: scan -> detect -> plan -> pick-and-place -> verify.

``SortingCell`` is shared by the automatic cycle (module B), the semi-automatic CLI and GUI
(module A), the benchmark and the operator agent's tools.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from cobotsort.cell.rules import SortingRules
from cobotsort.cell.slots import BinFullError, Slot, SlotAllocator
from cobotsort.config import CellConfig
from cobotsort.motion.trajectory import MotionError
from cobotsort.perception.camera import CameraModel
from cobotsort.perception.detector import Detection, Detector
from cobotsort.robot.api import CobotAPI, SafetyViolation, make_api
from cobotsort.robot.backend import CobotError, ProtectiveStop
from cobotsort.sim.world import SimWorld

log = logging.getLogger(__name__)


@dataclass
class Scan:
    scan_id: str
    detections: list[Detection]
    sim_time: float
    image: np.ndarray | None = None


@dataclass
class PickRecord:
    class_name: str
    confidence: float
    world_xy: tuple[float, float]
    bin: str
    slot: str
    success: bool
    verified: bool
    attempt: int
    reason: str
    sim_time: float
    duration: float


@dataclass
class SortReport:
    seed: int | None
    detector: str
    picks: list[PickRecord] = field(default_factory=list)
    flagged: list[dict] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    protective_stops: int = 0
    unplanned_contacts: int = 0
    completed: bool = False
    stop_reason: str = ""
    sim_time: float = 0.0
    wall_time: float = 0.0
    scans: int = 0
    localization_errors_mm: list[float] = field(default_factory=list)

    @property
    def placed(self) -> int:
        return sum(1 for p in self.picks if p.verified)


class SortingCell:
    """Owns the cell logic on top of a CobotAPI and a Detector."""

    def __init__(
        self,
        world: SimWorld,
        api: CobotAPI,
        detector: Detector,
        rules: SortingRules | None = None,
        renderer=None,
    ) -> None:
        self.world = world
        self.api = api
        self.detector = detector
        self.cfg: CellConfig = world.cfg
        self.rules = rules or SortingRules(min_confidence=self.cfg.vision.min_confidence)
        self.slots = SlotAllocator(self.cfg.scene)
        self.renderer = renderer
        w, h = self.cfg.scene.top_cam_size
        world.forward()
        self.camera = CameraModel.from_sim(world.model, world.data, "top_cam", w, h)
        self._scan_count = 0
        self.last_scan: Scan | None = None
        self.placements_since_scan = 0
        self.log: list[str] = []
        self.listeners: list[Callable[[str], None]] = []
        self.localization_errors_mm: list[float] = []

    # ------------------------------------------------------------ helpers
    def note(self, text: str) -> None:
        self.log.append(text)
        log.info("%s", text)
        for cb in self.listeners:
            cb(text)

    @property
    def needs_image(self) -> bool:
        return self.detector.needs_image

    def render_top(self) -> np.ndarray:
        if self.renderer is None:
            from cobotsort.sim.render import CellRenderer

            self.renderer = CellRenderer(self.world.model)
        w, h = self.cfg.scene.top_cam_size
        return self.renderer.rgb(self.world.data, "top_cam", w, h)

    def at_park(self) -> bool:
        return bool(np.max(np.abs(self.world.q - self.api.home_q)) < 0.02)

    def bin_count(self, bin_name: str) -> int:
        """Simulated bin presence sensor: how many parts rest inside ``bin_name``."""
        return sum(1 for c in self.world.cubes() if self.world.bin_of(c.pos) == bin_name)

    def in_pick_area(self, xy: tuple[float, float], margin: float = 0.03) -> bool:
        sc = self.cfg.scene
        return (
            sc.pick_x[0] - margin <= xy[0] <= sc.pick_x[1] + margin
            and sc.pick_y[0] - margin <= xy[1] <= sc.pick_y[1] + margin
        )

    # ------------------------------------------------------------ scan
    def scan(self, park: bool = True) -> Scan:
        """Park the arm out of the camera view, detect, convert to world coordinates."""
        if park and not self.at_park():
            self.api.move_to_start()
        self.world.forward()
        image = self.render_top() if self.needs_image else None
        dets = self.detector.detect(image)
        z = self.cfg.scene.cube_edge / 2
        out = []
        for d in dets:
            if d.world_xy is None:
                d.to_world(self.camera, z)
            if self.in_pick_area(d.world_xy):
                out.append(d)
        self._scan_count += 1
        scan = Scan(f"scan-{self._scan_count}", out, self.world.time, image)
        self.last_scan = scan
        self.placements_since_scan = 0
        self._record_localization(out)
        self.note(f"{scan.scan_id}: {len(out)} parts detected ({self.detector.name})")
        return scan

    def _record_localization(self, dets: list[Detection]) -> None:
        """Metrics only: distance from each detection to the nearest true cube."""
        truth = [
            c.pos[:2]
            for c in self.world.cubes()
            if self.in_pick_area(tuple(c.pos[:2])) and c.pos[2] < 0.1
        ]
        for d in dets:
            if not truth:
                break
            dist = min(float(np.hypot(*(np.array(d.world_xy) - t))) for t in truth)
            if dist < 0.03:
                self.localization_errors_mm.append(dist * 1e3)

    # ------------------------------------------------------------ one part
    def pick_to_bin(
        self, det: Detection, bin_name: str, attempt: int = 1, verify: bool = True
    ) -> PickRecord:
        """Pick ``det`` and place it in the next free slot of ``bin_name``; verify by re-scan."""
        slot: Slot = self.slots.allocate(bin_name)
        t0 = self.world.time
        before = self.bin_count(bin_name)
        self.note(
            f"pick {det.class_name} at ({det.world_xy[0]:.3f}, {det.world_xy[1]:.3f})"
            f" -> slot {slot.label}"
        )
        try:
            res = self.api.pick_and_place(
                det.world_xy,
                (slot.x, slot.y),
                slot_label=f"slot {slot.label}",
                part_label=det.class_name,
            )
        except (SafetyViolation, MotionError) as exc:
            self.slots.release(slot)
            if self.world.attached is not None:
                raise
            return PickRecord(
                det.class_name,
                det.confidence,
                det.world_xy,
                bin_name,
                slot.label,
                False,
                False,
                attempt,
                f"refused: {exc}",
                self.world.time,
                self.world.time - t0,
            )
        if not res.success:
            self.slots.release(slot)
            return PickRecord(
                det.class_name,
                det.confidence,
                det.world_xy,
                bin_name,
                slot.label,
                False,
                False,
                attempt,
                res.reason,
                self.world.time,
                self.world.time - t0,
            )
        self.placements_since_scan += 1
        verified, reason = True, "placed"
        if verify:
            scan = self.scan()
            still_there = any(
                d.class_name == det.class_name
                and np.hypot(d.world_xy[0] - det.world_xy[0], d.world_xy[1] - det.world_xy[1])
                < 0.02
                for d in scan.detections
            )
            arrived = self.bin_count(bin_name) == before + 1
            verified = arrived and not still_there
            if not verified:
                reason = "verify failed: " + (
                    "part still in pick area" if still_there else "part not in bin"
                )
        self.note(f"{'verified' if verified else 'NOT verified'}: {det.class_name} -> {slot.label}")
        return PickRecord(
            det.class_name,
            det.confidence,
            det.world_xy,
            bin_name,
            slot.label,
            True,
            verified,
            attempt,
            reason,
            self.world.time,
            self.world.time - t0,
        )

    # ------------------------------------------------------------ auto cycle
    def run_auto(
        self,
        max_cycles: int = 40,
        retries: int = 2,
        should_stop: Callable[[], bool] | None = None,
    ) -> SortReport:
        """Fully automatic sorting until no sortable part remains (or ``should_stop()``)."""
        report = SortReport(self.world.seed, self.detector.name)
        wall0 = time.perf_counter()
        attempts: list[tuple[str, tuple[float, float], int]] = []  # (class, xy, attempts)
        given_up: list[tuple[str, tuple[float, float]]] = []
        if not self.api.engaged:
            self.api.engage()
        try:
            scan = self.scan()
            for _ in range(max_cycles):
                if should_stop is not None and should_stop():
                    report.stop_reason = "stopped by the operator"
                    break
                candidates = []
                for det in scan.detections:
                    near_given_up = any(
                        c == det.class_name
                        and np.hypot(det.world_xy[0] - xy[0], det.world_xy[1] - xy[1]) < 0.02
                        for c, xy in given_up
                    )
                    target = self.rules.bin_for(det)
                    if near_given_up:
                        continue
                    if target is None:
                        self._flag(report, det, self.rules.why_not(det))
                        continue
                    if self.slots.free(target) == 0:
                        self._flag(report, det, f"bin {target} full")
                        continue
                    candidates.append((det, target))
                if not candidates:
                    report.completed = True
                    break
                tcp = self.api.get_tool_position().pos
                det, target = min(
                    candidates,
                    key=lambda c: np.hypot(c[0].world_xy[0] - tcp[0], c[0].world_xy[1] - tcp[1]),
                )
                n_prev = next(
                    (
                        n
                        for c, xy, n in attempts
                        if c == det.class_name
                        and np.hypot(det.world_xy[0] - xy[0], det.world_xy[1] - xy[1]) < 0.02
                    ),
                    0,
                )
                rec = self.pick_to_bin(det, target, attempt=n_prev + 1)
                report.picks.append(rec)
                if rec.reason.startswith("refused"):
                    report.refused.append(rec.reason)
                if not rec.verified:
                    attempts = [
                        a
                        for a in attempts
                        if not (
                            a[0] == det.class_name
                            and np.hypot(det.world_xy[0] - a[1][0], det.world_xy[1] - a[1][1])
                            < 0.02
                        )
                    ]
                    attempts.append((det.class_name, det.world_xy, n_prev + 1))
                    if n_prev + 1 > retries or rec.reason.startswith("refused"):
                        given_up.append((det.class_name, det.world_xy))
                        self._flag(
                            report, det, f"gave up after {n_prev + 1} attempts: {rec.reason}"
                        )
                scan = (
                    self.last_scan
                    if (self.last_scan and self.placements_since_scan == 0)
                    else self.scan()
                )
            else:
                report.stop_reason = "max cycles reached"
        except ProtectiveStop as exc:
            report.protective_stops += 1
            report.stop_reason = f"protective stop: {exc}"
            self.note(report.stop_reason)
        except (CobotError, BinFullError) as exc:
            report.stop_reason = f"stopped: {exc}"
            self.note(report.stop_reason)
        report.sim_time = self.world.time
        report.wall_time = time.perf_counter() - wall0
        report.scans = self._scan_count
        report.localization_errors_mm = list(self.localization_errors_mm)
        guard = self.api.guard
        if guard is not None:
            report.unplanned_contacts = guard.unplanned_contacts
        if report.flagged:
            self.api.warning = f"{len(report.flagged)} part(s) need the operator"
            self.api._lamp(reason="flagged parts")
        return report

    def _flag(self, report: SortReport, det: Detection, reason: str) -> None:
        key = (det.class_name, round(det.world_xy[0], 2), round(det.world_xy[1], 2))
        if any(f["key"] == key for f in report.flagged):
            return
        report.flagged.append(
            {
                "key": key,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "world_xy": det.world_xy,
                "reason": reason,
            }
        )
        self.note(
            f"flagged {det.class_name} at ({det.world_xy[0]:.3f}, {det.world_xy[1]:.3f}): {reason}"
        )


# ---------------------------------------------------------------- factory
def make_detector(name: str, world: SimWorld, camera: CameraModel) -> Detector:
    if name == "oracle":
        from cobotsort.perception.oracle import OracleDetector

        return OracleDetector(world, camera)
    if name == "hsv":
        from cobotsort.perception.hsv import HSVDetector

        return HSVDetector()
    if name == "yolo":
        from cobotsort.perception.yolo import YoloDetector

        return YoloDetector(world.cfg.vision.yolo_weights)
    raise ValueError(f"unknown detector {name!r} (oracle, hsv, yolo)")


def build_cell(
    detector: str | Detector = "oracle",
    seed: int | None = 0,
    cfg: CellConfig | None = None,
    guard: bool = True,
    world: SimWorld | None = None,
    renderer=None,
) -> SortingCell:
    """A ready cell: world reset to ``seed``, API connected (not engaged), guard attached.

    ``detector`` is a name (oracle, hsv, yolo) or an instance to reuse; pass ``world`` and
    ``renderer`` to reuse them across episodes.
    """
    cfg = cfg or CellConfig()
    world = world or SimWorld(cfg)
    world.reset(seed=seed)
    api = make_api(world, cfg=cfg)
    if guard:
        from cobotsort.safety.guard import SafetyGuard

        api.guard = SafetyGuard(world, api.ik, cfg.safety)
    world.forward()
    w, h = cfg.scene.top_cam_size
    camera = CameraModel.from_sim(world.model, world.data, "top_cam", w, h)
    det = make_detector(detector, world, camera) if isinstance(detector, str) else detector
    return SortingCell(world, api, det, renderer=renderer)
