"""Demo video: the operator agent runs a work order while the perspective camera records.

Overlays: signal lamp, current robot command, an inset of the top camera with the YOLO boxes of
the latest scan, the last 3 agent/cell log lines, a speed label, and a final card with the
metrics of this run. Simulated time is compressed with labeled speed factors so a full
10-part order fits in about a minute.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from cobotsort.agent.operator import OperatorAgent
from cobotsort.cell.sorter import SortReport, build_cell
from cobotsort.config import ROOT, CellConfig
from cobotsort.metrics.benchmark import score_episode
from cobotsort.sim.render import CellRenderer, draw_boxes, put_label
from cobotsort.sim.world import SimWorld

log = logging.getLogger(__name__)

LAMP_BGR = {"GREEN": (40, 220, 60), "YELLOW": (255, 200, 20), "RED": (255, 40, 30)}  # RGB


@dataclass
class RecordConfig:
    fps: int = 30
    size: tuple[int, int] = (1280, 720)
    transfer_speed: float = 6.0  # MoveJ, parking, scanning
    approach_speed: float = 3.0  # MoveL, vacuum, waits
    title_s: float = 2.5
    final_s: float = 5.0
    gif_start_s: float = 3.0  # video time where the GIF window starts
    gif_seconds: float = 10.0
    gif_width: int = 640
    gif_fps: int = 10


class Recorder:
    """Renders frames from a physics step callback at a variable speed-up."""

    def __init__(self, world: SimWorld, cell, out: Path, cfg: RecordConfig) -> None:
        self.world = world
        self.cell = cell
        self.cfg = cfg
        self.renderer = CellRenderer(world.model)
        self.writer = imageio.get_writer(
            str(out),
            fps=cfg.fps,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
            macro_block_size=8,
        )
        self.next_t = 0.0
        self.frames = 0
        self.lines: deque[str] = deque(maxlen=3)
        self.inset: np.ndarray | None = None
        self._inset_scan: str | None = None
        self.gif_frames: list[np.ndarray] = []
        self.order = ""

    # ------------------------------------------------------------ speed
    @property
    def speed(self) -> float:
        cmd = self.cell.api.command
        if cmd.startswith(("move_to_point_j", "move_to_start")):
            return self.cfg.transfer_speed
        return self.cfg.approach_speed

    def log_line(self, text: str) -> None:
        self.lines.append(text if len(text) <= 95 else text[:92] + "...")

    # ------------------------------------------------------------ frames
    def on_step(self, world: SimWorld) -> None:
        if world.time + 1e-9 >= self.next_t:
            self.write(self.compose(self.speed))
            self.next_t += self.speed / self.cfg.fps

    def write(self, frame: np.ndarray) -> None:
        self.writer.append_data(frame)
        t_video = self.frames / self.cfg.fps
        g0 = self.cfg.gif_start_s
        step = max(1, self.cfg.fps // self.cfg.gif_fps)
        if g0 <= t_video < g0 + self.cfg.gif_seconds and self.frames % step == 0:
            h = int(frame.shape[0] * self.cfg.gif_width / frame.shape[1])
            self.gif_frames.append(
                cv2.resize(frame, (self.cfg.gif_width, h), interpolation=cv2.INTER_AREA)
            )
        self.frames += 1

    def _update_inset(self) -> None:
        scan = self.cell.last_scan
        if scan is None or scan.image is None or scan.scan_id == self._inset_scan:
            return
        img = draw_boxes(scan.image, scan.detections)
        self.inset = cv2.resize(img, (384, 288), interpolation=cv2.INTER_AREA)
        self._inset_scan = scan.scan_id

    def compose(self, speed: float | None) -> np.ndarray:
        w, h = self.cfg.size
        frame = self.renderer.rgb(self.world.data, "persp_cam", w, h)
        api = self.cell.api
        # Header.
        put_label(frame, "CobotSort AI  |  simulation (MuJoCo, UR5e stand-in)", (20, 38), 0.8)
        lamp = api.lamp.state.value
        cv2.circle(frame, (36, 80), 14, LAMP_BGR[lamp], -1, cv2.LINE_AA)
        cv2.circle(frame, (36, 80), 14, (255, 255, 255), 2, cv2.LINE_AA)
        put_label(frame, f"lamp {lamp}", (60, 89), 0.7)
        put_label(frame, api.command, (20, 132), 0.7, color=(255, 240, 160))
        if speed is not None:
            put_label(frame, f"{speed:g}x speed", (20, 172), 0.7, color=(160, 220, 255))
        # Top camera inset.
        self._update_inset()
        if self.inset is not None:
            x0, y0 = w - 384 - 16, 16
            frame[y0 : y0 + 288, x0 : x0 + 384] = self.inset
            cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + 384, y0 + 288), (255, 255, 255), 2)
            put_label(
                frame,
                f"top camera + YOLO ({self._inset_scan})",
                (x0 + 8, y0 + 280),
                0.5,
                thickness=1,
            )
        # Agent order and log lines.
        if self.order:
            put_label(frame, f"order: {self.order}", (20, h - 118), 0.6, color=(255, 255, 255))
        for i, line in enumerate(self.lines):
            put_label(frame, line, (20, h - 82 + 30 * i), 0.55, color=(220, 220, 220), thickness=1)
        # Counters.
        placed = sum(1 for c in self.world.cubes() if self.world.bin_of(c.pos))
        total = len(self.world.cubes())
        put_label(
            frame,
            f"parts in bins {placed}/{total}   sim t = {self.world.time:5.1f} s",
            (w - 430, h - 22),
            0.6,
        )
        return frame

    def card(self, frame: np.ndarray, title: str, rows: list[str], seconds: float) -> None:
        out = (frame * 0.25).astype(np.uint8)
        put_label(out, title, (80, 140), 1.2, bg=None, thickness=3)
        for i, row in enumerate(rows):
            put_label(out, row, (80, 220 + 52 * i), 0.85, bg=None, thickness=2)
        for _ in range(int(seconds * self.cfg.fps)):
            self.write(out)

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    images = [
        Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


def record(
    seed: int = 7,
    detector: str = "yolo",
    order: str = "Separe as peças por cor e mande as defeituosas para o rejeito.",
    out: Path | None = None,
    provider: str | None = None,
    rec_cfg: RecordConfig | None = None,
    cfg: CellConfig | None = None,
) -> dict:
    rec_cfg = rec_cfg or RecordConfig()
    cfg = cfg or CellConfig()
    out = out or ROOT / "media" / "demo.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    cell = build_cell(detector, seed=seed, cfg=cfg)
    world = cell.world
    rec = Recorder(world, cell, out, rec_cfg)
    rec.order = order
    agent = OperatorAgent(cell, cfg.agent, provider=provider)
    agent.listeners.append(rec.log_line)
    cell.listeners.append(rec.log_line)
    # Title card on the first frame of the scene.
    first = rec.renderer.rgb(world.data, "persp_cam", *rec_cfg.size)
    rec.card(
        first,
        "CobotSort AI",
        [
            "Simulated cobot sorting cell: UR5e + vacuum, YOLO11n, safety layer, AI operator.",
            f'Work order: "{order}"',
            "Simulation only. Speed-ups are labeled on screen.",
        ],
        rec_cfg.title_s,
    )
    cell.api.backend.step_callbacks.append(rec.on_step)
    result = agent.run(order)
    cell.api.backend.step_callbacks.remove(rec.on_step)
    for _ in range(int(rec_cfg.fps * 1.0)):  # hold the last frame for a second
        rec.write(rec.compose(None))
    report = _metrics(cell, result, seed)
    report["wall_time_s"] = round(time.perf_counter() - wall0, 1)
    rows = [
        f"Parts in the correct bin: {report['correct']}/{report['cubes']}"
        f"   wrong bin: {report['wrong_bin']}",
        f"Unplanned contacts: {report['unplanned_contacts']}"
        f"   protective stops: {report['protective_stops']}",
        f"Placement error vs slot center: mean {report['placement_error_mean_mm']:.2f} mm",
        f"YOLO localization error: mean {report['localization_error_mean_mm']:.2f} mm",
        f"Agent: {report['tool_calls']} tool calls ({report['provider']}), "
        f"simulated time {report['sim_time_s']:.0f} s",
    ]
    last = rec.renderer.rgb(world.data, "persp_cam", *rec_cfg.size)
    rec.card(last, f"This run (seed {seed}, {detector})", rows, rec_cfg.final_s)
    rec.close()
    gif = out.with_suffix(".gif")
    save_gif(rec.gif_frames, gif, rec_cfg.gif_fps)
    report.update(
        video=out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out),
        video_seconds=round(rec.frames / rec_cfg.fps, 1),
        video_mb=round(out.stat().st_size / 1e6, 2),
        gif_mb=round(gif.stat().st_size / 1e6, 2),
        answer=result.final_text,
        order=order,
        command=(
            f'cobotsort record --seed {seed} --detector {detector} --order "{order}" '
            f"--out {out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out}"
        ),
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "demo_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _metrics(cell, result, seed: int) -> dict:
    run = SortReport(
        seed,
        cell.detector.name,
        unplanned_contacts=cell.api.guard.unplanned_contacts if cell.api.guard else 0,
        protective_stops=sum(1 for e in cell.api.safety_events if "protective" in e),
        completed=True,
        sim_time=cell.world.time,
        localization_errors_mm=list(cell.localization_errors_mm),
    )
    ep = score_episode(cell, run, seed)
    place = ep.placement_errors_mm
    loc = cell.localization_errors_mm
    return {
        "seed": seed,
        "detector": cell.detector.name,
        "provider": result.provider_used,
        "cubes": ep.cubes,
        "correct": ep.correct,
        "wrong_bin": ep.wrong_bin,
        "unplanned_contacts": ep.unplanned_contacts,
        "protective_stops": ep.protective_stops,
        "placement_error_mean_mm": float(np.mean(place)) if place else float("nan"),
        "localization_error_mean_mm": float(np.mean(loc)) if loc else float("nan"),
        "tool_calls": len(result.calls),
        "sim_time_s": round(cell.world.time, 1),
    }
