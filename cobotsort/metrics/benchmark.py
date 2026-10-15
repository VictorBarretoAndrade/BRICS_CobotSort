"""Benchmark with the competition criteria, in priority order:
safety > degree of automation > precision > speed.

Each detector runs the fully automatic cycle (module B) on the same seeds. Ground truth from
the simulator is used only to score the episode, never to drive the robot (except for the
oracle detector, which is the upper bound by definition).
"""

from __future__ import annotations

import csv
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from cobotsort.cell.slots import SlotAllocator
from cobotsort.cell.sorter import SortingCell, build_cell, make_detector
from cobotsort.config import ROOT, CellConfig
from cobotsort.perception.camera import CameraModel
from cobotsort.safety.adversarial import run_adversarial_suite
from cobotsort.sim.render import CellRenderer
from cobotsort.sim.world import SimWorld

log = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    detector: str
    seed: int
    cubes: int
    correct: int
    wrong_bin: int
    not_in_bin: int
    flagged: int
    unplanned_contacts: int
    protective_stops: int
    completed: bool
    sim_time: float
    wall_time: float
    placement_errors_mm: list[float] = field(default_factory=list)
    localization_errors_mm: list[float] = field(default_factory=list)


def score_episode(cell: SortingCell, report, seed: int) -> EpisodeResult:
    world = cell.world
    slots = SlotAllocator(world.cfg.scene).slots
    correct = wrong = left = 0
    placement = []
    for cube in world.cubes():
        where = world.bin_of(cube.pos)
        if where is None:
            left += 1
            continue
        if where == cell.rules.mapping.get(cube.class_name):
            correct += 1
        else:
            wrong += 1
        nearest = min(np.hypot(s.x - cube.pos[0], s.y - cube.pos[1]) for s in slots[where])
        placement.append(float(nearest) * 1e3)
    cubes = len(world.cubes())
    return EpisodeResult(
        detector=cell.detector.name,
        seed=seed,
        cubes=cubes,
        correct=correct,
        wrong_bin=wrong,
        not_in_bin=left,
        flagged=len(report.flagged),
        unplanned_contacts=report.unplanned_contacts,
        protective_stops=report.protective_stops,
        # End to end: the cycle ran to its end on its own and every cube is in its bin.
        completed=bool(report.completed and report.protective_stops == 0 and correct == cubes),
        sim_time=report.sim_time,
        wall_time=report.wall_time,
        placement_errors_mm=placement,
        localization_errors_mm=list(report.localization_errors_mm),
    )


def run_detector(name: str, seeds: list[int], cfg: CellConfig) -> list[EpisodeResult]:
    world = SimWorld(cfg)
    renderer = CellRenderer(world.model)
    world.forward()
    w, h = cfg.scene.top_cam_size
    detector = make_detector(
        name, world, CameraModel.from_sim(world.model, world.data, "top_cam", w, h)
    )
    out = []
    for seed in seeds:
        cell = build_cell(detector, seed=seed, cfg=cfg, world=world, renderer=renderer)
        t0 = time.perf_counter()
        report = cell.run_auto()
        report.wall_time = time.perf_counter() - t0
        ep = score_episode(cell, report, seed)
        out.append(ep)
        log.info(
            "%s seed %d: %d/%d correct, stops %d, sim %.0f s, wall %.1f s",
            name,
            seed,
            ep.correct,
            ep.cubes,
            ep.protective_stops,
            ep.sim_time,
            ep.wall_time,
        )
    renderer.close()
    return out


def summarize(name: str, eps: list[EpisodeResult], adversarial: tuple[int, int]) -> dict:
    cubes = sum(e.cubes for e in eps)
    placed = sum(e.correct + e.wrong_bin for e in eps)
    place = np.concatenate([e.placement_errors_mm for e in eps]) if placed else np.zeros(0)
    loc = np.concatenate([e.localization_errors_mm for e in eps]) if eps else np.zeros(0)
    sorted_cubes = sum(e.correct for e in eps)
    return {
        "detector": name,
        "episodes": len(eps),
        "cubes": cubes,
        # 1. Safety
        "unplanned_contacts": sum(e.unplanned_contacts for e in eps),
        "protective_stops": sum(e.protective_stops for e in eps),
        "adversarial_blocked": f"{adversarial[0]}/{adversarial[1]}",
        # 2. Automation
        "cubes_sorted_pct": 100.0 * sorted_cubes / max(cubes, 1),
        "episodes_end_to_end_pct": 100.0 * sum(e.completed for e in eps) / max(len(eps), 1),
        "flagged_for_operator": sum(e.flagged for e in eps),
        # 3. Precision
        "placement_error_mean_mm": float(place.mean()) if place.size else float("nan"),
        "placement_error_p95_mm": float(np.percentile(place, 95)) if place.size else float("nan"),
        "wrong_bin_rate_pct": 100.0 * sum(e.wrong_bin for e in eps) / max(placed, 1),
        "localization_error_mean_mm": float(loc.mean()) if loc.size else float("nan"),
        # 4. Speed
        "sim_s_per_cube": sum(e.sim_time for e in eps) / max(sorted_cubes, 1),
        "sim_s_per_episode": float(np.mean([e.sim_time for e in eps])),
        "wall_s_per_episode": float(np.mean([e.wall_time for e in eps])),
    }


COLUMNS = [
    ("detector", "Detector", "{}"),
    ("unplanned_contacts", "Unplanned contacts", "{}"),
    ("protective_stops", "Protective stops", "{}"),
    ("adversarial_blocked", "Adversarial blocked", "{}"),
    ("cubes_sorted_pct", "Cubes sorted w/o operator (%)", "{:.1f}"),
    ("episodes_end_to_end_pct", "Episodes end to end (%)", "{:.0f}"),
    ("placement_error_mean_mm", "Placement error mean (mm)", "{:.2f}"),
    ("placement_error_p95_mm", "Placement error p95 (mm)", "{:.2f}"),
    ("wrong_bin_rate_pct", "Wrong-bin rate (%)", "{:.1f}"),
    ("localization_error_mean_mm", "Perception loc. error (mm)", "{:.2f}"),
    ("sim_s_per_cube", "Sim s / cube", "{:.1f}"),
    ("sim_s_per_episode", "Sim s / episode", "{:.1f}"),
]


def write_results(
    rows: list[dict], episodes: list[EpisodeResult], args_text: str, out: Path
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "benchmark_episodes.csv").open("w", newline="") as f:
        fields = [k for k in asdict(episodes[0]) if not k.endswith("_mm")]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in episodes:
            writer.writerow({k: v for k, v in asdict(e).items() if k in fields})
    cpu = platform.processor() or platform.machine()
    lines = [
        "# Benchmark",
        "",
        f"Command: `{args_text}`",
        "",
        "Columns follow the competition's judging order: **safety > degree of automation >"
        " precision > speed**. Every episode runs the fully automatic cycle (module B): park,"
        " scan, detect, nearest-first pick-and-place, re-scan to verify, until no sortable cube"
        " remains. Collaborative speed limits (0.5 rad/s per joint, 0.25 m/s TCP). Simulation"
        " only.",
        "",
        "| " + " | ".join(c[1] for c in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for r in rows:
        lines.append("| " + " | ".join(fmt.format(r[k]) for k, _, fmt in COLUMNS) + " |")
    lines += [
        "",
        "| Detector | Episodes | Cubes | Flagged for the operator | Wall-clock s / episode"
        f" (author's CPU: {cpu}) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['detector']} | {r['episodes']} | {r['cubes']} | {r['flagged_for_operator']}"
            f" | {r['wall_s_per_episode']:.1f} |"
        )
    lines += [
        "",
        "Definitions:",
        "",
        "- *Unplanned contacts / protective stops*: counted by the runtime monitor during the"
        " episodes.",
        "- *Adversarial blocked*: the 10 unsafe requests of `tests/test_safety.py`, run against"
        " this detector's cell.",
        "- *Cubes sorted w/o operator*: cubes that ended in their rule's bin with no human input."
        " Cubes left in place because their bin was full (4 slots per bin) count as not sorted.",
        "- *Episodes end to end*: the cycle ended on its own, with no protective stop, and every"
        " cube is in its bin.",
        "- *Placement error*: distance from each cube in a bin to the nearest slot center.",
        "- *Perception localization error*: detection after `pixel_to_world` vs. the true cube"
        " position, at every scan of the episode.",
        "- *Sim s / cube*: total simulated time divided by the number of sorted cubes (scans and"
        " verification included).",
        "- The agent provider does not drive these episodes; agent task success per provider is in"
        " `results/agent_eval.md`.",
        "",
        "Per-episode rows: `results/benchmark_episodes.csv`.",
    ]
    (out / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    seeds: int = 20,
    detectors: list[str] | None = None,
    provider: str = "mock",
    out: Path | None = None,
    cfg: CellConfig | None = None,
) -> list[dict]:
    detectors = detectors or ["oracle", "hsv", "yolo"]
    cfg = cfg or CellConfig()
    out = out or ROOT / "results"
    seed_list = list(range(seeds))
    rows, all_eps = [], []
    for name in detectors:
        adv = run_adversarial_suite(name, cfg=cfg)
        blocked = (sum(r.blocked for r in adv), len(adv))
        eps = run_detector(name, seed_list, cfg)
        all_eps += eps
        rows.append(summarize(name, eps, blocked))
    args_text = (
        f"cobotsort benchmark --seeds {seeds} --detector {' '.join(detectors)}"
        f" --provider {provider}"
    )
    write_results(rows, all_eps, args_text, out)
    return rows
