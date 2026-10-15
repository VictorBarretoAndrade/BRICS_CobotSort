"""Kinematics report: DH vs MuJoCo for both DH tables, and IK accuracy on reachable targets."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

from cobotsort.config import ROOT
from cobotsort.kinematics.dh import (
    UR5E_MENAGERIE,
    UR5E_OFFICIAL,
    DHModel,
    pose_error,
    pose_to_matrix,
)
from cobotsort.kinematics.ik import IKSolver, Pose
from cobotsort.kinematics.manipulability import manipulability
from cobotsort.sim.world import SimWorld


def _flange(world: SimWorld, q: np.ndarray) -> np.ndarray:
    world.data.qpos[world.info.robot_qpos_adr] = q
    mujoco.mj_kinematics(world.model, world.data)
    s = world.info.flange_site
    return pose_to_matrix(
        world.data.site_xpos[s].copy(), world.data.site_xmat[s].reshape(3, 3).copy()
    )


def _random_q(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.uniform(-np.pi, np.pi, (n, 6))
    q[:, 2] = rng.uniform(-2.8, 2.8, n)
    return q


def run(out: Path | None = None, seed: int = 0) -> str:
    out = out or ROOT / "results" / "kinematics.md"
    world = SimWorld()
    rng = np.random.default_rng(seed)
    fit_q, test_q = _random_q(rng, 30), _random_q(rng, 50)
    fit_p = [_flange(world, q) for q in fit_q]
    test_p = [_flange(world, q) for q in test_q]
    lines = [
        "# Kinematics report",
        "",
        f"Command: `cobotsort kinematics-report` (seed {seed}).",
        "",
        "## Forward kinematics: DH vs MuJoCo",
        "",
        "Constant base and flange transforms are fitted once on 30 random configurations; the",
        "errors below are on 50 other random configurations (flange pose).",
        "",
        "| DH table | Position error max (mm) | Position error RMS (mm)"
        " | Rotation error max (deg) |",
        "|---|---|---|---|",
    ]
    for table in (UR5E_MENAGERIE, UR5E_OFFICIAL):
        model = DHModel(table)
        model.fit(fit_q, fit_p)
        pairs = zip(test_q, test_p, strict=True)
        errs = np.array([pose_error(model.fk(q), p) for q, p in pairs])
        lines.append(
            f"| {table.name} | {errs[:, 0].max() * 1e3:.4f} | "
            f"{np.sqrt(np.mean(errs[:, 0] ** 2)) * 1e3:.4f} | {errs[:, 1].max():.4f} |"
        )
    ik = IKSolver(world.info)
    targets = []
    while len(targets) < 100:
        q = _random_q(rng, 1)[0]
        if manipulability(ik.jacobian(q)) >= 0.01:
            targets.append(Pose(*ik.site_pose(q)))
    t0 = time.perf_counter()
    sols = [ik.solve_all(t, world.info.home_q, rng=rng) for t in targets]
    dt = (time.perf_counter() - t0) / len(targets)
    solved = [s[0] for s in sols if s]
    n_solutions = [len(s) for s in sols]
    pos = np.array([s.pos_error for s in solved]) * 1e3
    rot = np.degrees([s.rot_error for s in solved])
    lines += [
        "",
        "## Inverse kinematics (damped least squares, multi-seed)",
        "",
        "100 reachable targets: random joint vectors with manipulability ≥ 0.01, mapped to a",
        "TCP pose by forward kinematics, then solved from the home configuration.",
        "",
        "| Solved | Position error max (mm) | Rotation error max (deg)"
        " | Distinct solutions per target (mean, min-max) | Time per target, all seeds (ms) |",
        "|---|---|---|---|---|",
        f"| {len(solved)}/100 | {pos.max():.4f} | {rot.max():.4f} | {np.mean(n_solutions):.1f}"
        f" ({min(n_solutions)}-{max(n_solutions)}) | {dt * 1e3:.1f} |",
        "",
        "An unreachable target (TCP 1.5 m from the base) returns no solution: "
        f"{len(ik.solve_all(Pose.tool_down(1.5, 0.0, 0.1), world.info.home_q))} solutions.",
    ]
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
