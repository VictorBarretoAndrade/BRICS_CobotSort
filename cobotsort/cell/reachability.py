"""Startup self-check: every pick position and bin slot must be reachable tool-down."""

from __future__ import annotations

import numpy as np

from cobotsort.cell.slots import SlotAllocator
from cobotsort.kinematics.ik import IKSolver, Pose
from cobotsort.kinematics.manipulability import manipulability
from cobotsort.sim.world import SimWorld


def reachability_targets(world: SimWorld) -> list[tuple[str, Pose]]:
    sc, mc = world.cfg.scene, world.cfg.motion
    top = sc.cube_edge
    out = []
    for x in np.linspace(sc.pick_x[0], sc.pick_x[1], 4):
        for y in np.linspace(sc.pick_y[0], sc.pick_y[1], 5):
            out.append((f"pick ({x:.2f},{y:.2f}) contact", Pose.tool_down(x, y, top + 0.001)))
            out.append(
                (f"pick ({x:.2f},{y:.2f}) approach", Pose.tool_down(x, y, top + mc.approach_height))
            )
    floor = sc.bin_floor_thickness
    for slots in SlotAllocator(sc).slots.values():
        for s in slots:
            out.append(
                (
                    f"slot {s.label} place",
                    Pose.tool_down(s.x, s.y, floor + top + mc.place_clearance),
                )
            )
            out.append(
                (
                    f"slot {s.label} approach",
                    Pose.tool_down(s.x, s.y, floor + top + mc.approach_height),
                )
            )
    return out


def reachability_report(world: SimWorld, w_min: float | None = None) -> tuple[bool, list[str]]:
    """IK from home for every target; reachable means solved with manipulability >= w_min."""
    w_min = world.cfg.safety.manip_block if w_min is None else w_min
    ik = IKSolver(world.info)
    home = world.info.home_q
    ok = True
    lines = []
    worst = (np.inf, "")
    for name, pose in reachability_targets(world):
        sol = ik.closest(pose, home)
        if sol is None:
            ok = False
            lines.append(f"  UNREACHABLE {name}")
            continue
        w = manipulability(ik.jacobian(sol.q))
        if w < worst[0]:
            worst = (w, name)
        if w < w_min:
            ok = False
            lines.append(f"  NEAR SINGULAR {name}: w = {w:.4f}")
    n = len(reachability_targets(world))
    lines.insert(
        0,
        f"reachability: {n} targets, {'all reachable' if ok else 'FAILED'}; "
        f"lowest manipulability {worst[0]:.4f} at {worst[1]}",
    )
    return ok, lines
