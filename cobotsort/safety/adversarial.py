"""The 10 unsafe requests of the adversarial test set (SPEC 6.5).

Shared by ``tests/test_safety.py`` and the benchmark ("adversarial requests blocked x/10").
A case counts as blocked when the expected refusal is raised and neither the robot nor any
part moved.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cobotsort.agent.guardrails import GuardrailError, Guardrails
from cobotsort.cell.slots import BinFullError
from cobotsort.cell.sorter import SortingCell, build_cell
from cobotsort.config import AgentConfig, CellConfig
from cobotsort.kinematics.ik import Pose
from cobotsort.robot.api import SafetyViolation
from cobotsort.robot.backend import NotEngagedError, ProtectiveStop

log = logging.getLogger(__name__)


@dataclass
class AdversarialResult:
    case: str
    blocked: bool
    detail: str


def _motion(cell: SortingCell, fn: Callable[[], object], expect: tuple, must_mention: str = ""):
    """Run ``fn`` and require one of ``expect`` with nothing moving."""
    q0 = cell.world.q
    cubes0 = np.array([c.pos for c in cell.world.cubes()])
    try:
        fn()
    except expect as exc:
        moved = np.max(np.abs(cell.world.q - q0)) > 1e-4
        parts = np.array([c.pos for c in cell.world.cubes()])
        parts_moved = np.max(np.abs(parts - cubes0)) > 1e-3
        text = str(exc)
        ok = not moved and not parts_moved and must_mention.lower() in text.lower()
        return ok, f"{type(exc).__name__}: {text}"
    return False, "not blocked"


def _case_below_table(cell):
    return _motion(
        cell,
        lambda: cell.api.move_to_point_j(Pose.tool_down(0.45, 0.0, -0.05)),
        (SafetyViolation,),
        "workspace",
    )


def _case_outside_workspace(cell):
    return _motion(
        cell,
        lambda: cell.api.move_to_point_j(Pose.tool_down(0.0, -0.70, 0.30)),
        (SafetyViolation,),
        "workspace",
    )


def _case_bin_wall(cell):
    inside = cell.api.ik.closest(Pose.tool_down(0.15, 0.45, 0.025), cell.world.q)
    cell.world.set_robot_q(inside.q)
    cell.api.backend.connect()
    return _motion(
        cell,
        lambda: cell.api.move_to_point_l(Pose.tool_down(0.15, 0.30, 0.025)),
        (SafetyViolation,),
        "collision",
    )


def _case_stretched(cell):
    q = cell.world.q.copy()
    q[2] = 0.03  # elbow almost straight: the arm is stretched
    return _motion(cell, lambda: cell.api.move_to_point_j(q), (SafetyViolation,), "manipulability")


def _case_speed(cell):
    return _motion(
        cell,
        lambda: cell.api.move_to_point_j(Pose.tool_down(0.45, 0.0, 0.19), speed=2.0),
        (SafetyViolation,),
        "speed",
    )


def _case_not_engaged(cell):
    cell.api.disengage()
    return _motion(
        cell, lambda: cell.api.move_to_point_j(Pose.tool_down(0.45, 0.0, 0.19)), (NotEngagedError,)
    )


def _case_estop(cell):
    cell.api.emergency_stop()

    def attempt():
        cell.api.move_to_point_j(Pose.tool_down(0.45, 0.0, 0.19))

    ok, detail = _motion(cell, attempt, (ProtectiveStop,), "E-stop")
    try:
        cell.api.engage()
        return False, "engage() succeeded during E-stop"
    except Exception:  # noqa: BLE001 - any refusal is fine here
        return ok and cell.api.lamp.state.value == "RED", detail


def _case_full_bin(cell):
    scan = cell.scan()
    for _ in range(4):
        cell.slots.allocate("REJECT")
    det = scan.detections[0]
    return _motion(cell, lambda: cell.pick_to_bin(det, "REJECT"), (BinFullError,), "full")


def _case_unknown_object(cell):
    rails = Guardrails(cell, AgentConfig())
    scan = cell.scan()
    return _motion(
        cell, lambda: rails.validate_pick(scan.scan_id, 99, "A"), (GuardrailError,), "not in"
    )


def _case_stale_scan(cell):
    rails = Guardrails(cell, AgentConfig())
    scan = cell.scan()
    det = scan.detections[0]
    cell.pick_to_bin(det, cell.rules.bin_for(det) or "REJECT", verify=False)
    cell.api.move_to_start()
    return _motion(
        cell, lambda: rails.validate_pick(scan.scan_id, 1, "A"), (GuardrailError,), "stale"
    )


CASES: dict[str, Callable[[SortingCell], tuple[bool, str]]] = {
    "target below the table": _case_below_table,
    "target outside the workspace": _case_outside_workspace,
    "path through a bin wall": _case_bin_wall,
    "nearly stretched pose": _case_stretched,
    "speed above the limit": _case_speed,
    "move while not engaged": _case_not_engaged,
    "move during E-stop": _case_estop,
    "place into a full bin": _case_full_bin,
    "unknown object id": _case_unknown_object,
    "stale scan id (agent)": _case_stale_scan,
}


def run_adversarial_suite(detector: str = "oracle", seed: int = 0, cfg: CellConfig | None = None):
    """Run all 10 cases, each on a fresh, engaged cell. Returns a list of results."""
    results = []
    for name, case in CASES.items():
        cell = build_cell(detector, seed=seed, cfg=cfg)
        cell.api.engage()
        try:
            ok, detail = case(cell)
        except Exception as exc:  # noqa: BLE001 - an unexpected error is a failed case
            ok, detail = False, f"unexpected {type(exc).__name__}: {exc}"
        results.append(AdversarialResult(name, ok, detail))
        log.info("adversarial %-30s %s  %s", name, "BLOCKED" if ok else "NOT BLOCKED", detail)
    return results
