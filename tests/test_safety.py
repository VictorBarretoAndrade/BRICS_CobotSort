"""Safety layer: the 10 adversarial requests, the runtime monitor and the signal lamp."""

from __future__ import annotations

import numpy as np
import pytest

from cobotsort.kinematics.ik import Pose
from cobotsort.robot.backend import NotEngagedError, ProtectiveStop
from cobotsort.safety.adversarial import CASES, run_adversarial_suite


@pytest.fixture(scope="module")
def adversarial():
    return {r.case: r for r in run_adversarial_suite()}


def test_suite_has_ten_cases():
    assert len(CASES) == 10


@pytest.mark.parametrize("case", list(CASES))
def test_unsafe_request_is_blocked(adversarial, case):
    result = adversarial[case]
    assert result.blocked, f"{case}: {result.detail}"


def test_stretched_pose_is_blocked_by_manipulability(adversarial):
    assert "manipulability" in adversarial["nearly stretched pose"].detail


@pytest.fixture
def guarded():
    from cobotsort.cell.sorter import build_cell

    cell = build_cell("oracle", seed=0, cfg=None)
    cell.world.reset(
        seed=0, classes=["red_cube", "blue_cube"], positions=[(0.45, 0.05), (0.40, -0.15)]
    )
    cell.api.engage()
    return cell


def test_runtime_monitor_stops_on_unplanned_contact(guarded):
    """An obstacle that appears after planning stops the robot (protective stop, lamp RED)."""
    world, api = guarded.world, guarded.api
    state = {"done": False}

    def intruder(w):
        if not state["done"] and w.time > 2.0:
            tcp, _ = w.tcp_pose()
            w._set_cube_pose(1, tcp - np.array([0.0, 0.0, 0.015]), 0.0)
            state["done"] = True

    api.backend.step_callbacks.append(intruder)
    with pytest.raises(ProtectiveStop, match="unplanned contact"):
        api.move_to_point_j(Pose.tool_down(0.50, -0.10, 0.19))
    assert api.guard.unplanned_contacts == 1
    assert api.lamp.state.value == "RED"
    assert not api.engaged and api.fault
    # The flow from the camp's GUI: reset errors, then engage again.
    api.reset_errors()
    with pytest.raises(NotEngagedError):
        api.move_to_start()
    api.engage()
    api.backend.step_callbacks.clear()
    world._set_cube_pose(1, np.array([0.40, -0.15, 0.0205]), 0.0)
    assert api.move_to_start().completed


def test_lamp_states(guarded):
    api = guarded.api
    seen = set()
    api.backend.step_callbacks.append(lambda w: seen.add(api.lamp.state.value))
    assert api.lamp.state.value == "YELLOW"  # engaged, idle
    api.move_to_point_j(Pose.tool_down(0.45, 0.0, 0.19))
    assert "GREEN" in seen  # executing
    assert api.lamp.state.value == "YELLOW"
    api.emergency_stop()
    assert api.lamp.state.value == "RED"
    api.reset_errors()
    assert api.lamp.state.value == "YELLOW"


def test_auto_cycle_runs_clean_under_the_guard(guarded):
    report = guarded.run_auto()
    assert report.completed
    assert report.protective_stops == 0 and report.unplanned_contacts == 0
    assert report.placed == 2
