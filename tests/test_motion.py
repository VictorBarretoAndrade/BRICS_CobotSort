"""MoveJ / MoveL trajectories, execution, vacuum and the pick-and-place sequence."""

from __future__ import annotations

import numpy as np
import pytest

from cobotsort.kinematics.ik import Pose
from cobotsort.motion.trajectory import MotionError


def test_movej_is_synchronized_and_within_limits(api, world):
    q0 = world.q
    q1 = api.ik.closest(Pose.tool_down(0.45, 0.0, 0.19), q0).q
    traj = api.planner.movej(q0, q1)
    cfg = world.cfg.motion
    assert np.allclose(traj.q[0], q0) and np.allclose(traj.q[-1], q1)
    assert np.allclose(traj.qd[0], 0) and np.allclose(traj.qd[-1], 0)
    # Synchronized: every moving joint is at the same fraction of its travel at all times.
    frac = (traj.q - q0) / (q1 - q0)
    moving = np.abs(q1 - q0) > 1e-3
    assert np.allclose(frac[:, moving], frac[:, moving][:, :1], atol=1e-9)
    assert traj.peak_joint_speed <= cfg.v_max * 1.001
    assert traj.peak_tcp_speed <= cfg.tcp_speed_max * 1.02


def test_movel_is_a_straight_line(api, world):
    api.move_to_point_j(Pose.tool_down(0.45, 0.0, 0.19))
    start = world.tcp_pose()[0]
    end = np.array([0.45, 0.0, 0.041])
    traj = api.planner.movel(world.q, Pose.tool_down(*end))
    d = (end - start) / np.linalg.norm(end - start)
    rel = traj.tcp - start
    off_line = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
    assert off_line.max() < 5e-4, f"TCP leaves the line by {off_line.max() * 1e3:.2f} mm"
    assert traj.peak_tcp_speed <= world.cfg.motion.tcp_speed_max * 1.02
    assert np.linalg.norm(traj.tcp[-1] - end) < 1e-4


def test_movel_refuses_wrist_flip_near_singularity(api, world):
    q0 = api.ik.closest(Pose.tool_down(0.30, 0.12, 0.40), world.info.home_q).q
    with pytest.raises(MotionError):
        api.planner.movel(q0, Pose.tool_down(-0.30, -0.12, 0.40, yaw=0.0))


def test_actuated_tracking_error_below_2mm(api, world):
    res = api.move_to_point_j(Pose.tool_down(0.50, -0.15, 0.20))
    assert res.completed
    assert res.max_tracking_error < 2e-3
    pos, rot = world.tcp_pose()
    assert np.linalg.norm(pos - [0.50, -0.15, 0.20]) < 1e-3


def test_vacuum_needs_contact_and_vertical_tool(api, world):
    api.move_to_point_j(Pose.tool_down(0.45, 0.05, 0.10))
    res = api.tool_on()
    assert res.part is None and "no part" in res.message
    api.move_to_point_l(Pose.tool_down(0.45, 0.05, 0.041), final_approach=True)
    res = api.tool_on()
    assert res.part == 0
    api.tool_off()
    tilt = Pose.tool_down(0.45, 0.05, 0.041)
    c, s = np.cos(np.radians(20)), np.sin(np.radians(20))
    tilt.rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ tilt.rot
    api.move_to_point_l(tilt, final_approach=True)
    assert api.tool_on().part is None


def test_pick_and_place_lands_in_slot(api, world):
    res = api.pick_and_place((0.45, 0.05), (0.12, 0.48), slot_label="slot A1", target_part=0)
    assert res.success
    pos, _ = world.cube_pose(0)
    assert world.bin_of(pos) == "A"
    assert np.hypot(pos[0] - 0.12, pos[1] - 0.48) < 5e-3


def test_oracle_auto_sorts_every_cube_seed_1():
    from cobotsort.cell.sorter import build_cell

    cell = build_cell("oracle", seed=1)
    report = cell.run_auto()
    assert report.completed and report.protective_stops == 0
    for cube in cell.world.cubes():
        assert cell.world.bin_of(cube.pos) == cell.rules.mapping[cube.class_name]
