"""Forward kinematics (DH), inverse kinematics, manipulability and reachability."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from cobotsort.cell.reachability import reachability_report
from cobotsort.kinematics.dh import (
    UR5E_MENAGERIE,
    UR5E_OFFICIAL,
    DHModel,
    pose_error,
    pose_to_matrix,
    tcp_positions,
)
from cobotsort.kinematics.ik import IKSolver, Pose
from cobotsort.kinematics.manipulability import manipulability


def _mj_flange(world, q: np.ndarray) -> np.ndarray:
    world.data.qpos[world.info.robot_qpos_adr] = q
    mujoco.mj_kinematics(world.model, world.data)
    s = world.info.flange_site
    return pose_to_matrix(
        world.data.site_xpos[s].copy(), world.data.site_xmat[s].reshape(3, 3).copy()
    )


def _random_q(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.uniform(-np.pi, np.pi, (n, 6))
    q[:, 2] = rng.uniform(-2.8, 2.8, n)  # the elbow is limited to +-pi
    return q


@pytest.fixture(scope="module")
def fitted(request):
    from cobotsort.sim.world import SimWorld

    world = SimWorld()
    rng = np.random.default_rng(0)
    fit_q = _random_q(rng, 30)
    models = {}
    for table in (UR5E_MENAGERIE, UR5E_OFFICIAL):
        m = DHModel(table)
        m.fit(fit_q, [_mj_flange(world, q) for q in fit_q])
        models[table.name] = m
    test_q = _random_q(rng, 50)
    return world, models, test_q, [_mj_flange(world, q) for q in test_q]


def test_dh_matches_mujoco_within_1mm(fitted):
    """Fit constant base/flange transforms once; FK must match MuJoCo on 50 random configs."""
    _, models, test_q, test_p = fitted
    model = models[UR5E_MENAGERIE.name]
    errs = np.array([pose_error(model.fk(q), p) for q, p in zip(test_q, test_p, strict=True)])
    assert errs[:, 0].max() < 1e-3, f"max position error {errs[:, 0].max() * 1e3:.4f} mm"
    assert errs[:, 1].max() < 0.1, f"max rotation error {errs[:, 1].max():.4f} deg"


def test_official_dh_table_is_close_but_rounded(fitted):
    """The official UR5e table differs from the Menagerie geometry by about a millimeter."""
    _, models, test_q, test_p = fitted
    model = models[UR5E_OFFICIAL.name]
    errs = np.array([pose_error(model.fk(q), p) for q, p in zip(test_q, test_p, strict=True)])
    assert errs[:, 0].max() < 2.5e-3
    assert errs[:, 1].max() < 0.1


def test_batch_tcp_matches_mujoco(world):
    rng = np.random.default_rng(3)
    qs = _random_q(rng, 20)
    tcp = tcp_positions(qs)
    for q, p in zip(qs, tcp, strict=True):
        world.data.qpos[world.info.robot_qpos_adr] = q
        mujoco.mj_kinematics(world.model, world.data)
        assert np.linalg.norm(world.data.site_xpos[world.info.tcp_site] - p) < 1e-9


def test_ik_100_reachable_targets(world):
    """IK error < 0.1 mm / 0.1 deg on 100 reachable (non-singular) targets, solved from home."""
    ik = IKSolver(world.info)
    rng = np.random.default_rng(7)
    targets = []
    while len(targets) < 100:
        q = _random_q(rng, 1)[0]
        if manipulability(ik.jacobian(q)) < 0.01:
            continue
        targets.append(Pose(*ik.site_pose(q)))
    for target in targets:
        sols = ik.solve_all(target, world.info.home_q, rng=rng)
        assert sols, f"no IK solution for reachable target {target.pos}"
        best = sols[0]
        assert best.pos_error < 1e-4
        assert np.degrees(best.rot_error) < 0.1


def test_ik_many_or_no_solutions(world):
    ik = IKSolver(world.info)
    pick = Pose.tool_down(0.45, 0.0, 0.10)
    sols = ik.solve_all(pick, world.info.home_q, n_random=12)
    assert len(sols) >= 2, "a reachable pose should have several IK solutions"
    norms = [np.linalg.norm(s.q - world.info.home_q) for s in sols]
    assert norms == sorted(norms), "solutions must be sorted by distance to the current joints"
    assert ik.solve_all(Pose.tool_down(1.5, 0.0, 0.10), world.info.home_q) == []


def test_all_slots_and_pick_area_reachable(world):
    ok, lines = reachability_report(world)
    assert ok, "\n".join(lines)


def test_stretched_arm_is_near_singular(world):
    ik = IKSolver(world.info)
    stretched = world.info.home_q.copy()
    stretched[2] = 0.02  # elbow almost straight
    assert manipulability(ik.jacobian(stretched)) < world.cfg.safety.manip_block
    assert manipulability(ik.jacobian(world.info.home_q)) > world.cfg.safety.manip_warn
