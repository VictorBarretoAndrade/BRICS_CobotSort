"""Damped least-squares inverse kinematics on the MuJoCo site Jacobian.

Unlike forward kinematics (one answer per joint vector), IK can have many solutions
(elbow up/down, wrist flips, +-2pi turns) or none. ``solve_all`` runs DLS from several
seeds and returns every distinct solution, sorted by distance to the current joints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from cobotsort.sim.scene import SceneInfo

log = logging.getLogger(__name__)


@dataclass
class Pose:
    """A TCP pose: position (3,) and rotation matrix (3, 3) in the world frame."""

    pos: np.ndarray
    rot: np.ndarray

    @classmethod
    def tool_down(cls, x: float, y: float, z: float, yaw: float | None = None) -> Pose:
        """Tool z axis pointing at the table.

        ``yaw`` is the angle of the tool x axis in the world xy plane. By default it follows
        the direction from the robot base, which keeps the last wrist joint near its home value.
        """
        if yaw is None:
            yaw = float(np.arctan2(y, x) - np.pi / 2)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])
        return cls(np.array([x, y, z], dtype=float), rot)

    @property
    def quat(self) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.rot.flatten())
        return q


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    pos_error: float
    rot_error: float
    iterations: int


def orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) that takes ``current`` to ``target``."""
    qt, qc, qe = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(qt, target.flatten())
    mujoco.mju_mat2Quat(qc, current.flatten())
    qc_inv = np.zeros(4)
    mujoco.mju_negQuat(qc_inv, qc)
    mujoco.mju_mulQuat(qe, qt, qc_inv)
    if qe[0] < 0:
        qe = -qe
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, qe, 1.0)
    return vel


class IKSolver:
    """DLS IK for the TCP site using a private scratch ``MjData``."""

    def __init__(
        self,
        info: SceneInfo,
        site: int | None = None,
        tol_pos: float = 1e-5,
        tol_rot: float = 1e-4,
        max_iter: int = 200,
        lambda_min: float = 1e-3,
        lambda_max: float = 0.08,
        w_threshold: float = 0.04,
        max_step: float = 0.4,
    ) -> None:
        self.info = info
        self.model = info.model
        self.data = mujoco.MjData(self.model)
        self.site = info.tcp_site if site is None else site
        self.tol_pos = tol_pos
        self.tol_rot = tol_rot
        self.max_iter = max_iter
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.w_threshold = w_threshold
        self.max_step = max_step
        jids = [self.model.dof_jntid[d] for d in info.robot_dof_adr]
        self.lower = self.model.jnt_range[jids, 0].copy()
        self.upper = self.model.jnt_range[jids, 1].copy()
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    # ------------------------------------------------------------ primitives
    def site_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.info.robot_qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return (
            self.data.site_xpos[self.site].copy(),
            self.data.site_xmat[self.site].reshape(3, 3).copy(),
        )

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6x6 geometric Jacobian [linear; angular] of the site (world frame)."""
        self.data.qpos[self.info.robot_qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site)
        dofs = self.info.robot_dof_adr
        return np.vstack([self._jacp[:, dofs], self._jacr[:, dofs]])

    # ------------------------------------------------------------ solvers
    def solve(self, target: Pose, q_seed: np.ndarray, max_iter: int | None = None) -> IKResult:
        """Single-seed damped least squares with adaptive damping."""
        q = np.clip(np.asarray(q_seed, float).copy(), self.lower, self.upper)
        it = 0
        ep = er = np.inf
        for it in range(1, (max_iter or self.max_iter) + 1):
            jac = self.jacobian(q)
            pos = self.data.site_xpos[self.site]
            rot = self.data.site_xmat[self.site].reshape(3, 3)
            e = np.concatenate([target.pos - pos, orientation_error(target.rot, rot)])
            ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if ep < self.tol_pos and er < self.tol_rot:
                return IKResult(q, True, ep, er, it)
            jjt = jac @ jac.T
            w = float(np.sqrt(max(np.linalg.det(jjt), 0.0)))
            # Damping grows smoothly as the arm approaches a singularity and fades out as the
            # error shrinks, so the last iterations are plain Gauss-Newton steps.
            lam = self.lambda_min * min(1.0, ep / 1e-3)
            if w < self.w_threshold:
                fade = min(1.0, ep / 5e-3 + er / 5e-2)
                lam += self.lambda_max * (1.0 - (w / self.w_threshold) ** 2) * fade
            dq = jac.T @ np.linalg.solve(jjt + lam**2 * np.eye(6), e)
            norm = np.linalg.norm(dq)
            if norm > self.max_step:
                dq *= self.max_step / norm
            q = np.clip(q + dq, self.lower, self.upper)
        pos, rot = self.site_pose(q)
        ep = float(np.linalg.norm(target.pos - pos))
        er = float(np.linalg.norm(orientation_error(target.rot, rot)))
        return IKResult(q, ep < self.tol_pos and er < self.tol_rot, ep, er, it)

    def seeds(self, q_current: np.ndarray, n_random: int, rng: np.random.Generator) -> list:
        """Current joints, canonical elbow-up / elbow-down / wrist-flip seeds and random ones."""
        out = [np.asarray(q_current, float)]
        pan = float(q_current[0])
        for lift, elbow in ((-1.2, 1.8), (-2.0, -1.8), (-0.9, 1.2)):
            for w1, w2 in ((-2.2, -np.pi / 2), (-0.9, np.pi / 2)):
                out.append(np.array([pan, lift, elbow, w1, w2, 0.0]))
        for _ in range(n_random):
            out.append(rng.uniform(np.maximum(self.lower, -np.pi), np.minimum(self.upper, np.pi)))
        return out

    def solve_all(
        self,
        target: Pose,
        q_current: np.ndarray,
        n_random: int = 6,
        rng: np.random.Generator | None = None,
        dedup_tol: float = 1e-2,
    ) -> list[IKResult]:
        """All distinct solutions from several seeds, closest to ``q_current`` first."""
        rng = rng or np.random.default_rng(0)
        sols: list[IKResult] = []
        for seed in self.seeds(q_current, n_random, rng):
            res = self.solve(target, seed)
            if not res.success:
                continue
            if all(np.max(np.abs(res.q - s.q)) > dedup_tol for s in sols):
                sols.append(res)
        sols.sort(key=lambda r: float(np.linalg.norm(r.q - q_current)))
        return sols

    def closest(self, target: Pose, q_current: np.ndarray) -> IKResult | None:
        """Best solution for a point-to-point move: try the current seed first."""
        res = self.solve(target, q_current)
        sols = self.solve_all(target, q_current)
        if res.success and (
            not sols
            or np.linalg.norm(res.q - q_current) <= np.linalg.norm(sols[0].q - q_current) + 1e-6
        ):
            return res
        return sols[0] if sols else None
