"""Trajectory generation: MoveJ (joint space) and MoveL (straight-line TCP).

* MoveJ: all joints start and stop together on a quintic time scaling
  ``s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5``. The duration comes from the slowest joint
  under ``v_max``/``a_max`` and is stretched if the TCP would exceed its speed limit.
* MoveL: waypoints every 5 mm on the line, orientation by slerp, IK seeded from the
  previous waypoint. A joint jump above 0.2 rad between waypoints means the arm is
  flipping its wrist near a singularity, so the move is refused.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cobotsort.config import MotionConfig
from cobotsort.kinematics.dh import tcp_positions
from cobotsort.kinematics.ik import IKSolver, Pose

log = logging.getLogger(__name__)

# Peak velocity and acceleration of the quintic profile, relative to distance / T and / T^2.
QUINTIC_V = 1.875
QUINTIC_A = 5.773502691896258


class MotionError(RuntimeError):
    """The requested motion cannot be planned."""


def quintic(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position and derivative (w.r.t. tau) of the quintic time scaling."""
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    return s, ds


@dataclass
class Trajectory:
    """A time-sampled joint trajectory (one sample per physics step)."""

    kind: str
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tcp: np.ndarray
    label: str = ""
    final_approach: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def q_final(self) -> np.ndarray:
        return self.q[-1]

    @property
    def peak_joint_speed(self) -> float:
        return float(np.max(np.abs(self.qd))) if len(self.qd) else 0.0

    @property
    def peak_tcp_speed(self) -> float:
        if len(self.t) < 2:
            return 0.0
        v = np.linalg.norm(np.diff(self.tcp, axis=0), axis=1) / np.diff(self.t)
        return float(np.max(v))


class TrajectoryPlanner:
    """Plans MoveJ and MoveL trajectories sampled at the physics timestep."""

    def __init__(self, ik: IKSolver, cfg: MotionConfig, dt: float) -> None:
        self.ik = ik
        self.cfg = cfg
        self.dt = dt

    # ------------------------------------------------------------ MoveJ
    def movej(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        v_max: float | None = None,
        tcp_speed_max: float | None = None,
        label: str = "",
    ) -> Trajectory:
        v_max = self.cfg.v_max if v_max is None else v_max
        tcp_max = self.cfg.tcp_speed_max if tcp_speed_max is None else tcp_speed_max
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        dq = q1 - q0
        dist = float(np.max(np.abs(dq)))
        if dist < 1e-9:
            return self._sample_joint(q0, dq, self.dt, "movej", label)
        duration = max(QUINTIC_V * dist / v_max, np.sqrt(QUINTIC_A * dist / self.cfg.a_max), 0.1)
        traj = self._sample_joint(q0, dq, duration, "movej", label)
        peak = traj.peak_tcp_speed
        if peak > tcp_max:
            # Same path, slower: speeds scale with 1 / duration.
            traj = self._sample_joint(q0, dq, duration * peak / tcp_max * 1.01, "movej", label)
        traj.meta.update(v_max=v_max, tcp_speed_max=tcp_max)
        return traj

    def _sample_joint(
        self, q0: np.ndarray, dq: np.ndarray, duration: float, kind: str, label: str
    ) -> Trajectory:
        n = max(2, int(np.ceil(duration / self.dt)) + 1)
        t = np.linspace(0.0, duration, n)
        s, ds = quintic(t / duration)
        q = q0 + s[:, None] * dq
        qd = (ds / duration)[:, None] * dq
        return Trajectory(kind, t, q, qd, tcp_positions(q), label)

    # ------------------------------------------------------------ MoveL
    def movel(
        self,
        q0: np.ndarray,
        target: Pose,
        tcp_speed_max: float | None = None,
        v_max: float | None = None,
        label: str = "",
        final_approach: bool = False,
    ) -> Trajectory:
        tcp_max = self.cfg.tcp_speed_max if tcp_speed_max is None else tcp_speed_max
        v_max = self.cfg.v_max if v_max is None else v_max
        q0 = np.asarray(q0, float)
        p0, r0 = self.ik.site_pose(q0)
        length = float(np.linalg.norm(target.pos - p0))
        rots = Rotation.from_matrix(np.stack([r0, target.rot]))
        angle = float((rots[0].inv() * rots[1]).magnitude())
        n_seg = max(1, int(np.ceil(length / self.cfg.waypoint_step)), int(np.ceil(angle / 0.035)))
        s_wp = np.linspace(0.0, 1.0, n_seg + 1)
        slerp = Slerp([0.0, 1.0], rots)
        rot_wp = slerp(s_wp).as_matrix()
        q_wp = np.zeros((n_seg + 1, 6))
        q_wp[0] = q0
        for i in range(1, n_seg + 1):
            pose = Pose(p0 + s_wp[i] * (target.pos - p0), rot_wp[i])
            res = self.ik.solve(pose, q_wp[i - 1], max_iter=100)
            if not res.success:
                raise MotionError(f"MoveL: no IK solution at waypoint {i}/{n_seg} ({label})")
            jump = float(np.max(np.abs(res.q - q_wp[i - 1])))
            if jump > self.cfg.max_joint_jump:
                raise MotionError(
                    f"MoveL: joint jump {jump:.3f} rad at waypoint {i}/{n_seg} "
                    f"(wrist flip near a singularity) ({label})"
                )
            q_wp[i] = res.q
        # Duration from the TCP speed, the tool rotation speed and the joint speeds.
        dq_ds = np.max(np.abs(np.diff(q_wp, axis=0)), axis=0) * n_seg  # per unit of s
        duration = max(
            QUINTIC_V * length / tcp_max,
            np.sqrt(QUINTIC_A * length / self.cfg.tcp_accel_max),
            QUINTIC_V * angle / self.cfg.rot_speed_max,
            QUINTIC_V * float(np.max(dq_ds)) / v_max,
            0.1,
        )
        n = max(2, int(np.ceil(duration / self.dt)) + 1)
        t = np.linspace(0.0, duration, n)
        s, ds = quintic(t / duration)
        q = np.stack([np.interp(s, s_wp, q_wp[:, j]) for j in range(6)], axis=1)
        seg = np.clip(np.searchsorted(s_wp, s, side="right") - 1, 0, n_seg - 1)
        slope = np.diff(q_wp, axis=0) / np.diff(s_wp)[:, None]
        qd = slope[seg] * (ds / duration)[:, None]
        traj = Trajectory("movel", t, q, qd, tcp_positions(q), label, final_approach)
        traj.meta.update(
            length=length, waypoints=n_seg + 1, v_max=v_max, tcp_speed_max=tcp_max, q_wp=q_wp
        )
        return traj
