"""MuJoCo implementation of :class:`RobotBackend`."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from cobotsort.config import MotionConfig, SafetyConfig
from cobotsort.kinematics.dh import tcp_positions
from cobotsort.kinematics.ik import Pose
from cobotsort.motion.trajectory import Trajectory
from cobotsort.robot.backend import ExecResult, Monitor, RobotBackend, VacuumResult
from cobotsort.sim.world import SimWorld, quat_to_mat

log = logging.getLogger(__name__)

# Called after every physics step (used by the video recorder).
StepCallback = Callable[[SimWorld], None]


class SimBackend(RobotBackend):
    """Streams joint setpoints to the Menagerie position servos (or writes them directly)."""

    def __init__(
        self,
        world: SimWorld,
        motion: MotionConfig | None = None,
        safety: SafetyConfig | None = None,
        monitor_every: int = 5,
    ) -> None:
        self.world = world
        self.motion = motion or world.cfg.motion
        self.safety = safety or world.cfg.safety
        self.monitor_every = monitor_every
        self.step_callbacks: list[StepCallback] = []
        self._hold_q = world.q

    # ------------------------------------------------------------ basics
    def connect(self) -> None:
        self._hold_q = self.world.q

    def joint_positions(self) -> np.ndarray:
        return self.world.q

    def tcp_pose(self) -> Pose:
        pos, rot = self.world.tcp_pose()
        return Pose(pos, rot)

    @property
    def time(self) -> float:
        return self.world.time

    @property
    def held_part(self) -> int | None:
        return self.world.attached

    @property
    def kinematic(self) -> bool:
        return self.motion.kinematic

    def _step(self, q_des: np.ndarray, qd_des: np.ndarray | None = None) -> None:
        self.world.step(q_des, qd_des, kinematic=self.kinematic)
        for cb in self.step_callbacks:
            cb(self.world)

    # ------------------------------------------------------------ motion
    def execute(self, traj: Trajectory, monitor: Monitor | None = None) -> ExecResult:
        """Stream the trajectory, then settle on the final setpoint."""
        t0 = self.world.time
        max_err = 0.0
        for i in range(1, len(traj.t)):
            self._step(traj.q[i], traj.qd[i])
            if i % self.monitor_every == 0 or i == len(traj.t) - 1:
                err = self.tracking_error(traj.q[i])
                max_err = max(max_err, err)
                reason = monitor(self, traj.q[i]) if monitor else None
                if reason is None and err > self.safety.tracking_error_stop:
                    limit = self.safety.tracking_error_stop * 1e3
                    reason = f"tracking error {err * 1e3:.1f} mm > {limit:.0f} mm"
                if reason:
                    self.hold()
                    return ExecResult(False, self.world.time - t0, max_err, reason)
        settle = self._settle(traj.q_final, monitor)
        max_err = max(max_err, settle.max_tracking_error)
        self._hold_q = traj.q_final.copy()
        if not settle.completed:
            return ExecResult(False, self.world.time - t0, max_err, settle.stop_reason)
        return ExecResult(True, self.world.time - t0, max_err)

    def _settle(self, q_final: np.ndarray, monitor: Monitor | None) -> ExecResult:
        if self.kinematic:
            self.world.set_robot_q(q_final)
            return ExecResult(True, 0.0, 0.0)
        t0 = self.world.time
        n = 0
        while self.world.time - t0 < self.motion.settle_timeout:
            if np.max(np.abs(self.world.q - q_final)) < self.motion.settle_tol:
                break
            self._step(q_final)
            n += 1
            if monitor and n % self.monitor_every == 0:
                reason = monitor(self, q_final)
                if reason:
                    self.hold()
                    return ExecResult(False, self.world.time - t0, 0.0, reason)
        self.world.forward()
        return ExecResult(True, self.world.time - t0, self.tracking_error(q_final))

    def wait(self, seconds: float, monitor: Monitor | None = None) -> ExecResult:
        t0 = self.world.time
        n = 0
        while self.world.time - t0 < seconds - 1e-9:
            self._step(self._hold_q)
            n += 1
            if monitor and n % self.monitor_every == 0:
                reason = monitor(self, self._hold_q)
                if reason:
                    self.hold()
                    return ExecResult(False, self.world.time - t0, 0.0, reason)
        self.world.forward()
        return ExecResult(True, self.world.time - t0)

    def hold(self) -> None:
        """Protective stop: freeze the setpoint at the measured position."""
        self._hold_q = self.world.q
        self.world.data.ctrl[self.world.info.actuator_ids] = self._hold_q
        self.world.data.qvel[self.world.info.robot_dof_adr] = 0.0
        self.world.forward()

    def tracking_error(self, q_des: np.ndarray) -> float:
        """TCP distance between commanded and measured joints (m)."""
        p = tcp_positions(np.stack([q_des, self.world.q]))
        return float(np.linalg.norm(p[0] - p[1]))

    # ------------------------------------------------------------ vacuum
    def set_vacuum(self, on: bool) -> VacuumResult:
        if not on:
            part = self.world.detach()
            return VacuumResult(False, part, "released" if part is not None else "no part held")
        if self.world.attached is not None:
            return VacuumResult(True, self.world.attached, "already holding a part")
        tcp, rot = self.world.tcp_pose()
        tilt = np.degrees(np.arccos(np.clip(-rot[2, 2], -1.0, 1.0)))
        if tilt > self.safety.vacuum_max_tilt_deg:
            return VacuumResult(True, None, f"no part (tool tilted {tilt:.1f} deg)")
        best, best_d = None, np.inf
        half = self.world.cfg.scene.cube_edge / 2
        for cube in self.world.cubes():
            local = quat_to_mat(cube.quat).T @ (tcp - cube.pos)
            d = float(np.linalg.norm(local - np.clip(local, -half, half)))
            if d < best_d:
                best, best_d = cube.index, d
        if best is None or best_d > self.safety.vacuum_max_distance:
            return VacuumResult(True, None, f"no part (closest {best_d * 1e3:.1f} mm)")
        self.world.attach(best)
        return VacuumResult(True, best, f"part {best} attached ({best_d * 1e3:.1f} mm)")
