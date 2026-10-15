"""Placeholder for the ARM95 cobot used at the BRICS Hi-Tech Skills Camp.

The ARM95 model and protocol are not public, so nothing here talks to hardware. The class
documents how :class:`CobotAPI` calls would map onto the command set practiced at the camp,
so that a real backend could be written later without touching the cell logic.

=======================  =============================================
CobotAPI                 Camp command set (as practiced)
=======================  =============================================
``connect()``            open the controller session
``engage()``             enable / engage the drives
``move_to_start()``      move to the start (home) position
``move_to_point_j(p)``   joint-interpolated move to a point
``move_to_point_l(p)``   linear (Cartesian) move to a point
``tool_on()``            vacuum on
``tool_off()``           vacuum off
``stop()``               stop motion
``reset_errors()``       clear errors (then engage again)
=======================  =============================================
"""

from __future__ import annotations

import numpy as np

from cobotsort.kinematics.ik import Pose
from cobotsort.motion.trajectory import Trajectory
from cobotsort.robot.backend import ExecResult, Monitor, RobotBackend, VacuumResult


class Arm95Backend(RobotBackend):
    """Not implemented: a real cell would stream the planned waypoints to the controller."""

    def _todo(self) -> NotImplementedError:
        return NotImplementedError("Arm95Backend is a stub; this project runs in simulation only")

    def connect(self) -> None:
        raise self._todo()

    def joint_positions(self) -> np.ndarray:
        raise self._todo()

    def tcp_pose(self) -> Pose:
        raise self._todo()

    def execute(self, traj: Trajectory, monitor: Monitor | None = None) -> ExecResult:
        raise self._todo()

    def wait(self, seconds: float, monitor: Monitor | None = None) -> ExecResult:
        raise self._todo()

    def hold(self) -> None:
        raise self._todo()

    def set_vacuum(self, on: bool) -> VacuumResult:
        raise self._todo()

    @property
    def held_part(self) -> int | None:
        raise self._todo()

    @property
    def time(self) -> float:
        raise self._todo()
