"""Hardware abstraction: what the cell needs from any robot, simulated or real."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cobotsort.kinematics.ik import Pose
from cobotsort.motion.trajectory import Trajectory

# A runtime monitor receives the backend after a control step and returns None to continue
# or a reason string to trigger a protective stop.
Monitor = Callable[["RobotBackend", np.ndarray], str | None]


class CobotError(RuntimeError):
    """Base class for robot errors."""


class NotEngagedError(CobotError):
    """Motion was requested while the robot is not engaged."""


class ProtectiveStop(CobotError):
    """The runtime monitor or the operator stopped the robot."""


@dataclass
class ExecResult:
    """Outcome of executing a trajectory or a wait."""

    completed: bool
    duration: float
    max_tracking_error: float = 0.0
    stop_reason: str | None = None


@dataclass
class VacuumResult:
    """Outcome of switching the vacuum."""

    on: bool
    part: int | None
    message: str


class RobotBackend(abc.ABC):
    """Minimal command set a cell needs from a robot."""

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def joint_positions(self) -> np.ndarray: ...

    @abc.abstractmethod
    def tcp_pose(self) -> Pose: ...

    @abc.abstractmethod
    def execute(self, traj: Trajectory, monitor: Monitor | None = None) -> ExecResult: ...

    @abc.abstractmethod
    def wait(self, seconds: float, monitor: Monitor | None = None) -> ExecResult: ...

    @abc.abstractmethod
    def hold(self) -> None:
        """Hold the current position (protective stop)."""

    @abc.abstractmethod
    def set_vacuum(self, on: bool) -> VacuumResult: ...

    @property
    @abc.abstractmethod
    def held_part(self) -> int | None: ...

    @property
    @abc.abstractmethod
    def time(self) -> float:
        """Robot clock in seconds (simulated time for the simulator)."""
