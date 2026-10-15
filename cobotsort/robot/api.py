"""CobotAPI: the command set the cell logic and the operator agent use.

The method names are inspired by the command set practiced at the BRICS Hi-Tech Skills
Camp; this is not the real ARM95 MotionAPI. Every motion is planned first, checked by the
SafetyGuard (when one is attached) and only then executed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cobotsort.config import CellConfig
from cobotsort.kinematics.ik import IKSolver, Pose
from cobotsort.motion.trajectory import MotionError, Trajectory, TrajectoryPlanner
from cobotsort.robot.backend import (
    CobotError,
    ExecResult,
    NotEngagedError,
    ProtectiveStop,
    RobotBackend,
    VacuumResult,
)
from cobotsort.safety.lamp import LampState, SignalLamp

log = logging.getLogger(__name__)


class SafetyViolation(CobotError):
    """The SafetyGuard refused a planned motion."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass
class MoveContext:
    """What the guard needs to know about a motion besides the trajectory."""

    kind: str
    requested_speed: float | None = None
    final_approach: bool = False
    target_part: int | None = None
    target_slot: tuple[float, float] | None = None


@dataclass
class PickPlaceResult:
    success: bool
    reason: str
    part: int | None = None
    duration: float = 0.0


class CobotAPI:
    """High-level robot interface with an engage / E-stop / reset state machine."""

    def __init__(
        self,
        backend: RobotBackend,
        ik: IKSolver,
        planner: TrajectoryPlanner,
        home_q: np.ndarray,
        cfg: CellConfig | None = None,
        guard=None,
        lamp: SignalLamp | None = None,
    ) -> None:
        self.backend = backend
        self.ik = ik
        self.planner = planner
        self.home_q = np.asarray(home_q, float)
        self.cfg = cfg or CellConfig()
        self.guard = guard
        self.lamp = lamp or SignalLamp()
        self.connected = False
        self.engaged = False
        self.estop_active = False
        self.fault: str | None = None
        self.warning: str | None = None
        self.command = "idle"
        self.listeners: list[Callable[[str], None]] = []
        self.safety_events: list[str] = []

    # ------------------------------------------------------------ state machine
    def _emit(self, text: str) -> None:
        self.command = text
        log.debug("%s", text)
        for cb in self.listeners:
            cb(text)

    def _lamp(self, executing: bool = False, reason: str = "") -> None:
        self.lamp.update(
            engaged=self.engaged,
            executing=executing,
            fault=self.estop_active or self.fault is not None,
            warning=self.warning is not None,
            reason=reason,
        )

    def connect(self) -> None:
        self.backend.connect()
        self.connected = True
        self._emit("connect")
        self._lamp()

    def engage(self) -> None:
        if not self.connected:
            raise CobotError("connect() first")
        if self.estop_active:
            raise CobotError("E-stop active: reset_errors() first")
        if self.fault:
            raise CobotError(f"fault active ({self.fault}): reset_errors() first")
        self.engaged = True
        self._emit("engage")
        self._lamp()

    def disengage(self) -> None:
        self.engaged = False
        self._lamp()

    def stop(self, reason: str = "operator stop") -> None:
        """Protective stop: hold position, lamp RED, engage required after reset."""
        self.backend.hold()
        self.engaged = False
        self.fault = reason
        self.safety_events.append(reason)
        self._emit(f"stop: {reason}")
        self._lamp(reason=reason)

    def emergency_stop(self) -> None:
        self.estop_active = True
        self.stop("E-stop")

    def reset_errors(self) -> None:
        """Clear faults and the E-stop latch. The robot must be engaged again."""
        self.estop_active = False
        self.fault = None
        self.warning = None
        self.engaged = False
        clear = getattr(self.backend, "clear_stop", None)
        if clear is not None:
            clear()
        self._emit("reset_errors")
        self._lamp()

    def get_tool_position(self) -> Pose:
        return self.backend.tcp_pose()

    def _require_motion(self) -> None:
        if self.estop_active:
            raise ProtectiveStop("E-stop active")
        if self.fault:
            raise ProtectiveStop(f"fault active: {self.fault}")
        if not (self.connected and self.engaged):
            raise NotEngagedError("robot is not engaged")

    # ------------------------------------------------------------ motion
    def _run(self, traj: Trajectory, ctx: MoveContext) -> ExecResult:
        self._require_motion()
        if self.guard is not None:
            decision = self.guard.check(traj, ctx, self)
            if decision.warnings:
                self.warning = "; ".join(decision.warnings)
            if not decision.allowed:
                self._emit(f"refused {traj.label}: {'; '.join(decision.reasons)}")
                self._lamp(reason="refused")
                raise SafetyViolation(decision.reasons)
        self._emit(traj.label or traj.kind)
        self._lamp(executing=True)
        monitor = self.guard.monitor(ctx) if self.guard is not None else None
        res = self.backend.execute(traj, monitor)
        if not res.completed:
            self.stop(f"protective stop: {res.stop_reason}")
            raise ProtectiveStop(res.stop_reason or "protective stop")
        self.warning = None
        self._lamp()
        return res

    def plan_j(
        self, target: np.ndarray | Pose, speed: float | None = None, label: str = ""
    ) -> Trajectory:
        q0 = self.backend.joint_positions()
        if isinstance(target, Pose):
            sol = self.ik.closest(target, q0)
            if sol is None:
                raise MotionError(f"no IK solution for {np.round(target.pos, 3).tolist()}")
            q1 = sol.q
        else:
            q1 = np.asarray(target, float)
        return self.planner.movej(q0, q1, v_max=speed, label=label or "move_to_point_j")

    def move_to_point_j(
        self, target: np.ndarray | Pose, speed: float | None = None, label: str = "", **ctx
    ) -> ExecResult:
        """Joint-space move to a joint vector or a pose (IK picks the closest solution)."""
        self._require_motion()
        traj = self.plan_j(target, speed, label)
        return self._run(traj, MoveContext("movej", requested_speed=speed, **ctx))

    def move_to_point_l(
        self,
        pose: Pose,
        speed: float | None = None,
        label: str = "",
        final_approach: bool = False,
        **ctx,
    ) -> ExecResult:
        """Straight-line TCP move."""
        self._require_motion()
        q0 = self.backend.joint_positions()
        traj = self.planner.movel(
            q0,
            pose,
            tcp_speed_max=speed,
            label=label or "move_to_point_l",
            final_approach=final_approach,
        )
        return self._run(
            traj, MoveContext("movel", requested_speed=speed, final_approach=final_approach, **ctx)
        )

    def move_to_start(self) -> ExecResult:
        return self.move_to_point_j(self.home_q, label="move_to_start")

    def wait(self, seconds: float) -> ExecResult:
        monitor = self.guard.monitor(MoveContext("wait")) if self.guard is not None else None
        res = self.backend.wait(seconds, monitor)
        if not res.completed:
            self.stop(f"protective stop: {res.stop_reason}")
            raise ProtectiveStop(res.stop_reason or "protective stop")
        return res

    # ------------------------------------------------------------ vacuum
    def tool_on(self) -> VacuumResult:
        self._require_motion()
        res = self.backend.set_vacuum(True)
        self._emit(f"tool_on: {res.message}")
        return res

    def tool_off(self) -> VacuumResult:
        res = self.backend.set_vacuum(False)
        self._emit(f"tool_off: {res.message}")
        return res

    # ------------------------------------------------------------ sequence
    def pick_and_place(
        self,
        pick_xy: tuple[float, float],
        place_xy: tuple[float, float],
        part_top_z: float | None = None,
        place_floor_z: float | None = None,
        slot_label: str = "slot",
        part_label: str = "part",
        target_part: int | None = None,
    ) -> PickPlaceResult:
        """The 8-step pick-and-place sequence (see SPEC section 6.2)."""
        sc, mc = self.cfg.scene, self.cfg.motion
        top = sc.cube_edge if part_top_z is None else part_top_z
        floor = sc.bin_floor_thickness if place_floor_z is None else place_floor_z
        t0 = self.backend.time
        px, py = pick_xy
        sx, sy = place_xy
        safe_pick = top + mc.approach_height
        safe_place = floor + sc.cube_edge + mc.approach_height
        self.move_to_point_j(
            Pose.tool_down(px, py, safe_pick), label=f"move_to_point_j -> above {part_label}"
        )
        self.move_to_point_l(
            Pose.tool_down(px, py, top + 0.001),
            label=f"move_to_point_l -> {part_label}",
            final_approach=True,
            target_part=target_part,
        )
        vac = self.tool_on()
        self.wait(mc.vacuum_wait)
        if vac.part is None:
            self.move_to_point_l(
                Pose.tool_down(px, py, safe_pick), label="move_to_point_l -> safe height"
            )
            return PickPlaceResult(False, vac.message, None, self.backend.time - t0)
        self.move_to_point_l(
            Pose.tool_down(px, py, safe_pick), label="move_to_point_l -> safe height"
        )
        self.move_to_point_j(
            Pose.tool_down(sx, sy, safe_place), label=f"move_to_point_j -> above {slot_label}"
        )
        self.move_to_point_l(
            Pose.tool_down(sx, sy, floor + sc.cube_edge + mc.place_clearance),
            label=f"move_to_point_l -> {slot_label}",
            final_approach=True,
            target_slot=(sx, sy),
        )
        self.tool_off()
        self.wait(mc.vacuum_wait)
        self.move_to_point_l(
            Pose.tool_down(sx, sy, safe_place), label="move_to_point_l -> safe height"
        )
        return PickPlaceResult(True, "placed", vac.part, self.backend.time - t0)


def make_api(world, guard=None, cfg: CellConfig | None = None) -> CobotAPI:
    """Wire a SimWorld into a ready-to-use (connected, not engaged) CobotAPI."""
    from cobotsort.robot.sim_backend import SimBackend

    cfg = cfg or world.cfg
    ik = IKSolver(world.info)
    planner = TrajectoryPlanner(ik, cfg.motion, world.model.opt.timestep)
    backend = SimBackend(world, cfg.motion, cfg.safety)
    lamp = SignalLamp(sink=world.set_lamp, clock=lambda: world.time)
    api = CobotAPI(backend, ik, planner, world.info.home_q, cfg, guard=guard, lamp=lamp)
    api.connect()
    return api


__all__ = ["CobotAPI", "PickPlaceResult", "SafetyViolation", "MoveContext", "LampState", "make_api"]
