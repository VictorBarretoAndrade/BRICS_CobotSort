"""SafetyGuard: pre-motion checks on the whole planned trajectory and a runtime monitor.

Pre-motion checks (all are evaluated; every failing one is reported):

1. joint limits;
2. workspace box for the TCP (lower z bound relaxed only for a final approach);
3. requested speed and the planned joint / TCP speeds;
4. manipulability (block near singularities, warn when low);
5. collisions, by replaying the trajectory in a scratch ``MjData`` (robot or held part against the
   table, bins, signal tower or other cubes; the cup may touch its target during a pick);
6. the robot is engaged and the E-stop is not active.

Runtime monitor: any unplanned contact during execution triggers a protective stop. (The
tracking-error stop is in the backend.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mujoco
import numpy as np

from cobotsort.config import MotionConfig, SafetyConfig
from cobotsort.kinematics.dh import forward_batch
from cobotsort.kinematics.ik import IKSolver
from cobotsort.kinematics.manipulability import ManipulabilityCheck
from cobotsort.motion.trajectory import Trajectory
from cobotsort.sim.world import SimWorld, mat_to_quat, quat_to_mat

log = logging.getLogger(__name__)

TOOL_LENGTH = 0.03


@dataclass
class SafetyDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    min_manipulability: float = float("nan")


class SafetyGuard:
    """Checks every planned motion before it runs and watches it while it runs."""

    def __init__(
        self,
        world: SimWorld,
        ik: IKSolver,
        cfg: SafetyConfig | None = None,
        motion: MotionConfig | None = None,
    ) -> None:
        self.world = world
        self.cfg = cfg or world.cfg.safety
        self.motion = motion or world.cfg.motion
        self.model = world.model
        self.scratch = mujoco.MjData(self.model)
        self.manip = ManipulabilityCheck(ik, self.cfg.manip_warn, self.cfg.manip_block)
        info = world.info
        self.robot_geoms = set(int(g) for g in info.robot_geom_ids)
        self.cup_geom = int(self.model.geom("cup").id)
        self.cube_geoms = [int(g) for g in info.cube_geom_ids]
        self.env_geoms = (
            {int(info.table_geom)}
            | {int(g) for ids in info.bin_geom_ids.values() for g in ids}
            | {int(g) for g in info.obstacle_geom_ids}
        )
        jids = [self.model.dof_jntid[d] for d in info.robot_dof_adr]
        self.q_lower = self.model.jnt_range[jids, 0]
        self.q_upper = self.model.jnt_range[jids, 1]
        self.unplanned_contacts = 0
        self.blocked: list[list[str]] = []
        # What the held part was resting on when it was attached (allowed until lift-off).
        self._support_for: int | None = None
        self._support: set[int] = set()
        self._support_z = 0.0

    def _refresh_support(self) -> None:
        held = self.world.attached
        if held == self._support_for:
            return
        self._support_for, self._support = held, set()
        if held is None:
            return
        g = self.cube_geoms[held]
        d = self.world.data
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if g in (g1, g2):
                other = g2 if g1 == g else g1
                if other in self.env_geoms or other in self.cube_geoms:
                    self._support.add(other)
        self._support_z = float(self.world.cube_pose(held)[0][2])

    # ------------------------------------------------------------ contacts
    def _bad_contacts(
        self, data: mujoco.MjData, held: int | None, allowed_cube: int | None
    ) -> list[str]:
        held_geom = self.cube_geoms[held] if held is not None else None
        allowed_geom = self.cube_geoms[allowed_cube] if allowed_cube is not None else None
        lifting = False
        if held is not None:
            self._refresh_support()
            adr = self.world.info.cube_qpos_adr[held]
            lifting = float(data.qpos[adr + 2]) < self._support_z + 0.005
        bad = []
        for i in range(data.ncon):
            g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
            for a, b in ((g1, g2), (g2, g1)):
                mover = a in self.robot_geoms or a == held_geom
                if not mover:
                    continue
                if b in self.robot_geoms or b == held_geom:
                    # Robot self-contact (the cup touching the held part is fine).
                    if {a, b} == {self.cup_geom, held_geom} or a == b:
                        continue
                    if a in self.robot_geoms and b in self.robot_geoms:
                        bad.append(f"{self.world.geom_name(a)}-{self.world.geom_name(b)}")
                    continue
                if b == allowed_geom and a == self.cup_geom:
                    continue
                if a == held_geom and lifting and b in self._support:
                    continue  # the part is still lifting off its support
                if b in self.env_geoms or b in self.cube_geoms:
                    bad.append(f"{self.world.geom_name(a)}-{self.world.geom_name(b)}")
                break
        return sorted(set(bad))

    def _target_cube(self, tcp_final: np.ndarray) -> int | None:
        """The cube under the final TCP position (the pick target), if any."""
        best, best_d = None, 0.03
        for cube in self.world.cubes():
            d = float(np.hypot(*(cube.pos[:2] - tcp_final[:2])))
            if d < best_d and cube.index != self.world.attached:
                best, best_d = cube.index, d
        return best

    def replay_collisions(self, traj: Trajectory, allowed_cube: int | None) -> list[str]:
        """Replay the trajectory in a scratch MjData and list forbidden contacts."""
        d = self.scratch
        d.qpos[:] = self.world.data.qpos
        held = self.world.attached
        if held is not None:
            off_p, off_q = self.world._attach_pos, self.world._attach_quat
            off_r = quat_to_mat(off_q)
            adr = self.world.info.cube_qpos_adr[held]
        stride = max(1, int(round(self.cfg.collision_check_dt / self.model.opt.timestep)))
        idx = list(range(0, len(traj.q), stride))
        if idx[-1] != len(traj.q) - 1:
            idx.append(len(traj.q) - 1)
        flanges = forward_batch(traj.q[idx])
        bad: set[str] = set()
        for k, i in enumerate(idx):
            d.qpos[self.world.info.robot_qpos_adr] = traj.q[i]
            if held is not None:
                rot = flanges[k, :3, :3]
                tcp = flanges[k, :3, 3] + TOOL_LENGTH * rot[:, 2]
                d.qpos[adr : adr + 3] = tcp + rot @ off_p
                d.qpos[adr + 3 : adr + 7] = mat_to_quat(rot @ off_r)
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_collision(self.model, d)
            bad.update(self._bad_contacts(d, held, allowed_cube))
        return sorted(bad)

    # ------------------------------------------------------------ pre-motion
    def check(self, traj: Trajectory, ctx, api) -> SafetyDecision:
        """Run every pre-motion check on a planned trajectory."""
        reasons: list[str] = []
        warnings: list[str] = []
        cfg = self.cfg
        # 6. Cell state.
        if getattr(api, "estop_active", False):
            reasons.append("E-stop active")
        if not getattr(api, "engaged", True):
            reasons.append("robot not engaged")
        # 1. Joint limits.
        if np.any(traj.q < self.q_lower - 1e-6) or np.any(traj.q > self.q_upper + 1e-6):
            reasons.append("joint limit exceeded")
        # 2. Workspace box on every TCP sample.
        lo = np.array([cfg.workspace_x[0], cfg.workspace_y[0], cfg.workspace_z[0]])
        hi = np.array([cfg.workspace_x[1], cfg.workspace_y[1], cfg.workspace_z[1]])
        if ctx.final_approach:
            lo[2] = 0.0
        outside = np.any((traj.tcp < lo - 1e-6) | (traj.tcp > hi + 1e-6), axis=1)
        if np.any(outside):
            p = traj.tcp[int(np.argmax(outside))]
            reasons.append(f"TCP leaves the workspace at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
        # 3. Speeds.
        v_lim, tcp_lim = self.motion.v_max, self.motion.tcp_speed_max
        req = ctx.requested_speed
        if req is not None:
            limit = v_lim if ctx.kind == "movej" else tcp_lim
            unit = "rad/s" if ctx.kind == "movej" else "m/s"
            if req > limit * cfg.speed_tolerance:
                reasons.append(f"requested speed {req:g} {unit} exceeds the limit {limit:g} {unit}")
        if traj.peak_joint_speed > v_lim * cfg.speed_tolerance:
            reasons.append(f"joint speed {traj.peak_joint_speed:.2f} rad/s > {v_lim:g} rad/s")
        if traj.peak_tcp_speed > tcp_lim * cfg.speed_tolerance:
            reasons.append(f"TCP speed {traj.peak_tcp_speed:.3f} m/s > {tcp_lim:g} m/s")
        # 4. Manipulability.
        w_min, _ = self.manip.min_along(traj.q, stride=10)
        if w_min < cfg.manip_block:
            reasons.append(f"manipulability {w_min:.4f} < {cfg.manip_block} (near singularity)")
        elif w_min < cfg.manip_warn:
            warnings.append(f"low manipulability {w_min:.4f}")
        # 5. Collisions (skipped when the target is already outside the safe envelope).
        if not reasons:
            allowed = ctx.target_part
            if ctx.final_approach and allowed is None and self.world.attached is None:
                allowed = self._target_cube(traj.tcp[-1])
            bad = self.replay_collisions(traj, allowed if ctx.final_approach else None)
            if bad:
                reasons.append("collision: " + ", ".join(bad))
        decision = SafetyDecision(not reasons, reasons, warnings, w_min)
        if reasons:
            self.blocked.append(reasons)
            log.info("blocked %s: %s", traj.label or traj.kind, "; ".join(reasons))
        return decision

    # ------------------------------------------------------------ runtime
    def monitor(self, ctx):
        """A runtime monitor for one motion: stops on any unplanned contact."""
        allowed = ctx.target_part  # for a pick without a known id, resolved from the live TCP

        def _monitor(backend, q_des) -> str | None:
            nonlocal allowed
            if ctx.final_approach and allowed is None and self.world.attached is None:
                allowed = self._target_cube(self.world.tcp_pose()[0])
            bad = self._bad_contacts(
                self.world.data, self.world.attached, allowed if ctx.final_approach else None
            )
            if bad:
                self.unplanned_contacts += 1
                return "unplanned contact " + ", ".join(bad)
            return None

        return _monitor
