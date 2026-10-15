"""SimWorld: episode reset, stepping, ground truth, contacts and vacuum attach."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from cobotsort.config import CLASS_NAMES, CLASS_RGBA, CellConfig
from cobotsort.sim.scene import LAMP_COLORS, SceneInfo, build_model, park_position

log = logging.getLogger(__name__)


@dataclass
class CubeState:
    """Ground truth for one active cube."""

    index: int
    class_name: str
    pos: np.ndarray
    quat: np.ndarray

    @property
    def yaw(self) -> float:
        w, x, y, z = self.quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@dataclass
class Contact:
    """A contact between two geoms."""

    geom1: int
    geom2: int
    dist: float


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, float))
    return mat.reshape(3, 3)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, float).flatten())
    return quat


class SimWorld:
    """Owns the MjModel/MjData of the cell and the episode state."""

    def __init__(self, cfg: CellConfig | None = None) -> None:
        self.cfg = cfg or CellConfig()
        self.info: SceneInfo = build_model(self.cfg.scene)
        self.model: mujoco.MjModel = self.info.model
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(0)
        self.cube_classes: list[str | None] = [None] * self.cfg.scene.max_cubes
        self.attached: int | None = None
        self._attach_pos = np.zeros(3)
        self._attach_quat = np.array([1.0, 0, 0, 0])
        acts = self.info.actuator_ids
        self.kp = self.model.actuator_gainprm[acts, 0].copy()
        self.kv = -self.model.actuator_biasprm[acts, 2].copy()
        self.lamp_state = "YELLOW"
        self.seed: int | None = None
        self._refresh_parked()
        self.set_lamp("YELLOW")

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed: int | None = None,
        classes: list[str] | None = None,
        positions: list[tuple[float, float]] | None = None,
    ) -> list[CubeState]:
        """Start a new episode. Returns the ground truth of the active cubes."""
        sc = self.cfg.scene
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.attached = None
        if classes is None:
            n = int(self.rng.integers(sc.min_cubes, sc.max_cubes_episode + 1))
            if self.rng.random() < sc.defect_probability:
                classes = ["defect"] + list(self.rng.choice(CLASS_NAMES, size=n - 1))
            else:
                classes = list(self.rng.choice(CLASS_NAMES[:3], size=n))
            self.rng.shuffle(classes)
        n = len(classes)
        if n > sc.max_cubes:
            raise ValueError(f"at most {sc.max_cubes} cubes")
        if positions is None:
            positions = self._sample_positions(n)
        self.cube_classes = [None] * sc.max_cubes
        for i in range(sc.max_cubes):
            if i < n:
                self.cube_classes[i] = str(classes[i])
                yaw = float(self.rng.uniform(0, np.pi / 2))
                x, y = positions[i]
                self._set_cube_pose(i, np.array([x, y, sc.cube_edge / 2 + 5e-4]), yaw)
                self.model.geom_rgba[self.info.cube_geom_ids[i]] = CLASS_RGBA[str(classes[i])]
            else:
                self._set_cube_pose(i, park_position(i), 0.0)
                self.model.geom_rgba[self.info.cube_geom_ids[i]] = (0.5, 0.5, 0.5, 1.0)
        self._refresh_parked()
        self.set_robot_q(self.info.home_q)
        self.data.ctrl[self.info.actuator_ids] = self.info.home_q
        self.data.time = 0.0
        # Let the cubes settle on the table.
        for _ in range(100):
            self.step()
        self.data.time = 0.0
        self.set_lamp("YELLOW")
        return self.cubes()

    def _sample_positions(self, n: int) -> list[tuple[float, float]]:
        sc = self.cfg.scene
        margin = sc.cube_edge * 0.75
        for _ in range(200):
            pts: list[tuple[float, float]] = []
            for _ in range(2000):
                if len(pts) == n:
                    break
                p = (
                    float(self.rng.uniform(sc.pick_x[0] + margin, sc.pick_x[1] - margin)),
                    float(self.rng.uniform(sc.pick_y[0] + margin, sc.pick_y[1] - margin)),
                )
                if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= sc.min_cube_spacing for q in pts):
                    pts.append(p)
            if len(pts) == n:
                return pts
        raise RuntimeError("could not place cubes with the required spacing")

    def _set_cube_pose(self, i: int, pos: np.ndarray, yaw: float) -> None:
        adr = self.info.cube_qpos_adr[i]
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        dof = self.info.cube_dof_adr[i]
        self.data.qvel[dof : dof + 6] = 0.0

    def _refresh_parked(self) -> None:
        parked = [i for i, c in enumerate(self.cube_classes) if c is None]
        self._parked_qpos = (
            np.concatenate([np.arange(a, a + 7) for a in self.info.cube_qpos_adr[parked]]).astype(
                int
            )
            if parked
            else np.zeros(0, int)
        )
        self._parked_dof = (
            np.concatenate([np.arange(a, a + 6) for a in self.info.cube_dof_adr[parked]]).astype(
                int
            )
            if parked
            else np.zeros(0, int)
        )
        self._parked_vals = self.data.qpos[self._parked_qpos].copy()

    # ------------------------------------------------------------------ robot
    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.info.robot_qpos_adr].copy()

    @property
    def qd(self) -> np.ndarray:
        return self.data.qvel[self.info.robot_dof_adr].copy()

    def set_robot_q(self, q: np.ndarray) -> None:
        """Teleport the arm (kinematic) and recompute positions."""
        self.data.qpos[self.info.robot_qpos_adr] = q
        self.data.qvel[self.info.robot_dof_adr] = 0.0
        self.data.ctrl[self.info.actuator_ids] = q
        mujoco.mj_forward(self.model, self.data)
        self._update_attached()

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP position (3,) and rotation matrix (3, 3) in the world frame."""
        s = self.info.tcp_site
        return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy()

    def step(
        self,
        q_des: np.ndarray | None = None,
        qd_des: np.ndarray | None = None,
        kinematic: bool = False,
    ) -> None:
        """Advance one physics step, streaming a joint setpoint if given.

        In actuated mode the setpoint gets a velocity feed-forward term so the
        Menagerie position servos (force = kp(u - q) - kv qd) track without lag.
        In kinematic mode the arm joints are written directly.
        """
        ids = self.info
        if q_des is not None:
            qd = np.zeros(6) if qd_des is None else qd_des
            if kinematic:
                self.data.qpos[ids.robot_qpos_adr] = q_des
                self.data.qvel[ids.robot_dof_adr] = qd
                self.data.ctrl[ids.actuator_ids] = q_des
            else:
                self.data.ctrl[ids.actuator_ids] = q_des + (self.kv / self.kp) * qd
        mujoco.mj_step(self.model, self.data)
        if self._parked_qpos.size:
            self.data.qpos[self._parked_qpos] = self._parked_vals
            self.data.qvel[self._parked_dof] = 0.0
        if self.attached is not None:
            mujoco.mj_kinematics(self.model, self.data)
            self._update_attached()

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ vacuum
    def attach(self, cube: int) -> None:
        """Rigidly attach ``cube`` to the TCP (kinematic attach)."""
        pos, mat = self.tcp_pose()
        cpos, cquat = self.cube_pose(cube)
        self._attach_pos = mat.T @ (cpos - pos)
        self._attach_quat = mat_to_quat(mat.T @ quat_to_mat(cquat))
        self.attached = cube
        self._update_attached()

    def detach(self) -> int | None:
        cube, self.attached = self.attached, None
        if cube is not None:
            dof = self.info.cube_dof_adr[cube]
            self.data.qvel[dof : dof + 6] = 0.0
        return cube

    def _update_attached(self) -> None:
        if self.attached is None:
            return
        pos, mat = self.tcp_pose()
        adr = self.info.cube_qpos_adr[self.attached]
        self.data.qpos[adr : adr + 3] = pos + mat @ self._attach_pos
        self.data.qpos[adr + 3 : adr + 7] = mat_to_quat(mat @ quat_to_mat(self._attach_quat))
        dof = self.info.cube_dof_adr[self.attached]
        self.data.qvel[dof : dof + 6] = 0.0

    # ------------------------------------------------------------------ truth
    def cube_pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        adr = self.info.cube_qpos_adr[i]
        return self.data.qpos[adr : adr + 3].copy(), self.data.qpos[adr + 3 : adr + 7].copy()

    def cubes(self) -> list[CubeState]:
        out = []
        for i, c in enumerate(self.cube_classes):
            if c is not None:
                pos, quat = self.cube_pose(i)
                out.append(CubeState(i, c, pos, quat))
        return out

    def in_pick_area(self, pos: np.ndarray, margin: float = 0.03) -> bool:
        sc = self.cfg.scene
        return (
            sc.pick_x[0] - margin <= pos[0] <= sc.pick_x[1] + margin
            and sc.pick_y[0] - margin <= pos[1] <= sc.pick_y[1] + margin
            and pos[2] < 0.1
        )

    def bin_of(self, pos: np.ndarray) -> str | None:
        """Name of the bin whose inner volume contains ``pos``, if any."""
        sc = self.cfg.scene
        half = sc.bin_inner / 2
        for b in sc.bins:
            if (
                abs(pos[0] - b.x) <= half
                and abs(pos[1] - b.y) <= half
                and pos[2] < sc.bin_wall_height + 0.03
            ):
                return b.name
        return None

    def contacts(self) -> list[Contact]:
        d = self.data
        return [
            Contact(int(d.contact[i].geom1), int(d.contact[i].geom2), float(d.contact[i].dist))
            for i in range(d.ncon)
        ]

    def geom_name(self, g: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"

    # ------------------------------------------------------------------ lamp
    def set_lamp(self, state: str) -> None:
        """Light one lamp of the signal tower and dim the others."""
        if state not in LAMP_COLORS:
            raise ValueError(state)
        self.lamp_state = state
        for name, gid in self.info.lamp_geom_ids.items():
            on, off = LAMP_COLORS[name]
            self.model.geom_rgba[gid] = on if name == state else off
