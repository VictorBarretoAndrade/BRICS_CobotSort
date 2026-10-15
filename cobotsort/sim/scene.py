"""Builds the MuJoCo model of the sorting cell with ``mujoco.MjSpec``.

The model contains the Menagerie UR5e with a suction cup, a table, a gray pick
area, 12 pre-created cubes (unused ones are parked out of view), four open
bins, a three-light signal tower, a top camera for perception and a
perspective camera for video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from cobotsort.config import SceneConfig

log = logging.getLogger(__name__)

ROBOT_BODIES: tuple[str, ...] = (
    "base",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "vacuum_cup",
)
ROBOT_JOINTS: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
LAMP_COLORS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    # name: (on rgba, dimmed rgba)
    "GREEN": (np.array([0.1, 1.0, 0.15, 1.0]), np.array([0.08, 0.25, 0.08, 1.0])),
    "YELLOW": (np.array([1.0, 0.85, 0.05, 1.0]), np.array([0.28, 0.25, 0.06, 1.0])),
    "RED": (np.array([1.0, 0.08, 0.05, 1.0]), np.array([0.28, 0.06, 0.05, 1.0])),
}
PARK_Z = -1.0


def park_position(index: int) -> np.ndarray:
    """Out-of-view parking spot for unused cube ``index``."""
    return np.array([3.0 + 0.2 * index, 3.0, PARK_Z])


def look_at_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quaternion for a MuJoCo camera at ``pos`` looking at ``target`` (z up).

    A MuJoCo camera looks along its -z axis with +y up in the image.
    """
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    mat = np.column_stack([right, up, -forward])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


@dataclass
class SceneInfo:
    """Names and ids resolved after compilation."""

    model: mujoco.MjModel
    robot_qpos_adr: np.ndarray
    robot_dof_adr: np.ndarray
    robot_body_ids: np.ndarray
    robot_geom_ids: np.ndarray
    actuator_ids: np.ndarray
    tcp_site: int
    flange_site: int
    cube_body_ids: np.ndarray
    cube_geom_ids: np.ndarray
    cube_qpos_adr: np.ndarray
    cube_dof_adr: np.ndarray
    table_geom: int
    bin_geom_ids: dict[str, np.ndarray]
    obstacle_geom_ids: np.ndarray
    lamp_geom_ids: dict[str, int]
    distractor_mocap_ids: np.ndarray
    distractor_geom_ids: np.ndarray
    top_cam: int
    persp_cam: int
    home_q: np.ndarray


def _add_box(body, name: str, size, pos, rgba, collide: bool = True, group: int = 0):
    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = list(size)
    geom.pos = list(pos)
    geom.rgba = list(rgba)
    geom.group = group
    geom.contype = 1 if collide else 0
    geom.conaffinity = 1 if collide else 0
    return geom


def _add_cylinder(body, name: str, radius: float, half: float, pos, rgba, collide: bool = True):
    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.size = [radius, half, 0.0]
    geom.pos = list(pos)
    geom.rgba = list(rgba)
    geom.contype = 1 if collide else 0
    geom.conaffinity = 1 if collide else 0
    return geom


def build_spec(cfg: SceneConfig | None = None) -> mujoco.MjSpec:
    """Return the editable spec of the full cell."""
    cfg = cfg or SceneConfig()
    if not cfg.robot_xml.exists():
        raise FileNotFoundError(
            f"{cfg.robot_xml} not found. Run `python scripts/fetch_assets.py` first."
        )
    spec = mujoco.MjSpec.from_file(str(cfg.robot_xml))
    spec.modelname = "cobotsort_cell"
    spec.option.timestep = cfg.timestep

    # Offscreen buffer large enough for the 1280x720 video camera.
    spec.visual.global_.offwidth = max(cfg.persp_cam_size[0], cfg.top_cam_size[0])
    spec.visual.global_.offheight = max(cfg.persp_cam_size[1], cfg.top_cam_size[1])
    spec.visual.headlight.ambient = [0.30, 0.30, 0.30]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 4096

    # Replace the Menagerie spotlight with our own lighting.
    for light in spec.lights:
        light.active = False

    # Gravity compensation on the arm, as a cobot controller would do.
    for name in ROBOT_BODIES[1:-1]:
        spec.body(name).gravcomp = 1.0

    world = spec.worldbody

    sun = world.add_light()
    sun.name = "sun"
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.pos = [0.4, -0.3, 2.0]
    sun.dir = [-0.15, 0.1, -1.0]
    sun.diffuse = [0.50, 0.50, 0.50]
    sun.specular = [0.1, 0.1, 0.1]
    sun.castshadow = True

    # Floor (visual only, so parked cubes never touch it) and table.
    floor_tex = spec.add_texture()
    floor_tex.name = "floor_tex"
    floor_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_tex.rgb1 = [0.32, 0.33, 0.36]
    floor_tex.rgb2 = [0.27, 0.28, 0.31]
    floor_tex.width = 512
    floor_tex.height = 512
    floor_mat = spec.add_material()
    floor_mat.name = "floor_mat"
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    floor_mat.texrepeat = [6, 6]
    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [4.0, 4.0, 0.1]
    floor.pos = [0.0, 0.0, -0.75]
    floor.material = "floor_mat"
    floor.contype = 0
    floor.conaffinity = 0

    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.55, 0.62, 0.70]
    sky.rgb2 = [0.12, 0.14, 0.18]
    sky.width = 256
    sky.height = 256

    table = world.add_body()
    table.name = "table"
    _add_box(table, "table", (0.55, 0.75, 0.02), (0.25, 0.0, -0.02), (0.80, 0.78, 0.74, 1.0))
    for i, (lx, ly) in enumerate([(-0.27, -0.72), (-0.27, 0.72), (0.77, -0.72), (0.77, 0.72)]):
        _add_box(
            table,
            f"table_leg_{i}",
            (0.025, 0.025, 0.365),
            (lx, ly, -0.405),
            (0.35, 0.35, 0.38, 1.0),
            collide=False,
        )
    # Pick area marker (visual only).
    px, py = cfg.pick_x, cfg.pick_y
    _add_box(
        table,
        "pick_area",
        ((px[1] - px[0]) / 2 + 0.03, (py[1] - py[0]) / 2 + 0.03, 0.0005),
        ((px[0] + px[1]) / 2, (py[0] + py[1]) / 2, 0.0005),
        (0.56, 0.57, 0.59, 1.0),
        collide=False,
    )

    # Bins: dark gray boxes with a small colored tag on the far wall.
    tag_rgba = {
        "A": (0.85, 0.15, 0.15, 1.0),
        "B": (0.15, 0.70, 0.25, 1.0),
        "C": (0.15, 0.30, 0.85, 1.0),
        "REJECT": (0.55, 0.18, 0.70, 1.0),
    }
    half_in = cfg.bin_inner / 2
    t = cfg.bin_wall_thickness
    h = cfg.bin_wall_height
    half_out = half_in + t
    bin_rgba = (0.24, 0.25, 0.28, 1.0)
    for b in cfg.bins:
        body = world.add_body()
        body.name = f"bin_{b.name}"
        body.pos = [b.x, b.y, 0.0]
        _add_box(
            body,
            f"bin_{b.name}_floor",
            (half_out, half_out, cfg.bin_floor_thickness / 2),
            (0, 0, cfg.bin_floor_thickness / 2),
            bin_rgba,
        )
        _add_box(
            body,
            f"bin_{b.name}_wall_yp",
            (half_out, t / 2, h / 2),
            (0, half_in + t / 2, h / 2),
            bin_rgba,
        )
        _add_box(
            body,
            f"bin_{b.name}_wall_yn",
            (half_out, t / 2, h / 2),
            (0, -half_in - t / 2, h / 2),
            bin_rgba,
        )
        _add_box(
            body,
            f"bin_{b.name}_wall_xp",
            (t / 2, half_in, h / 2),
            (half_in + t / 2, 0, h / 2),
            bin_rgba,
        )
        _add_box(
            body,
            f"bin_{b.name}_wall_xn",
            (t / 2, half_in, h / 2),
            (-half_in - t / 2, 0, h / 2),
            bin_rgba,
        )
        far = 1.0 if b.y > 0 else -1.0
        _add_box(
            body,
            f"bin_{b.name}_tag",
            (0.03, 0.002, 0.012),
            (0, far * (half_out + 0.002), h / 2),
            tag_rgba[b.name],
            collide=False,
        )

    # Signal tower.
    lamp = world.add_body()
    lamp.name = "signal_tower"
    lamp.pos = [cfg.lamp_xy[0], cfg.lamp_xy[1], 0.0]
    _add_cylinder(lamp, "lamp_pole", 0.012, 0.05, (0, 0, 0.05), (0.2, 0.2, 0.22, 1.0))
    for i, name in enumerate(("GREEN", "YELLOW", "RED")):
        _add_cylinder(
            lamp,
            f"lamp_{name.lower()}",
            0.025,
            0.02,
            (0, 0, 0.12 + 0.04 * i),
            tuple(LAMP_COLORS[name][1]),
        )

    # Suction cup at the flange with a TCP site at the cup tip (z out of the tool).
    wrist3 = spec.body("wrist_3_link")
    cup = wrist3.add_body()
    cup.name = "vacuum_cup"
    cup.pos = [0.0, 0.1, 0.0]
    cup.quat = [-1.0, 1.0, 0.0, 0.0]
    cup.mass = 0.05
    cup.inertia = [1e-5, 1e-5, 1e-5]
    cup.explicitinertial = True
    _add_cylinder(cup, "cup", 0.02, 0.015, (0, 0, 0.015), (0.12, 0.12, 0.12, 1.0))
    tcp = cup.add_site()
    tcp.name = "tcp"
    tcp.pos = [0.0, 0.0, 0.03]
    tcp.size = [0.004, 0.004, 0.004]
    tcp.rgba = [1.0, 0.5, 0.0, 0.0]
    flange = cup.add_site()
    flange.name = "flange"
    flange.size = [0.004, 0.004, 0.004]
    flange.rgba = [0.0, 0.0, 0.0, 0.0]

    # Cubes, all parked; the episode reset places and colors them.
    for i in range(cfg.max_cubes):
        body = world.add_body()
        body.name = f"cube_{i}"
        body.pos = list(park_position(i))
        body.add_freejoint().name = f"cube_{i}_joint"
        geom = _add_box(
            body, f"cube_{i}", (cfg.cube_edge / 2,) * 3, (0, 0, 0), (0.5, 0.5, 0.5, 1.0)
        )
        geom.mass = cfg.cube_mass

    # Distractor cylinders (dataset only), mocap so they never move by physics.
    for i in range(cfg.n_distractors):
        body = world.add_body()
        body.name = f"distractor_{i}"
        body.mocap = True
        body.pos = [3.0 + 0.2 * i, -3.0, PARK_Z]
        _add_cylinder(
            body, f"distractor_{i}", 0.018, 0.03, (0, 0, 0), (0.5, 0.5, 0.5, 1.0), collide=False
        )

    # Cameras.
    top = world.add_camera()
    top.name = "top_cam"
    top.pos = list(cfg.top_cam_pos)
    top.quat = [1.0, 0.0, 0.0, 0.0]  # looks straight down, image up = +y, right = +x
    top.fovy = cfg.top_cam_fovy
    persp = world.add_camera()
    persp.name = "persp_cam"
    persp.pos = list(cfg.persp_cam_pos)
    persp.quat = list(look_at_quat(np.array(cfg.persp_cam_pos), np.array(cfg.persp_cam_target)))
    persp.fovy = cfg.persp_cam_fovy
    return spec


def build_model(cfg: SceneConfig | None = None) -> SceneInfo:
    """Compile the cell and resolve every id the rest of the code needs."""
    cfg = cfg or SceneConfig()
    model = build_spec(cfg).compile()

    def jid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def gid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def bid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    joints = [jid(n) for n in ROBOT_JOINTS]
    robot_body_ids = np.array([bid(n) for n in ROBOT_BODIES])
    robot_geom_ids = np.array(
        [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] in robot_body_ids
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
    )
    cube_body_ids = np.array([bid(f"cube_{i}") for i in range(cfg.max_cubes)])
    cube_joint_ids = [jid(f"cube_{i}_joint") for i in range(cfg.max_cubes)]
    bin_geom_ids = {
        b.name: np.array(
            [
                gid(f"bin_{b.name}_{part}")
                for part in ("floor", "wall_yp", "wall_yn", "wall_xp", "wall_xn")
            ]
        )
        for b in cfg.bins
    }
    obstacle = [gid("lamp_pole")] + [gid(f"lamp_{c}") for c in ("green", "yellow", "red")]
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    home_q = (
        model.key_qpos[home_key][[model.jnt_qposadr[j] for j in joints]].copy()
        if home_key >= 0
        else np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
    )
    info = SceneInfo(
        model=model,
        robot_qpos_adr=np.array([model.jnt_qposadr[j] for j in joints]),
        robot_dof_adr=np.array([model.jnt_dofadr[j] for j in joints]),
        robot_body_ids=robot_body_ids,
        robot_geom_ids=robot_geom_ids,
        actuator_ids=np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
            ]
        ),
        tcp_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp"),
        flange_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "flange"),
        cube_body_ids=cube_body_ids,
        cube_geom_ids=np.array([gid(f"cube_{i}") for i in range(cfg.max_cubes)]),
        cube_qpos_adr=np.array([model.jnt_qposadr[j] for j in cube_joint_ids]),
        cube_dof_adr=np.array([model.jnt_dofadr[j] for j in cube_joint_ids]),
        table_geom=gid("table"),
        bin_geom_ids=bin_geom_ids,
        obstacle_geom_ids=np.array(obstacle),
        lamp_geom_ids={c: gid(f"lamp_{c.lower()}") for c in ("GREEN", "YELLOW", "RED")},
        distractor_mocap_ids=np.array(
            [model.body_mocapid[bid(f"distractor_{i}")] for i in range(cfg.n_distractors)]
        ),
        distractor_geom_ids=np.array([gid(f"distractor_{i}") for i in range(cfg.n_distractors)]),
        top_cam=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam"),
        persp_cam=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "persp_cam"),
        home_q=home_q,
    )
    log.debug("compiled cell: nq=%d nv=%d ngeom=%d", model.nq, model.nv, model.ngeom)
    return info
