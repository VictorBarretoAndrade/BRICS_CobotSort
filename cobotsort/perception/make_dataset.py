"""Synthetic dataset with domain randomization and labels from segmentation rendering.

Each cube geom's *visible* pixels in a segmentation render give its exact bounding box, so the
labels stay correct under occlusion (by other cubes or the robot arm). Boxes with fewer than
40 visible pixels are dropped.

Output (YOLO format)::

    data/synth/images/{train,val,test}/*.png
    data/synth/labels/{train,val,test}/*.txt   # "class cx cy w h", normalized
    data/synth/data.yaml
    data/synth/meta.json                       # camera pose + ground truth per image

``synth_hard`` is a separate test-only set with stronger lighting shifts (out of distribution).
"""

from __future__ import annotations

import colorsys
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cobotsort.config import CLASS_NAMES, CLASS_RGBA, CellConfig
from cobotsort.kinematics.ik import IKSolver, Pose
from cobotsort.sim.render import CellRenderer
from cobotsort.sim.scene import PARK_Z, park_position
from cobotsort.sim.world import SimWorld

log = logging.getLogger(__name__)


@dataclass
class RandomizationConfig:
    """Ranges of the domain randomization."""

    n_cubes: tuple[int, int] = (4, 12)
    region_x: tuple[float, float] = (0.25, 0.65)
    region_y: tuple[float, float] = (-0.27, 0.27)
    min_spacing: float = 0.06
    hue_jitter_deg: float = 8.0
    sat_scale: tuple[float, float] = (0.8, 1.15)
    val_scale: tuple[float, float] = (0.75, 1.15)
    light_tilt_deg: float = 35.0
    light_diffuse: tuple[float, float] = (0.30, 0.80)
    light_tint: float = 0.0  # max per-channel color shift of the lights
    ambient: tuple[float, float] = (0.15, 0.45)
    headlight_diffuse: tuple[float, float] = (0.10, 0.40)
    table_tint: float = 0.08
    cam_pos_jitter: float = 0.01  # m
    cam_rot_jitter_deg: float = 1.0
    p_robot: float = 0.30
    p_distractors: float = 0.20

    @classmethod
    def hard(cls) -> RandomizationConfig:
        """Stronger lighting shifts for the out-of-distribution test set."""
        return cls(
            light_tilt_deg=60.0,
            light_diffuse=(0.08, 1.30),
            light_tint=0.25,
            ambient=(0.02, 0.70),
            headlight_diffuse=(0.0, 0.6),
        )


class SceneRandomizer:
    """Places cubes, robot, distractors, lights and camera for one synthetic image."""

    def __init__(self, world: SimWorld, rand: RandomizationConfig) -> None:
        self.world = world
        self.model = world.model
        self.data = world.data
        self.rand = rand
        self.info = world.info
        m = self.model
        self.sun = m.light("sun").id
        self.top_cam = self.info.top_cam
        self.table = self.info.table_geom
        self.pick_area = m.geom("pick_area").id
        self.defaults = {
            "cam_pos": m.cam_pos[self.top_cam].copy(),
            "cam_quat": m.cam_quat[self.top_cam].copy(),
            "light_dir": m.light_dir[self.sun].copy(),
            "light_diffuse": m.light_diffuse[self.sun].copy(),
            "head_ambient": m.vis.headlight.ambient.copy(),
            "head_diffuse": m.vis.headlight.diffuse.copy(),
            "table_rgba": m.geom_rgba[self.table].copy(),
            "pick_rgba": m.geom_rgba[self.pick_area].copy(),
        }
        self.arm_poses = self._arm_poses(60)

    def _arm_poses(self, n: int) -> list[np.ndarray]:
        """Joint vectors that put the arm over the pick area (precomputed once)."""
        ik = IKSolver(self.info)
        rng = np.random.default_rng(123)
        out = []
        while len(out) < n:
            x, y = rng.uniform(0.30, 0.62), rng.uniform(-0.25, 0.25)
            z, yaw = rng.uniform(0.08, 0.40), rng.uniform(-np.pi, np.pi)
            sol = ik.closest(Pose.tool_down(x, y, z, yaw=yaw), self.info.home_q)
            if sol is not None:
                out.append(sol.q)
        return out

    def _jitter_rgba(self, rgba, rng) -> np.ndarray:
        r = self.rand
        h, s, v = colorsys.rgb_to_hsv(*rgba[:3])
        h = (h + rng.uniform(-r.hue_jitter_deg, r.hue_jitter_deg) / 360.0) % 1.0
        s = float(np.clip(s * rng.uniform(*r.sat_scale), 0, 1))
        v = float(np.clip(v * rng.uniform(*r.val_scale), 0, 1))
        return np.array([*colorsys.hsv_to_rgb(h, s, v), 1.0])

    def apply(self, rng: np.random.Generator) -> list[dict]:
        """Randomize the scene; returns the placed cubes (index, class, position)."""
        r, m, d = self.rand, self.model, self.data
        sc = self.world.cfg.scene
        n = int(rng.integers(r.n_cubes[0], r.n_cubes[1] + 1))
        pts: list[np.ndarray] = []
        while len(pts) < n:
            p = np.array([rng.uniform(*r.region_x), rng.uniform(*r.region_y)])
            if all(np.linalg.norm(p - q) >= r.min_spacing for q in pts):
                pts.append(p)
        cubes = []
        for i in range(sc.max_cubes):
            if i < n:
                cls = str(rng.choice(CLASS_NAMES))
                pos = np.array([pts[i][0], pts[i][1], sc.cube_edge / 2])
                self.world._set_cube_pose(i, pos, float(rng.uniform(0, np.pi / 2)))
                m.geom_rgba[self.info.cube_geom_ids[i]] = self._jitter_rgba(CLASS_RGBA[cls], rng)
                cubes.append({"index": i, "class": cls, "pos": pos.tolist()})
            else:
                self.world._set_cube_pose(i, park_position(i), 0.0)
        # Robot: over the pick area in some images, parked otherwise.
        if rng.random() < r.p_robot:
            q = self.arm_poses[int(rng.integers(len(self.arm_poses)))]
        else:
            q = self.info.home_q
        d.qpos[self.info.robot_qpos_adr] = q
        # Gray distractor cylinders (unlabeled).
        for k, mocap in enumerate(self.info.distractor_mocap_ids):
            gid = self.info.distractor_geom_ids[k]
            if rng.random() < r.p_distractors:
                for _ in range(50):
                    p = np.array([rng.uniform(*r.region_x), rng.uniform(*r.region_y)])
                    if all(np.linalg.norm(p - q) >= r.min_spacing for q in pts):
                        break
                d.mocap_pos[mocap] = [p[0], p[1], 0.03]
                g = rng.uniform(0.35, 0.75)
                m.geom_rgba[gid] = [g, g, g, 1.0]
            else:
                d.mocap_pos[mocap] = [3.0 + 0.2 * k, -3.0, PARK_Z]
        # Lights.
        df = self.defaults
        tilt = np.radians(rng.uniform(0, r.light_tilt_deg))
        az = rng.uniform(-np.pi, np.pi)
        m.light_dir[self.sun] = [
            np.sin(tilt) * np.cos(az),
            np.sin(tilt) * np.sin(az),
            -np.cos(tilt),
        ]
        tint = rng.uniform(-r.light_tint, r.light_tint, 3) if r.light_tint else np.zeros(3)
        m.light_diffuse[self.sun] = np.clip(rng.uniform(*r.light_diffuse) + tint, 0, 1.5)
        m.vis.headlight.ambient = np.clip(rng.uniform(*r.ambient) + tint * 0.5, 0, 1)
        m.vis.headlight.diffuse = np.full(3, rng.uniform(*r.headlight_diffuse))
        # Table tint.
        m.geom_rgba[self.table][:3] = np.clip(
            df["table_rgba"][:3] + rng.uniform(-r.table_tint, r.table_tint, 3), 0, 1
        )
        m.geom_rgba[self.pick_area][:3] = np.clip(
            df["pick_rgba"][:3] + rng.uniform(-r.table_tint, r.table_tint), 0, 1
        )
        # Camera pose jitter.
        m.cam_pos[self.top_cam] = df["cam_pos"] + rng.uniform(
            -r.cam_pos_jitter, r.cam_pos_jitter, 3
        )
        rot = Rotation.from_euler("xyz", rng.uniform(-1, 1, 3) * r.cam_rot_jitter_deg, degrees=True)
        base = Rotation.from_quat(df["cam_quat"][[1, 2, 3, 0]])
        q = (base * rot).as_quat()
        m.cam_quat[self.top_cam] = [q[3], q[0], q[1], q[2]]
        mujoco.mj_forward(m, d)
        return cubes

    def restore(self) -> None:
        m, df = self.model, self.defaults
        m.cam_pos[self.top_cam] = df["cam_pos"]
        m.cam_quat[self.top_cam] = df["cam_quat"]
        m.light_dir[self.sun] = df["light_dir"]
        m.light_diffuse[self.sun] = df["light_diffuse"]
        m.vis.headlight.ambient = df["head_ambient"]
        m.vis.headlight.diffuse = df["head_diffuse"]
        m.geom_rgba[self.table] = df["table_rgba"]
        m.geom_rgba[self.pick_area] = df["pick_rgba"]


def labels_from_segmentation(
    seg: np.ndarray, cubes: list[dict], cube_geoms: np.ndarray, min_px: int
) -> list[dict]:
    """Exact boxes of the visible pixels of each cube."""
    out = []
    for c in cubes:
        mask = seg == int(cube_geoms[c["index"]])
        n_px = int(mask.sum())
        if n_px < min_px:
            continue
        ys, xs = np.nonzero(mask)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        out.append({**c, "bbox": box, "visible_px": n_px})
    return out


def generate(
    out_dir: Path,
    n: int,
    seed: int,
    rand: RandomizationConfig,
    splits: dict[str, float] | None = None,
    cfg: CellConfig | None = None,
) -> dict:
    """Render ``n`` labeled images into ``out_dir`` (YOLO layout). Returns summary stats."""
    cfg = cfg or CellConfig()
    splits = splits or {"train": 0.70, "val": 0.15, "test": 0.15}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    world = SimWorld(cfg)
    world.reset(seed=0, classes=["red_cube"])
    randomizer = SceneRandomizer(world, rand)
    renderer = CellRenderer(world.model)
    w, h = cfg.scene.top_cam_size
    rng = np.random.default_rng(seed)
    names = list(splits)
    bounds = np.cumsum([splits[k] for k in names]) * n
    meta: dict[str, dict] = {}
    counts = {k: 0 for k in names}
    n_boxes = 0
    class_counts = {c: 0 for c in CLASS_NAMES}
    for k in names:
        (out_dir / "images" / k).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / k).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        split = names[int(np.searchsorted(bounds, i, side="right"))]
        cubes = randomizer.apply(rng)
        rgb = renderer.rgb(world.data, "top_cam", w, h)
        seg = renderer.segmentation(world.data, "top_cam", w, h)
        labels = labels_from_segmentation(
            seg, cubes, world.info.cube_geom_ids, cfg.vision.min_visible_px
        )
        stem = f"{i:05d}"
        cv2.imwrite(
            str(out_dir / "images" / split / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        lines = []
        for lab in labels:
            x1, y1, x2, y2 = lab["bbox"]
            cls_id = CLASS_NAMES.index(lab["class"])
            lines.append(
                f"{cls_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
            class_counts[lab["class"]] += 1
        (out_dir / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        cam = world.info.top_cam
        meta[stem] = {
            "split": split,
            "cam_pos": world.data.cam_xpos[cam].tolist(),
            "cam_mat": world.data.cam_xmat[cam].tolist(),
            "objects": labels,
        }
        counts[split] += 1
        n_boxes += len(labels)
        if (i + 1) % 100 == 0:
            log.info("%s: %d/%d images", out_dir.name, i + 1, n)
    randomizer.restore()
    renderer.close()
    yaml = [f"path: {out_dir.resolve().as_posix()}"]
    for k in names:
        yaml.append(f"{k}: images/{k}")
    yaml.append("names:")
    yaml += [f"  {i}: {c}" for i, c in enumerate(CLASS_NAMES)]
    (out_dir / "data.yaml").write_text("\n".join(yaml) + "\n")
    (out_dir / "meta.json").write_text(json.dumps(meta))
    summary = {"images": counts, "boxes": n_boxes, "per_class": class_counts, "seed": seed}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s: %s", out_dir, summary)
    return summary


def make_all(n: int = 800, n_hard: int = 150, seed: int = 0, cfg: CellConfig | None = None) -> dict:
    cfg = cfg or CellConfig()
    main = generate(cfg.vision.dataset_dir, n, seed, RandomizationConfig(), cfg=cfg)
    hard = generate(
        cfg.vision.hard_dir,
        n_hard,
        seed + 1,
        RandomizationConfig.hard(),
        splits={"test": 1.0},
        cfg=cfg,
    )
    return {"synth": main, "synth_hard": hard}
