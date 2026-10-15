"""Camera model, detectors, dataset labels and metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cobotsort.config import CLASS_NAMES, CellConfig
from cobotsort.perception.camera import CameraModel
from cobotsort.perception.evaluate import average_precision, iou
from cobotsort.sim.render import CellRenderer
from cobotsort.sim.world import SimWorld


@pytest.fixture(scope="module")
def scene():
    world = SimWorld()
    world.reset(seed=3)
    w, h = world.cfg.scene.top_cam_size
    cam = CameraModel.from_sim(world.model, world.data, "top_cam", w, h)
    renderer = CellRenderer(world.model)
    rgb = renderer.rgb(world.data, "top_cam", w, h)
    seg = renderer.segmentation(world.data, "top_cam", w, h)
    yield world, cam, rgb, seg
    renderer.close()


def test_projection_roundtrip_under_3mm(scene):
    world, cam, _, _ = scene
    for cube in world.cubes():
        uv = cam.world_to_pixel(cube.pos[None])[0]
        back = cam.pixel_to_world(uv[0], uv[1], cube.pos[2])
        assert np.linalg.norm(back[:2] - cube.pos[:2]) < 3e-3


def test_projection_matches_the_render(scene):
    """Intrinsics and the v-axis flip agree with MuJoCo: projected centers sit on the cubes."""
    world, cam, _, seg = scene
    for cube in world.cubes():
        ys, xs = np.nonzero(seg == world.info.cube_geom_ids[cube.index])
        center = np.array([(xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2])
        uv = cam.world_to_pixel(cube.pos[None])[0]
        assert np.linalg.norm(uv - center) < 1.5


def test_hsv_finds_every_cube_within_6mm(scene):
    from cobotsort.perception.hsv import HSVDetector

    world, cam, rgb, _ = scene
    dets = [d.to_world(cam, world.cfg.scene.cube_edge / 2) for d in HSVDetector().detect(rgb)]
    cubes = world.cubes()
    assert len(dets) == len(cubes)
    for cube in cubes:
        d = min(dets, key=lambda d: np.hypot(*(np.array(d.world_xy) - cube.pos[:2])))
        assert d.class_name == cube.class_name
        assert np.hypot(*(np.array(d.world_xy) - cube.pos[:2])) < 6e-3


def test_dataset_labels_are_valid(tmp_path):
    from cobotsort.perception.make_dataset import RandomizationConfig, generate

    summary = generate(tmp_path / "ds", 6, seed=5, rand=RandomizationConfig(), cfg=CellConfig())
    meta = json.loads((tmp_path / "ds" / "meta.json").read_text())
    assert sum(summary["images"].values()) == 6
    for stem, m in meta.items():
        lines = (tmp_path / "ds" / "labels" / m["split"] / f"{stem}.txt").read_text().split()
        rows = np.array(lines, float).reshape(-1, 5) if lines else np.zeros((0, 5))
        assert len(rows) == len(m["objects"])
        assert np.all(rows[:, 0] < len(CLASS_NAMES))
        assert np.all((rows[:, 1:] > 0) & (rows[:, 1:] <= 1))
        for obj in m["objects"]:
            assert obj["visible_px"] >= 40


def test_average_precision_and_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    # 2 GT; detections: TP (0.9), FP (0.8), TP (0.7) -> AP = 0.5 * 1 + 0.5 * 2/3
    ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
    assert ap == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert average_precision(np.array([]), np.array([], bool), 3) == 0.0


def test_yolo_finds_every_cube(scene):
    cfg = CellConfig()
    if not cfg.vision.yolo_weights.exists():
        pytest.skip("no trained weights (run `cobotsort train`)")
    from cobotsort.perception.yolo import YoloDetector

    world, cam, rgb, _ = scene
    dets = [d for d in YoloDetector(cfg.vision.yolo_weights).detect(rgb) if d.confidence >= 0.6]
    assert len(dets) == len(world.cubes())
    for d in dets:
        d.to_world(cam, world.cfg.scene.cube_edge / 2)
        cube = min(world.cubes(), key=lambda c: np.hypot(*(np.array(d.world_xy) - c.pos[:2])))
        assert d.class_name == cube.class_name
        assert np.hypot(*(np.array(d.world_xy) - cube.pos[:2])) < 6e-3
