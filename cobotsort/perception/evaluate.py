"""Evaluate the HSV baseline and YOLO on the synthetic test set and the hard (OOD) set.

Reports, per detector and set: precision and recall at the cell's operating point
(confidence >= 0.6, IoU >= 0.5), mAP@50 (and mAP@50-95), CPU inference time per frame, and
the localization error in mm after ``pixel_to_world`` (using each image's true camera pose,
so camera jitter does not count as detector error). All metrics are computed here with the
same code for both detectors (all-point interpolated AP, as in VOC/COCO).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cobotsort.config import CLASS_NAMES, ROOT, CellConfig
from cobotsort.perception.camera import CameraModel
from cobotsort.perception.detector import Detection, Detector

log = logging.getLogger(__name__)

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_image(dets: list[Detection], gts: list[dict], thr: float) -> tuple[np.ndarray, list[int]]:
    """Greedy, class-aware matching by descending score. Returns TP flags and matched GT index."""
    order = np.argsort([-d.confidence for d in dets], kind="stable")
    used: set[int] = set()
    tp = np.zeros(len(dets), bool)
    gt_idx = [-1] * len(dets)
    for i in order:
        d = dets[i]
        best, best_iou = -1, thr
        for j, g in enumerate(gts):
            if j in used or g["class"] != d.class_name:
                continue
            v = iou(d.bbox_xyxy, g["bbox"])
            if v >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            used.add(best)
            tp[i] = True
            gt_idx[i] = best
    return tp, gt_idx


@dataclass
class SetResult:
    detector: str
    set_name: str
    images: int
    gt_boxes: int
    precision: float
    recall: float
    map50: float
    map50_95: float
    ms_per_frame: float
    loc_mean_mm: float
    loc_p95_mm: float
    per_class_ap50: dict[str, float]
    confusion: list[list[int]]


def load_set(root: Path, split: str) -> list[tuple[Path, dict]]:
    meta = json.loads((root / "meta.json").read_text())
    return [
        (root / "images" / split / f"{k}.png", v)
        for k, v in sorted(meta.items())
        if v["split"] == split
    ]


def confusion(dets: list[Detection], gts: list[dict], thr: float = 0.5) -> np.ndarray:
    """Class-agnostic matching; rows = true class (+ background), cols = predicted."""
    n = len(CLASS_NAMES)
    cm = np.zeros((n + 1, n + 1), int)
    used: set[int] = set()
    for d in sorted(dets, key=lambda d: -d.confidence):
        best, best_iou = -1, thr
        for j, g in enumerate(gts):
            if j not in used:
                v = iou(d.bbox_xyxy, g["bbox"])
                if v >= best_iou:
                    best, best_iou = j, v
        p = CLASS_NAMES.index(d.class_name)
        if best >= 0:
            used.add(best)
            cm[CLASS_NAMES.index(gts[best]["class"]), p] += 1
        else:
            cm[n, p] += 1
    for j, g in enumerate(gts):
        if j not in used:
            cm[CLASS_NAMES.index(g["class"]), n] += 1
    return cm


def evaluate_set(
    detector: Detector,
    items: list[tuple[Path, dict]],
    set_name: str,
    operating_conf: float = 0.6,
    full_curve_conf: float | None = None,
    cfg: CellConfig | None = None,
) -> SetResult:
    cfg = cfg or CellConfig()
    w, h = cfg.scene.top_cam_size
    z_center = cfg.scene.cube_edge / 2
    all_dets: list[list[Detection]] = []
    times: list[float] = []
    for k, (path, _meta) in enumerate(items):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        dets = detector.detect(rgb)
        dt = time.perf_counter() - t0
        if k >= 3:  # skip warm-up frames
            times.append(dt)
        if full_curve_conf is not None:
            dets = detector.detect(rgb, conf=full_curve_conf)  # type: ignore[call-arg]
        all_dets.append(dets)
    n_gt = sum(len(m["objects"]) for _, m in items)
    # mAP over IoU thresholds.
    aps = {}
    for thr in IOU_THRESHOLDS:
        per_class = {}
        flags = [
            match_image(d, m["objects"], thr) for d, (_, m) in zip(all_dets, items, strict=True)
        ]
        for c in CLASS_NAMES:
            scores, tps = [], []
            for dets, (tp, _) in zip(all_dets, flags, strict=True):
                for d, t in zip(dets, tp, strict=True):
                    if d.class_name == c:
                        scores.append(d.confidence)
                        tps.append(t)
            n_c = sum(1 for _, m in items for g in m["objects"] if g["class"] == c)
            per_class[c] = average_precision(np.array(scores), np.array(tps, bool), n_c)
        aps[float(thr)] = per_class
    map50 = float(np.nanmean(list(aps[0.5].values())))
    map5095 = float(np.mean([np.nanmean(list(v.values())) for v in aps.values()]))
    # Operating point: precision, recall, localization, confusion.
    tp_n = fp_n = 0
    loc = []
    cm = np.zeros((len(CLASS_NAMES) + 1,) * 2, int)
    for dets, (_, meta) in zip(all_dets, items, strict=True):
        op = [d for d in dets if d.confidence >= operating_conf]
        tp, gt_idx = match_image(op, meta["objects"], 0.5)
        tp_n += int(tp.sum())
        fp_n += int((~tp).sum())
        cam = CameraModel(
            w,
            h,
            cfg.scene.top_cam_fovy,
            np.array(meta["cam_pos"]),
            np.array(meta["cam_mat"]).reshape(3, 3),
        )
        for d, j in zip(op, gt_idx, strict=True):
            if j >= 0:
                p = cam.pixel_to_world(d.center_px[0], d.center_px[1], z_center)
                gt = meta["objects"][j]["pos"]
                loc.append(float(np.hypot(p[0] - gt[0], p[1] - gt[1])) * 1e3)
        cm += confusion(op, meta["objects"])
    return SetResult(
        detector=detector.name,
        set_name=set_name,
        images=len(items),
        gt_boxes=n_gt,
        precision=tp_n / max(tp_n + fp_n, 1),
        recall=tp_n / max(n_gt, 1),
        map50=map50,
        map50_95=map5095,
        ms_per_frame=1e3 * float(np.mean(times)) if times else float("nan"),
        loc_mean_mm=float(np.mean(loc)) if loc else float("nan"),
        loc_p95_mm=float(np.percentile(loc, 95)) if loc else float("nan"),
        per_class_ap50={k: float(v) for k, v in aps[0.5].items()},
        confusion=cm.tolist(),
    )


def plot_confusions(results: list[SetResult], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [c.replace("_cube", "") for c in CLASS_NAMES] + ["background"]
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.4))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results, strict=True):
        cm = np.array(r.confusion)
        ax.imshow(np.log1p(cm), cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 3 else "black",
                    )
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{r.detector} — {r.set_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run(cfg: CellConfig | None = None, out_dir: Path | None = None) -> list[SetResult]:
    from cobotsort.perception.hsv import HSVDetector
    from cobotsort.perception.yolo import YoloDetector

    cfg = cfg or CellConfig()
    out_dir = out_dir or ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = {
        "synth test": load_set(cfg.vision.dataset_dir, "test"),
        "synth_hard": load_set(cfg.vision.hard_dir, "test"),
    }
    detectors: list[tuple[Detector, float | None]] = [
        (HSVDetector(), None),
        (YoloDetector(cfg.vision.yolo_weights, device="cpu"), 0.001),
    ]
    results = []
    for det, curve_conf in detectors:
        for name, items in sets.items():
            log.info("evaluating %s on %s (%d images)", det.name, name, len(items))
            results.append(
                evaluate_set(det, items, name, cfg.vision.min_confidence, curve_conf, cfg)
            )
    plot_confusions(
        [r for r in results if r.set_name == "synth test"]
        + [r for r in results if r.set_name == "synth_hard"],
        out_dir / "confusion_matrix.png",
    )
    (out_dir / "perception.json").write_text(json.dumps([r.__dict__ for r in results], indent=2))
    write_markdown(results, out_dir / "perception.md")
    return results


def write_markdown(results: list[SetResult], path: Path) -> None:
    import platform

    cpu = platform.processor() or platform.machine()
    lines = [
        "# Perception results",
        "",
        "Command: `cobotsort eval-vision` (after `cobotsort dataset` and `cobotsort train`).",
        "",
        "Operating point: confidence ≥ 0.6 (the cell's threshold) and IoU ≥ 0.5. mAP is all-point",
        "interpolated AP averaged over the 4 classes (mAP@50-95: IoU 0.50:0.05:0.95),",
        "computed by `cobotsort/perception/evaluate.py` for both detectors. Localization error",
        "is the distance between `pixel_to_world(box center, z = 2 cm)` and the true cube",
        "center, using each image's true camera pose. Inference time is measured on the CPU",
        f"({cpu}), excluding the first 3 frames.",
        "",
        "| Detector | Set | Images | Boxes | Precision | Recall | mAP@50 | mAP@50-95 | CPU ms/frame"
        " | Loc. error mean / p95 (mm) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m5095 = f"{r.map50_95:.3f}" if r.detector == "yolo" else "—"
        lines.append(
            f"| {r.detector} | {r.set_name} | {r.images} | {r.gt_boxes} | {r.precision:.3f}"
            f" | {r.recall:.3f} | {r.map50:.3f} | {m5095} | {r.ms_per_frame:.1f}"
            f" | {r.loc_mean_mm:.2f} / {r.loc_p95_mm:.2f} |"
        )
    lines += [
        "",
        "AP@50 per class:",
        "",
        "| Detector | Set | " + " | ".join(CLASS_NAMES) + " |",
        "|---|---|" + "---|" * len(CLASS_NAMES),
    ]
    for r in results:
        lines.append(
            f"| {r.detector} | {r.set_name} | "
            + " | ".join(f"{r.per_class_ap50[c]:.3f}" for c in CLASS_NAMES)
            + " |"
        )
    lines += [
        "",
        "Confusion matrices (operating point): `results/confusion_matrix.png`.",
        "Training curves (train vs val loss, val mAP): `results/yolo_training_curves.png`.",
        "",
        "The cubes are small in the 640×480 top image (about 25–30 px across). Small objects",
        "are a known weakness of one-stage detectors; a two-stage detector (e.g. Faster R-CNN)",
        "is a planned comparison.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
