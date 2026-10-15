"""Fine-tune YOLO11n on the synthetic dataset and keep the training curves.

``--quick`` (imgsz 480, 12 epochs) is meant for CPU-only machines.
"""

from __future__ import annotations

import csv
import logging
import shutil
import sys
from pathlib import Path

from cobotsort.config import ROOT, VisionConfig

log = logging.getLogger(__name__)


def plot_curves(results_csv: Path, out_png: Path) -> None:
    """Train vs validation loss per epoch (box + cls + dfl), to check for overfitting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(results_csv.open()))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    epochs = [int(float(r["epoch"])) for r in rows]
    parts = ("box_loss", "cls_loss", "dfl_loss")
    train = [sum(float(r[f"train/{p}"]) for p in parts) for r in rows]
    val = [sum(float(r[f"val/{p}"]) for p in parts) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(epochs, train, label="train")
    ax[0].plot(epochs, val, label="val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss (box + cls + dfl)")
    ax[0].set_title("Training vs validation loss")
    ax[0].legend()
    ax[1].plot(epochs, [float(r["metrics/mAP50(B)"]) for r in rows], label="mAP@50")
    ax[1].plot(epochs, [float(r["metrics/mAP50-95(B)"]) for r in rows], label="mAP@50-95")
    ax[1].set_xlabel("epoch")
    ax[1].set_title("Validation mAP")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def train(
    data_yaml: Path | None = None,
    epochs: int = 40,
    imgsz: int = 640,
    quick: bool = False,
    device: str | None = None,
    batch: int = 16,
) -> Path:
    """Train from ``yolo11n.pt`` and copy the best weights to ``models/``."""
    from ultralytics import YOLO

    vision = VisionConfig()
    data_yaml = data_yaml or vision.dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found: run `cobotsort dataset` first")
    if quick:
        epochs, imgsz = 12, 480
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,  # None = auto (GPU if available)
        workers=0 if sys.platform == "win32" else 4,
        project=str(ROOT / "runs"),
        name="cobotsort_yolo11n",
        exist_ok=True,
        seed=0,
        deterministic=True,
        plots=True,
        verbose=False,
    )
    save_dir = (
        Path(results.save_dir) if results is not None else ROOT / "runs" / "cobotsort_yolo11n"
    )
    best = save_dir / "weights" / "best.pt"
    vision.yolo_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, vision.yolo_weights)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    shutil.copy(save_dir / "results.csv", out / "yolo_training.csv")
    plot_curves(save_dir / "results.csv", out / "yolo_training_curves.png")
    log.info("best weights -> %s", vision.yolo_weights)
    return vision.yolo_weights
