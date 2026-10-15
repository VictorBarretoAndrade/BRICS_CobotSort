# Perception results

Command: `cobotsort eval-vision` (after `cobotsort dataset` and `cobotsort train`).

Operating point: confidence ≥ 0.6 (the cell's threshold) and IoU ≥ 0.5. mAP is all-point
interpolated AP averaged over the 4 classes (mAP@50-95: IoU 0.50:0.05:0.95),
computed by `cobotsort/perception/evaluate.py` for both detectors. Localization error
is the distance between `pixel_to_world(box center, z = 2 cm)` and the true cube
center, using each image's true camera pose. Inference time is measured on the CPU
(Intel64 Family 6 Model 183 Stepping 1, GenuineIntel), excluding the first 3 frames.

| Detector | Set | Images | Boxes | Precision | Recall | mAP@50 | mAP@50-95 | CPU ms/frame | Loc. error mean / p95 (mm) |
|---|---|---|---|---|---|---|---|---|---|
| hsv | synth test | 120 | 901 | 1.000 | 0.977 | 0.992 | — | 2.1 | 1.64 / 3.93 |
| hsv | synth_hard | 150 | 1126 | 0.989 | 0.927 | 0.942 | — | 2.1 | 1.97 / 5.42 |
| yolo | synth test | 120 | 901 | 1.000 | 0.998 | 1.000 | 0.986 | 22.1 | 1.45 / 1.65 |
| yolo | synth_hard | 150 | 1126 | 0.989 | 0.978 | 0.991 | 0.950 | 21.8 | 1.38 / 2.37 |

AP@50 per class:

| Detector | Set | red_cube | green_cube | blue_cube | defect |
|---|---|---|---|---|---|
| hsv | synth test | 0.972 | 0.995 | 1.000 | 1.000 |
| hsv | synth_hard | 0.945 | 0.942 | 0.972 | 0.908 |
| yolo | synth test | 1.000 | 1.000 | 1.000 | 1.000 |
| yolo | synth_hard | 0.995 | 1.000 | 0.992 | 0.979 |

Confusion matrices (operating point): `results/confusion_matrix.png`.
Training curves (train vs val loss, val mAP): `results/yolo_training_curves.png`.

The cubes are small in the 640×480 top image (about 25–30 px across). Small objects
are a known weakness of one-stage detectors; a two-stage detector (e.g. Faster R-CNN)
is a planned comparison.
