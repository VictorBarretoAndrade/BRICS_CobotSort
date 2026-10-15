# Decisions and deviations from the spec

Each entry: what changed, why, and where it lives.

## M0 — Setup

- **Library versions.** Built and tested with MuJoCo 3.16.0, Ultralytics 8.4.177, Anthropic SDK 1.15.0,
  PyTorch 2.14.1 (CUDA 12.6 wheel; CPU works too), OpenCV 5.0, NumPy 2.5, Python 3.12.4 on Windows 11.
- **Home pose.** The Menagerie `home` keyframe is used as is
  (`q = [-π/2, -π/2, π/2, -π/2, -π/2, 0]`, tool pointing down at (-0.134, 0.492, 0.458)).
  It holds the arm over the +y side of the table, outside the top camera's field of view, so it
  doubles as the park pose for scanning. In this frame the arm points along -x at `q = 0`, so the
  pick area (+x) corresponds to a shoulder pan of about -π.
- **Menagerie spotlight.** Disabled; the cell uses one directional "sun" light (with shadows) plus
  the headlight, which the dataset generator randomizes.
- **Gravity compensation.** The arm bodies have `gravcomp = 1`, as a cobot controller compensates
  gravity internally. The Menagerie position servos compute `kp (u - q) - kv q̇`, which lags by
  `kv/kp = 0.2 s` while moving, so the executor adds a velocity feed-forward
  (`u = q_des + (kv/kp) q̇_des`).
- **Parked cubes.** The 12 cube bodies are pre-created. Unused ones are parked at z = -1 and pinned
  there every step (they have free joints, so they would otherwise fall). The floor plane is visual
  only so parked cubes never collide with it.
- **Bins are dark gray** with a small colored tag on the far wall. Colored bin walls would sit at the
  edge of the top camera's view and confuse the HSV baseline.

## M1 — Kinematics, motion, vacuum, cell

- **Two DH tables.** The official UR5e table (d1 = 0.1625, a3 = -0.3922, d4 = 0.1333, d5 = 0.0997,
  d6 = 0.0996) does not match the Menagerie MJCF, whose link offsets are rounded
  (d1 = 0.163, a3 = -0.392, d4 = 0.138 - 0.131 + 0.127 = 0.134, d5 = d6 = 0.1). With fitted base and
  flange transforms, the official table is off by 0.76 mm RMS and 1.08 mm max on random configurations,
  which is above the spec's 1 mm bound. `kinematics/dh.py` therefore keeps both tables: the
  Menagerie-derived one matches MuJoCo to ~1e-12 m with identity base/flange transforms (the DH base
  frame is the MuJoCo world frame) and is the one under test; the official one is tested to stay
  within 2.5 mm. Numbers: `pytest tests/test_kinematics.py -k dh`.
- **IK.** Internal tolerance is 0.01 mm / 0.0057° (tighter than the spec's 0.1 mm / 0.1°, so the
  test has margin). The adaptive damping fades out as the error shrinks, which fixed convergence on
  near-singular targets. The "100 reachable targets" test samples random joint vectors and keeps
  those with manipulability w ≥ 0.01 (exactly singular poses are reachable only in a measure-zero sense).
- **Manipulability thresholds** come from `cobotsort calibrate-manip --seeds 5`
  (`results/manipulability.md`): warn below 0.0405 (5th percentile), block below 0.0157 (half the minimum).
- **MoveJ also respects the TCP speed limit** (0.25 m/s): when the joint-limited quintic would move
  the TCP faster, the duration is stretched. **MoveL duration** is the largest of the TCP-speed,
  TCP-acceleration, tool-rotation and joint-speed bounds; vertical MoveLs are joint-speed limited.
- **Execution** uses the actuated mode (position servos + velocity feed-forward + gravcomp). The
  measured tracking error stays below 1 mm, so the `--kinematic` fallback is available but not needed.
- **Scan pose = home.** Before every scan the arm returns to the home pose, which is outside the top
  camera's view.
- **Verification.** "Left the pick area" is checked with the detector on a fresh scan. The bins sit
  outside the top camera's field of view, so "inside the expected bin" uses a simulated bin presence
  sensor (number of parts resting inside the bin), not the cube's identity.
- **TCP pick height** is the nominal cube top (0.04 m) + 1 mm; the detectors only provide x, y.

## M2 — Safety layer and signal lamp

- **Lift-off contacts.** A part that was just attached still rests on the table, so its contact with
  whatever supported it at attach time is allowed until it has risen 5 mm. Without this, every pick
  triggered a protective stop.
- **Workspace lower bound.** The spec's box (z ≥ 0.005 m) is applied to every TCP sample; during a
  flagged final approach to a known part or slot the lower bound is relaxed to 0.
- **Collision replay** samples the trajectory every 20 ms (≈ 5 mm of TCP travel at 0.25 m/s) and uses
  `mj_kinematics` + `mj_collision` on a scratch `MjData` (~8 µs per sample). A held part is moved with
  the TCP during the replay.
- **Cases 9 and 10 (unknown object id, stale scan id)** are enforced by `agent/guardrails.py`, written
  in M2 so the adversarial set could run in full; M5 builds the agent tools on top of it.
- **Bin full** is enforced by the slot allocator before any motion (`BinFullError`).
- The adversarial set lives in `safety/adversarial.py` so the tests and the benchmark use the same code.

## M3 — Perception

- **Segmentation context without multisampling.** With the default 4× MSAA, edge pixels in the
  segmentation render blend id colors into ids of other geoms, which stretched some boxes across
  several cubes. `CellRenderer.segmentation` now uses its own renderer created with
  `vis.quality.offsamples = 0`.
- **Dataset region** is slightly larger than the pick area (x 0.25–0.65 m, y ±0.27 m) and cubes keep
  ≥ 6 cm between centers so rotated cubes never interpenetrate. Class mix is uniform over the 4 classes.
- **Hard set** (`synth_hard`, 150 images): light tilt up to 60°, diffuse 0.08–1.3, ambient 0.02–0.7 and a
  per-channel color tint of the lights up to ±0.25 (colored light), otherwise the same randomization.
- **Training device.** The author's machine has an RTX 3060, so `cobotsort train` ran on the GPU
  (auto device). Inference in the cell and in the evaluation runs on the **CPU** by default
  (`COBOTSORT_YOLO_DEVICE` overrides it), and the reported ms/frame are CPU numbers.
- **Metrics code.** mAP is computed by `perception/evaluate.py` for both detectors (all-point
  interpolated AP), not by Ultralytics' validator, so HSV and YOLO are scored identically. YOLO's
  full PR curve uses detections down to confidence 0.001; precision/recall are reported at the cell's
  operating point (confidence ≥ 0.6, IoU ≥ 0.5). The HSV "confidence" is a heuristic
  (rectangularity × aspect × size) so it can be ranked for AP.
- **Localization error** uses each image's true (jittered) camera pose, so it measures the detector
  and the pixel-to-world model, not the camera calibration error.
