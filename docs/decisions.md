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
