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
