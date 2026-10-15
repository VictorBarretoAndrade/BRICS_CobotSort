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
  flange transforms, the official table is off by about 1.1 mm at most on random configurations, which is
  above the spec's 1 mm bound. `kinematics/dh.py` therefore keeps both tables: the Menagerie-derived one
  matches MuJoCo to numerical precision with identity base/flange transforms (the DH base frame is the
  MuJoCo world frame) and is the one under test; the official one is tested to stay within 2.5 mm.
  Numbers: `cobotsort kinematics-report` → `results/kinematics.md`.
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

## M4 — Cell integration and benchmark

- **Semi-automatic mode** lets the operator override the default rules (any detected part into any
  existing bin with a free slot); low-confidence parts get a warning but the operator decides. Detection
  ids are only valid for the scan that printed them. `--commands` runs a scripted session (used in tests).
- **Benchmark episodes run the automatic cycle (module B)**, not the agent. The `--provider` flag is kept
  for the spec's command line; the agent's task success per provider is reported in
  `results/agent_eval.md` (M5).
- **"Cubes sorted without operator input"** counts only cubes that ended in their rule's bin. With 4 slots
  per bin and 6–10 random cubes, some seeds have more parts of one class than slots; those parts are
  flagged and left in place, which lowers this number by design (a full bin refuses the move).
- **"Episodes completed end to end"** = the cycle ended on its own, with no protective stop, and every cube
  is in its correct bin.
- **Placement error** is measured to the nearest slot center of the bin the cube landed in.
- One `SimWorld`, renderer and detector instance are reused across the seeds of a detector
  (`build_cell(world=..., renderer=...)`) to avoid reloading YOLO and leaking GL contexts.

## M5 — Operator agent

- **Manual tool-use loop** (not the SDK's beta tool runner), because the provider layer, the
  rule-based fallback and the per-call audit log need control over every turn.
- **Anthropic request shape.** `claude-sonnet-5-5` (from `COBOTSORT_MODEL`), non-streaming,
  `max_tokens` 16000, 60 s timeout, 1 SDK retry, `tool_choice={"type": "auto",
  "disable_parallel_tool_use": true}`, effort `medium` (`COBOTSORT_EFFORT`). Thinking is left at the
  model's adaptive default (Sonnet 5.5 rejects `{"type": "disabled"}`), and the full assistant content is
  appended back unchanged. Server-side refusal fallbacks (`fallbacks: "default"`, beta
  `server-side-fallback-2026-07-01`) are on by default and can be turned off with
  `COBOTSORT_SERVER_FALLBACK=0`. A `refusal`, `max_tokens`, API error, timeout or missing key switches
  to the rule-based planner for the rest of the order, and the reason is logged.
- **Not evaluated with a live key.** No `ANTHROPIC_API_KEY` was available while building, so
  `results/agent_eval.md` reports the `mock` provider only; the Anthropic loop is covered by tests with a
  fake client (tool_result ids, `is_error`, fallback on API error and on refusal).
- **Verification scan in the pick result.** `pick_and_place` already re-scans to verify the placement;
  that scan is taken after the placement, so it is the latest scan and is returned to the agent. This
  keeps a 10-part order at 11 tool calls instead of 20 (the budget is 20 calls and 12 picks per order).
  The guardrail still refuses any older scan id.
- **Stop.** A stop word in an operator interrupt is executed by code (`stop_cell`) before any further
  model turn ("the E-stop always wins"); a stop in the order text itself goes through the provider, and
  the guardrails refuse every motion while a fault or E-stop is active.
- **Mock planner** picks the first matching object of the latest scan (the automatic cycle in M4 is
  nearest-first). The bin enum in the tool schema is a hint for the model; the guardrail is the check.

## M6 — Demo and docs

- **Video speed-ups.** The spec allows transfer moves at 2×. In collaborative mode a 10-part order takes
  about 225 s of simulated time, so a 40–60 s video needs more: transfers (MoveJ, parking, scanning) are shown
  at 6× and approaches (MoveL, vacuum, waits) at 3×. The current factor is always on screen, and the final
  card shows the simulated time.
- **Demo order and seed.** The spec's command is used as is (seed 7, YOLO, the Portuguese order). Seed 7 has
  no defective part, so nothing goes to REJECT in the video; the agent's summary says so implicitly by
  listing what moved.
- **GIF**: 10 s from the start of the video, 640 px wide, 10 fps, 128 colors.
- **IK second round.** `cobotsort kinematics-report` exposed a reachable target (arm hanging below the base)
  that the default seeds missed (99/100). `IKSolver.solve_all` now retries with 4× more random seeds when the
  first round finds nothing (100/100). The benchmark, agent evaluation and demo were re-run after this change.
- **Console encoding.** The CLI reconfigures stdout/stderr to UTF-8 so Portuguese output prints on Windows.
- **License.** Ultralytics is AGPL-3.0 and a fine-tuned checkpoint is committed, so `pyproject.toml`
  declares AGPL-3.0-or-later. Adding the full license text (a `LICENSE` file) is left to the owner.
