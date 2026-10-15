# Benchmark

Command: `cobotsort benchmark --seeds 20 --detector oracle hsv yolo --provider mock`

Columns follow the competition's judging order: **safety > degree of automation > precision > speed**. Every episode runs the fully automatic cycle (module B): park, scan, detect, nearest-first pick-and-place, re-scan to verify, until no sortable cube remains. Collaborative speed limits (0.5 rad/s per joint, 0.25 m/s TCP). Simulation only.

| Detector | Unplanned contacts | Protective stops | Adversarial blocked | Cubes sorted w/o operator (%) | Episodes end to end (%) | Placement error mean (mm) | Placement error p95 (mm) | Wrong-bin rate (%) | Perception loc. error (mm) | Sim s / cube | Sim s / episode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0 | 0 | 10/10 | 97.2 | 80 | 0.08 | 0.18 | 0.0 | 0.00 | 24.0 | 205.4 |
| hsv | 0 | 0 | 10/10 | 97.2 | 80 | 1.30 | 1.90 | 0.0 | 1.27 | 24.0 | 205.4 |
| yolo | 0 | 0 | 10/10 | 97.2 | 80 | 0.91 | 1.29 | 0.0 | 0.91 | 24.0 | 205.4 |

| Detector | Episodes | Cubes | Flagged for the operator | Wall-clock s / episode (author's CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel) |
|---|---|---|---|---|
| oracle | 20 | 176 | 5 | 8.2 |
| hsv | 20 | 176 | 5 | 8.5 |
| yolo | 20 | 176 | 5 | 8.6 |

Definitions:

- *Unplanned contacts / protective stops*: counted by the runtime monitor during the episodes.
- *Adversarial blocked*: the 10 unsafe requests of `tests/test_safety.py`, run against this detector's cell.
- *Cubes sorted w/o operator*: cubes that ended in their rule's bin with no human input. Cubes left in place because their bin was full (4 slots per bin) count as not sorted.
- *Episodes end to end*: the cycle ended on its own, with no protective stop, and every cube is in its bin.
- *Placement error*: distance from each cube in a bin to the nearest slot center.
- *Perception localization error*: detection after `pixel_to_world` vs. the true cube position, at every scan of the episode.
- *Sim s / cube*: total simulated time divided by the number of sorted cubes (scans and verification included).
- The agent provider does not drive these episodes; agent task success per provider is in `results/agent_eval.md`.

Per-episode rows: `results/benchmark_episodes.csv`.
