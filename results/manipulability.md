# Manipulability calibration

Command: `cobotsort calibrate-manip --seeds 5 --stride 25`

Yoshikawa index w = sqrt(det(J J^T)) sampled every 25 control steps
along every trajectory of 5 automatic oracle episodes (seeds 0..4), 22386 samples.

| statistic | value |
|---|---|
| min | 0.0314 |
| 5th percentile | 0.0405 |
| median | 0.0794 |
| max | 0.1174 |

Thresholds: **warn below 0.0405** (5th percentile), **block below 0.0157** (half the minimum).
These values are copied into `SafetyConfig.manip_warn` / `SafetyConfig.manip_block`.
