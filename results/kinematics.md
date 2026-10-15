# Kinematics report

Command: `cobotsort kinematics-report` (seed 0).

## Forward kinematics: DH vs MuJoCo

Constant base and flange transforms are fitted once on 30 random configurations; the
errors below are on 50 other random configurations (flange pose).

| DH table | Position error max (mm) | Position error RMS (mm) | Rotation error max (deg) |
|---|---|---|---|
| UR5e (MuJoCo Menagerie geometry) | 0.0000 | 0.0000 | 0.0000 |
| UR5e (Universal Robots) | 1.1233 | 0.8028 | 0.0096 |

## Inverse kinematics (damped least squares, multi-seed)

100 reachable targets: random joint vectors with manipulability ≥ 0.01, mapped to a
TCP pose by forward kinematics, then solved from the home configuration.

| Solved | Position error max (mm) | Rotation error max (deg) | Distinct solutions per target (mean, min-max) | Time per target, all seeds (ms) |
|---|---|---|---|---|
| 100/100 | 0.0098 | 0.0011 | 7.5 (2-10) | 24.7 |

An unreachable target (TCP 1.5 m from the base) returns no solution: 0 solutions.
