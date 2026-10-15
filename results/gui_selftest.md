# GUI self-test

Command: `cobotsort gui --selftest`

| Step | Result | Detail |
|---|---|---|
| engage | pass | lamp YELLOW |
| manual: unsafe jog J1 +10 deg is refused | pass | SafetyViolation: TCP leaves the workspace at (-0.200, 0.469, 0.458) |
| manual: jog J1 -10 deg | pass | J1 = -100.0 deg |
| manual: jog Z -50 mm | pass | TCP = (-0.046, 0.508, 0.407) m |
| manual: home | pass | at home |
| semi-auto: scan | pass | 10 parts detected |
| semi-auto: execute selection | pass | defect -> REJECT1 (verified) |
| auto: start, then stop | pass | 6 placed, 0 flagged, stopped by the operator; parts in bins 1 -> 7 |
| agent: Portuguese order (mock) | pass | Movi 3 peça(s): 1 defeituosa para a caixa REJECT; 2 vermelhas para a caixa A. |
| E-stop during a motion | pass | job failed as expected: ProtectiveStop: E-stop |
| reset errors -> engage -> home | pass | at home |
