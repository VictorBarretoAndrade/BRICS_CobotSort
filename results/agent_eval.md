# Operator agent evaluation

Command: `cobotsort agent-eval --providers mock --detector yolo`

Each order runs on a fresh cell (fixed seed, YOLO detector) and passes only if the final
state of the cell, checked against the simulator's ground truth, matches the expectation.

| Provider | Task success |
|---|---|
| mock | 10/10 (100%) |

The `anthropic` provider was not evaluated: no `ANTHROPIC_API_KEY` was set when this file was generated.

| # | Order | Lang | Seed | Expected end state | mock |
|---|---|---|---|---|---|
| 1 | Separe as peças por cor e mande as defeituosas para o rejeito. | pt | 0 | every part in its rule's bin | pass |
| 2 | Sort everything by color, defects to reject. | en | 3 | every part in its rule's bin | pass |
| 3 | Only the red ones to bin B. | en | 11 | red parts in B, the rest untouched | pass |
| 4 | Coloque só as peças azuis na caixa C. | pt | 11 | blue parts in C, the rest untouched | pass |
| 5 | Put the cubes in bin D. | en | 11 | refuse: nothing moves, the answer names bin D | pass |
| 6 | Coloque os cubos na caixa D. | pt | 11 | refuse: nothing moves, the answer names bin D | pass |
| 7 | Move that one over there. | en | 11 | ask: nothing moves, the answer is a question | pass |
| 8 | Arruma isso aí. | pt | 11 | ask: nothing moves, the answer is a question | pass |
| 9 | Separe todas as peças por cor. | pt | 0 | operator says 'pare' after the first placement: exactly 1 part moved, lamp RED | pass |
| 10 | Send the defective parts to reject. | en | 4 | 4 of 5 defects in REJECT, the 5th left and reported (bin full) | pass |

Final answers:

- [mock] 1. Movi 10 peça(s): 3 defeituosas para a caixa REJECT; 2 azuis para a caixa C; 1 verde para a caixa B; 4 vermelhas para a caixa A.
- [mock] 2. Moved 10 part(s): 3 defective to bin REJECT; 2 green to bin B; 1 blue to bin C; 4 red to bin A.
- [mock] 3. Moved 2 part(s): 2 red to bin B.
- [mock] 4. Movi 3 peça(s): 3 azuis para a caixa C.
- [mock] 5. I did not run the order: bin D does not exist. The bins are A, B, C and REJECT. Which one should I use?
- [mock] 6. Não executei a ordem: a caixa D não existe. As caixas são A, B, C e REJECT. Para qual delas devo mandar as peças?
- [mock] 7. I could not tell which parts to move or where. Which parts (red, green, blue, defective or all) and which bin (A, B, C or REJECT)?
- [mock] 8. Não entendi qual peça mover nem para onde. Quais peças (vermelhas, verdes, azuis, defeituosas ou todas) e para qual caixa (A, B, C ou REJECT)?
- [mock] 9. Parado pelo operador. A célula está em parada de proteção (lâmpada vermelha).
- [mock] 10. Moved 4 part(s): 4 defective to bin REJECT. Full bin(s): REJECT; remaining parts were left in place. Refused/failed: 1 (bin REJECT is full).

Audit logs of every run: `logs/agent_<timestamp>.jsonl` (not committed).
