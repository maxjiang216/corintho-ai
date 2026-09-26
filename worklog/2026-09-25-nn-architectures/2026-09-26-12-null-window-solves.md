# 12 — Two yes/no searches instead of one three-way search

2026-09-26. Commit `767b00a`. Data: `data/solver/bench-12-yesno-p30.tsv`.

Results are only win / draw / loss, so solve() now asks "is it a win?"
(window 0..1) and, only if not, "is it at least a draw?" (window -1..0),
the second search reusing the table and history of the first. A null
window lets every position prove just one side of one threshold (a draw
and a loss are the same answer to "is it a win?"). The full-window search
remains behind `SOLVER_FULL_WINDOW`.

| set (6 threads) | full window | yes/no | nodes |
|---|---|---|---|
| P <= 24 | 2.20M nodes, 0.44 s summed | 2.20M, 0.49 s | 1.00x |
| P <= 27 | 33.1M, 5.84 s | 30.0M, 5.50 s | 0.91x |
| P <= 30 | 398M, 77.6 s | 378M, 75.6 s | 0.93x (6154 solved both; 4 swapped at the 5M cap) |

Gate: 0 disagreements. A modest gain on deeper solves (the game-level
solver's range), none on easy ones: draws are rare (~3% of games), so the
three-way search seldom spent effort separating draws from losses.
