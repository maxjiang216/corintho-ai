# 15 — The node solver; games end when their outcome is known

2026-09-26. Commits `6b02792` (`--node-p`), `06e2da6` (end on a proven or
solvable position after every move). Data: `data/solver/node-p-sweep.txt`.

## Node solver

A new search leaf with P <= P_node is solved by a `thread_local` Solver
(table 2^20, cap `--node-cap` 20k positions) instead of being sent to the
network (`TrainMC::solveLeaf`). A solved leaf becomes a deduced win, draw
or loss, is propagated like a terminal and backs up its exact value;
capped, it goes to the network as before. Off by default.

Crash found in test mode (`--node-p 17`): the opponent's move could land on
a proven leaf, which has no children or priors, and it became the root
(`moveDown` failed). `receiveOpponentMove` now takes the fresh-root path
for such a node.

Timing (P_game 27, 8,000 games, same setup as entry 14):

| node-p | wall | engine | rows/game | leaves solved | node solve time | game solve time |
|---|---|---|---|---|---|---|
| 0 | 81.7 s | 68.3 s | 30,137 | | | 81 s |
| 15 | 86.2 s | 71.7 s | 30,027 | 1.48M | 2.3 s | 88 s |
| 17 | 92.9 s | 77.9 s | 29,861 | 3.51M | 13.3 s | 109 s |
| 19 | 96.4 s | 83.5 s | 29,471 | 7.03M | 76.3 s | 114 s |

Every leaf within the horizon was solved under the cap. Network rows drop
at most 2%, while engine time rises: node solving costs wall time at every
setting, so it only pays if it makes play stronger (below). Note the
node-p 0 row (81.7 s) against entry 14's P27 row (91.7 s): same setting,
different session; compare within one sweep only.

## A game ends as soon as its outcome is known

The developer asked why a proven leaf could become the root when the game
solver should have ended the game. Found: the horizon was checked only at
the start of an iteration, while known results were played out move after
move inside the move loop (`chooseMoveAndContinue`), past P 27. Then
the developer: "if one side has it solved, can't we just end the game? we
know the outcome".

Now, in solver mode, after every move and at the start of each iteration
(`SelfPlayer::tryEnd`): a proven, non-terminal position ends the game
with its exact result (logged `PROVEN at horizon N`); else P <= P_game is
submitted to the pool (`SOLVED at horizon N`). No samples are written past
that point either way.

Check, 256 logged games at `--solve-p 27 --node-p 17`: 212 SOLVED (at
26-27), 42 PROVEN (at 26-44, mostly 28-35), 2 won on the board (a line
made before the search had proven the position). gen_93 vs itself, 200 games:
100-3-97. Assert build clean; default digests unchanged.

## Open questions

- **Winning path.** The solver gives win/draw/loss only; its table's move
  is *a* winning move, not the shortest (the loser's is arbitrary, not the
  longest). Distance-to-mate scores would break the two null-window
  searches (entry 12). Not needed while games end at the proof; if wanted
  (samples along the line, real play), compute distance along one line
  afterwards ("win within d plies?", d <= P).
- **The gap P 18-27.** No sample is written at P <= 27 (games end there),
  and the developer questions training policy there at all. But the
  network still evaluates search leaves at P 18-27 from roots at P >= 28,
  with no data from that range: its values there may drift across
  generations. Options: value-only samples labelled by the solver (policy
  loss masked), or first measure the network's value error at P 18-27 per
  generation.

## Strength

loop-1 gen 15 (trt16) against itself, 1600 games per match, both sides
ending games at P 27, the node solver on the new side only (`--node-side
new`). Data: `data/solver/node-p-paired-matches.txt`.

The second player wins ~91% of these games, so only the upsets (~8% per
colour) carry signal. Raw totals over 5 seeds (upsets won by the node-solver
side vs by the plain side): node-p 15: 309 vs 314; 17: 308 vs 378; 19: 313
vs 377.

**Paired matches** (the developer's idea): a match is deterministic given
its seed (rerun: identical, game for game), so each treatment is compared
game by game with a baseline match of the same seed (`--node-p 0`); test
mode now writes `game_scores.txt` (the new side's score per game) and
`arch/paired.py` counts the games whose result changed:

| node-p | better | worse | unchanged | sign test z |
|---|---|---|---|---|
| 15 | 540 | 531 | 6929 | +0.28 |
| 17 | 548 | 608 | 6844 | -1.76 |
| 19 | 582 | 636 | 6782 | -1.55 |

(8,000 games each; baseline vs itself: 0 changed.) ~14% of games change:
any change in search sends the rest of the game down a different path, so
pairing removes less noise than hoped, but it isolates the games that
the change affected.

A check that every PROVEN ending matches the exact solver (a temporary
build, one match): no mismatch; the proofs are exact.

**Conclusion: no gain at any P_node, a mild harm trend at 17-19, and a
wall-time cost: the node solver stays off** (P_game 27 alone). Untested
guesses at the harm: exact +-1 values mixed with the network's milder
values in the same averages; proven children leaving selection (their
visits freeze).
