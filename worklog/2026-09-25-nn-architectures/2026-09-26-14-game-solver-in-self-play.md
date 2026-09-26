# 14 — The game-level solver in self-play

2026-09-26. Commits `7d367f2` (`--solve-p`), `a12e941` (a game at the
horizon ends at once). Engine: `solver_pool.h`, `SelfPlayer::solveStep /
finalize / adjudicate`, `Trainer::enableSolver / setSolver /
finalizeSolves`.

## Design (the developer's)

A game whose horizon P (2 x reserves + occupied spaces) reaches P_game is
over: its result is exact and only needs computing. The game is submitted
to a `SolverPool` (its own threads, each with a persistent table that is
never cleared) and freed at once; the result is collected by `finalize()`
before a chunk's samples, scores and counts are read. Every earlier
position of the game is labelled from the exact result
(`last_mover_value_` replaces "the last mover won"). A solve that hits the
cap (5M positions) is retried once at 20x; still unknown, the game counts
as a draw (never seen at P 27).

Flags: `--solve-p` (0 off), `--solve-cap`, `--solve-threads` (4),
`--solve-table` (22). Train mode shares one pool per run; test mode adjudicates
both players alike.

## Pause vs end (the developer: "why would we pause, the game is just done")

First version paused the game until its solve returned. Slower: paused
games kept their batch slots, the 4 solver threads saturated, rows per game
went up 36k -> 45-49k and wall +36-55% at P <= 29. Ending the game at once
and collecting the result at chunk end fixed it (`a12e941`).

## Cost by P_game

8,000 games, loop-1 gen 15 (trt16), 14 engine + 6 solver threads:

| P_game | wall | rows/game | turns/game | solver time | waited at chunk ends |
|---|---|---|---|---|---|
| off | 98.5 s | 36.0k | 29.9 | | |
| **27** | **91.7 s (-7%)** | 30.1k | 19.6 | 100 s | 0.5 s |
| 28 | 112.3 s | | | 301 s | 20 s |
| 29 | 153 s | | | 483 s | 72 s |

The developer first leaned to 28-29 ("at equal CPU time we get a big
boost in correctness"); measured, both cost wall time on this CPU, so
**P_game = 27**. Label-only solves at 28-30 (solve, but keep playing)
are skipped for now.

Sign check (test mode, gen_93 vs itself, 200 games): played out 103-8-89;
adjudicated at 16, 18, 20 alike.

Default digests unchanged (solver off).

Later (entry 15): the horizon is checked after every move, and a game also
ends when the search has proven its result.
