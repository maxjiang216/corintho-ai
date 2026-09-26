# 08 — The endgame solver

2026-09-26. Commits `85a92d1` (module), `3c5e769`, `937ade1`, `a188dff`,
`3c65943`, `30a7ce1`, `ca72b11`, `5c47208`, `e131266`; `e94b42f`
(bench/solve_cost). Data: `data/solver/`.

## Why, and the plan (the developer's design)

Two solvers: a game-level one that runs once per game when the game's
horizon P = 2 x reserves + occupied spaces (an upper bound on plies left;
every move lowers it) falls to P_game, ends the game and labels every
earlier position with the exact result; and a node-level one inside the
search tree for P <= P_node, much cheaper per call. Positions past the
thresholds need no training ("no decision to make"); the proven line can
be emitted as extra samples with exact labels.

## Measurements before building

- MCTS already proves results 3-11 plies before the end (typically 5),
  from gens 79-94's logs.
- Game-result value labels are wrong in sign on 5-18% of solved positions
  at P 22-29 (0% at P <= 18): the label noise a game solver removes.
- Break-even for the game solver: self-play spends ~7 ms of CPU per move,
  so solving at P with k plies left is free up to ~7k ms.
- Search depth (500 games, leaves vs their search root, in P; full table in
  `data/search-depth-by-root-P.txt`): median 7, p90 12. From root P 27, 47% of leaves are at P <= 19; from 30, 20%; 32,
  9%. The node solver would see a lot of use in the few moves before
  P_game, at a cost to be measured with a warm table.

## The solver (solver.h)

Negamax alpha-beta over exact results {-1, 0, 1}, no depth limit, a
transposition table with bounds and best move (epoch clears), node cap
(kUnknown). Tests: matches memoized plain minimax on 300 endgames (P <=
13), table reuse keeps results, cap gives kUnknown. Benchmark:
`bench/solver_bench.cpp` on `bench/results/solver-positions.bin` (9171
self-play positions, up to 300 per P); gate `bench/solver_gate.py` (every
common solve must keep its result).

## Optimizations (each gated: 0 changed results)

At each node now: table probe; the table's move alone first; the other
children classified by whether they make a line (branch-free hasLine);
line-making children get their legal moves (zero replies = immediate win)
and go first, fewest replies first; quiet children follow, stack moves then
base, column, capital placements, their legal moves (no line search)
generated only when searched.

| step | change | effect |
|---|---|---|
| 0 | probe's algorithm | P <= 24: 385.8M nodes, 64.1 s one thread |
| 1 | immediate wins, fewest replies first, legal moves passed down | nodes 0.023x |
| 2 | uninitialized child storage, insertion sort | instructions -24%, time -26% |
| 3 | table move before generating others | neutral cold; for warm tables |
| 4 | line-making moves first (the developer), quiet children lazy | nodes 0.35x, time 2.6x less |
| 5 | branch-free hasLine | instructions -20%, mispredicts -27% |
| 6 | pext, three types in one word | time -14% |
| 7 | no line search for known quiet children | instructions -7% |
| 8 | quiet order explicit: stack moves, then base, column, capital | identical (the reverse, the developer's intuition, was 7.5x more nodes) |

Net, one thread pinned: P <= 22 5.66 -> 0.21 s (27x); **P <= 24 64.1 ->
0.92 s (70x)**, 4350 -> 4371 of 4371 solved.

Frontier (5M-node cap, 20 threads, so per-solve times are pessimistic;
`data/solver/bench-8-final-p32.txt`), after step 1 -> final: P 26 99.7% /
4.8 ms -> 100% / 1.7 ms median; P 28 94% / 42 ms -> 99% / 9 ms; P 30 68% /
141 ms -> 93% / 62 ms; P 32 40% -> 64% solved.

Profiles (callgrind summaries, instructions, mispredicted branches, L1
misses) after steps 1, 4, 6 and 8 are in `data/solver/profile-*.txt`.

Later (after entry 09): **history heuristic** (`4408971`): quiet cutoffs
credited with P^2 per side and move, quiet moves ordered by it within each
rank group: nodes 0.86x at P <= 24, 0.80x at P <= 27 (summed time 9.62 ->
7.99 s). Letting history override the rank groups instead: 1.36x / 1.47x
more nodes than that; the fixed order is the stronger signal.

Tried and dropped: an AVX2 version classifying four children at once
(child boards as word arithmetic, top pieces in nibble space): +3%
instructions, no time gain (lane setup cost ~ savings; classification was
~15% of time). Table size barely matters at these depths (2^14-2^22).

Remaining profile (P <= 24): legal moves of line-making children
(findLines, lineBreakers) ~35%, the search's table probes ~22% (98% of L1
misses), basic move generation ~12%, hasLine ~8%, doMove ~7%.

## Thresholds, as of now

P_game ~ 26-27 (median ~5 ms, well under the ~60-70 ms break-even); P_node
~ 19 (median <= 13 us cold; a warm table per game to be measured).
