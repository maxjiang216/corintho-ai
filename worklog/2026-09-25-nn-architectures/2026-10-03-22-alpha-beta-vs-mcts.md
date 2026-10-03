# 22 — Alpha-beta with a small network against MCTS: a preliminary match

2026-10-03. Tool `pipeline/cpp/ab_match.cpp` (`make build/ab_match`).

## Question

The developer: is alpha-beta (AB) with the small network of entry 21
stronger than the best MCTS network, at the same thinking time, ~2-3 s
per move; a few hundred games with random openings; "doesn't have to be
too robust [...] just preliminary". (The web app, `web/`, runs MCTS with
an old small model in the browser; the comparison here uses our best AZ
network instead.)

## Setup

- MCTS: the engine's `TrainMC` in testing mode (c_puct 3, noise 0.25, as
  in test matches), tree kept between moves, `runs/solve-0/gen_5` through
  ONNX Runtime on the CPU (one intra-op thread), 16 rows per call.
- AB: iterative-deepening negamax with a transposition table (2^20),
  principal variation search, history ordering, the table's move first;
  terminal positions exact (quick wins preferred); the small network
  (h1024, all P, entry 21) at depth 0, in float, input-major layers
  (sparse inputs and zero activations skipped). A new iteration starts
  only if half the time is left.
- Both sides on one CPU thread per game, 18 games at once. Openings: 4
  random legal moves, each opening played twice with colours swapped.
  Games adjudicated by exact solution at P <= 27 for both sides.
- Calibration: 1600 MCTS searches take ~0.29 s per move under this load
  (~5.5k searches/s), so 2 s ~ 11,000 searches.

## Engineering notes

- First AB build: 22k nodes/s. The second layer was a dot-product
  reduction, which clang does not vectorize without fast-math; stored
  input-major ([h][32]) it updates 32 independent sums per activation:
  96k nodes/s (147k in the final run).
- Ordering every node of depth >= 2 by the network's value of each child
  cost more than it saved (depth 5.0 vs 5.2, 59k nodes/s); left off
  (`AB_ORDER_DEPTH`, default off).

## Results

Calibration, AB 2 s against MCTS 1600 searches (0.3 s), seed 99:

| AB build | AB vs MCTS | AB depth | nodes/s |
|---|---|---|---|
| first | 6-0-12 (18 games) | 4.3 | 22k |
| vectorized layer 2 | 5-2-11 (18) | 5.2 | 96k |
| + child-value ordering, PVS | 7-0-29 (36) | 5.0 | 59k |

**Equal time**, 200 games (100 opening pairs), seed 1, AB 2 s vs MCTS
11,000 searches:

| | |
|---|---|
| AB vs MCTS | **54-6-140, score 0.285** (decisive 0.278; ~-160 Elo) |
| opening pairs | AB won 2, MCTS won 48, split 50 |
| time per move | AB 1.60 s (mean depth 5.5, 147k nodes/s), MCTS 2.29 s |

## Reading

At equal time, MCTS with the AZ network is far stronger than this AB.
AB reaches only depth ~5.5 in 2 s with ~30 legal moves per position:
without a policy it cannot be selective, while MCTS spends its searches
along the lines its policy prefers. The small network matched the AZ
value closely (entry 21), so the gap is the search, not the evaluation.

What could close it, roughly in order of expected size: a policy for move
ordering and late-move reductions (selectivity); int8 and incremental
evaluation (~3-5x nodes, about one ply); quiescence over line threats.
Each is real work, against a ~160 Elo deficit. The developer asked only
for a preliminary read; no further AB work is planned without a decision.

## Search improvements (the developer's suggestions)

The developer: the solver's move ordering should be a good ordering for a
normal search too, and line-making moves should be extended like checks
in chess (also: Stockfish's search for ideas, which is GPLv3, so ideas
only, no code; and incremental evaluation, since a move changes only a
few inputs). Iterative deepening orders only through the table's move:
the first move at nodes searched in the previous iteration; every other
move, and every new node at the frontier, needs a static order.

Same 200 games (seed 1, equal time):

| AB | score | pairs (AB-MCTS-split) | depth | nodes/s |
|---|---|---|---|---|
| table move + history | 0.285 | 2-48-50 | 5.5 | 147k |
| + the solver's ordering (line-making moves by fewest replies, an immediate win returned at once; quiet moves by type, then history) | 0.328 | 1-37-62 | 6.9 | 195k |
| + search on while a line stands at depth 0 (`AB_LINE_EXTEND`) | 0.345 | 3-35-62 | 6.7 | 241k |

| + late-move reductions (`AB_LMR`): from the 4th quiet move at depth >= 3, 1 ply less (2 from the 9th), re-searched at full depth if it beats alpha; none while a line stands | 0.372 | 2-30-68 | 7.9 | 206k |
| + the whole time: no early stop; a move that beats the previous best in an unfinished iteration is played | **0.445** | **12-22-66** | 8.1 | 241k |

Seconds per move in the last two runs: AB 1.42 vs MCTS 2.28 (MCTS's time
varies with load), then AB 1.78 vs MCTS 1.79: only the last run is at
equal time.

(200 games: about +-0.035 per score.) The solver's ordering gives 1.4
plies; the line extension is within noise; late-move reductions 1.2
plies more. At equal time AB now scores 0.445 (~-40 Elo, about 1.5
standard errors from even). Next levers: incremental evaluation, int8,
a larger match to settle the gap.
