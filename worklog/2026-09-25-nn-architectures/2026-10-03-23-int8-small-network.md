# 23 — Smaller and 8-bit small networks: alpha-beta even with MCTS

2026-10-03. `pipeline/cpp/small_net.h`, `pipeline/cpp/ab_match.cpp`,
`pipeline/arch/nnue_distill.py --clip-w2`.

## Why

Entry 22: the network evaluation is 94% of an alpha-beta node's time.
The developer: try (1) smaller networks and (2) 8-bit evaluation. (A GPU
does not help alpha-beta playing one game: each evaluation depends on the
previous results, and one GPU call costs far more than a 2.6 us CPU
evaluation; batching across games would help only self-play, at the cost
of turning the search into a resumable state machine.)

## Smaller networks (float)

Same 200 games as entry 22 (seed 1, ~2 s per move, adjudicated at P 27):

| network | score vs MCTS | pairs (AB-MCTS-split) | depth | nodes/s |
|---|---|---|---|---|
| h1024 | 0.445 | 12-22-66 | 8.1 | 241k |
| h512 | 0.463 | 7-15-78 | 8.5 | 369k |
| h256 | 0.468 | 10-17-73 | 9.0 | 597k |

The faster network searches deeper and is at least as good, though the
three are within noise of each other (+-0.035).

## 8-bit evaluation

`small_net.h`, NNUE-style: layer 1 accumulates int32 (weights x 254;
activations scaled 127 with 3 fractional bits), clipped to [0, 127] as
uint8; layer 2 is uint8 x int8 -> int32 with AVX-VNNI (`vpdpbusd`;
the i7-12700H has it), weights laid out [H/4][32][4] so four activations
update all 32 outputs at once and groups of four zeros are skipped;
layer 3 in float.

Two problems on the way:
- **Speed:** int8 was first slower than float. The accumulators were
  `thread_local` vectors, and the compiler, unable to rule out aliasing
  with the weights, did not vectorize; stack arrays and `__restrict`
  fixed it (the float path did not change: ~2.1 us for h256, its
  per-activation branch dominates).
- **Accuracy:** a few layer-2 weights reach 7-8, so the largest scale that
  fits int8 was ~16 and typical weights (~0.2) got ~3 steps: max error
  0.82, 0.3-0.5% sign flips. Fixed scales on the existing networks (20000
  test positions, one thread):

  | scale | h1024 mean / max error | h256 mean / max error | h256 sign flips |
  |---|---|---|---|
  | 16 | 0.014 / 0.62 | 0.021 / 0.50 | 83 |
  | 32 | 0.008 / 0.28 | 0.012 / 0.30 | 69 |
  | 64 (saturating) | 0.007 / 1.29 | 0.009 / 0.72 | 42 |

  As NNUE does, layer 2's weights are now clipped to +-1.98 in training
  (`--clip-w2 1.98`), so scale 64 (the default) loses nothing. h256
  retrained this way: same accuracy (P 23-27 exact 0.327 / 90.1%, P 28-35
  vs teacher 0.0605, P 36+ 0.0151; unclipped h256 0.060 at P 28-35);
  int8 error mean 0.0062, max 0.29, 24 sign flips in 20000 (0.12%).

Speed, one thread (quiet machine):

| network | float | int8 | speedup |
|---|---|---|---|
| h1024 | 5.01 us | 1.93 us | 2.6x |
| h512 | 3.71 us | 0.85 us | 4.3x |
| h256 | 2.12 us | 0.48 us | 4.4x |

## Match

h256-clip in int8 against MCTS, the same 200 games:

| | |
|---|---|
| AB vs MCTS | **96-2-102, score 0.485** |
| pairs (AB-MCTS-split) | 12-15-73 |
| per move | AB 1.72 s, depth 9.6, **1122k nodes/s**; MCTS 1.70 s, 11,000 searches |

Alpha-beta with an 8-bit 256-wide network, distilled from the AZ network,
is now even with MCTS on the CPU at ~2 s per move (from 0.285 in entry
22's first version). Within noise of 0.5; a larger match would settle
whether it is ahead.

## Incremental first layer

The developer: "what about the incremental thing". Re-profiled first
(int8 h256, 36 games, `-DAB_PROFILE`): evaluation 71% of a node's time
(including the conversion of each position to floats and back to bytes),
move ordering 21%, move generation ~6%, child setup 2%.

The 64 board inputs are the bits of the board word (`Game::key`), so each
search frame keeps layer 1's board part and a child's is its parent's
plus the rows of bits turned on minus the rows of bits turned off; the 6
reserve inputs (counts 0-4, the side to move's first) are added at a leaf
from precomputed rows [slot][count]. No float conversion. Same integers as
the full int8 evaluation: `AB_CHECK_INCREMENTAL=1` compares every leaf
(18 games, no mismatch). `AB_INCREMENTAL=0` turns it off.

Same 200 games:

| | |
|---|---|
| AB vs MCTS | **104-3-93, score 0.527** |
| pairs (AB-MCTS-split) | 15-9-76 |
| per move | AB 1.73 s, depth 10.0, **1457k nodes/s** (+30%); MCTS 1.69 s |

Over the day: 0.285 -> 0.527 against MCTS with the AZ network at equal
CPU time, the last within noise of even. Openings are paired (each played
twice, colours swapped) and identical across all runs (seed 1).

## Cheaper move ordering and lazy accumulators

Profile with the incremental layer (TSC cycles per node): evaluation 615,
move ordering 416 (of which the sort 179, counting replies of line-making
moves 100, the per-move copy + doMove + line test ~137), child copy +
doMove + accumulator update 323, move generation 111.

- **Line test on the board word** (the developer: "a small lookup table to
  figure out if a move makes a line?"). `Game::hasLine` already tests all
  runs with a few shifted ANDs, so a table has little to beat; instead
  `Game::boardHasLine(word)` (refactor `7ea9b04`, digest unchanged) on the
  child's word computed as doMove does, no Game copy, the move made only
  for line-making moves. `AB_CHECK_LINES=1`: no mismatch. Ordering 416 ->
  400 cycles: as expected, the copy and move were cheap.
- **Lazy move picking:** the best remaining move is picked when needed
  instead of sorting all of them; sort 178 -> 124 cycles.
  Both: 200 games 100-4-96 (0.510), 1.51M nodes/s.
- **Lazy accumulators** (Stockfish's approach, the developer's choice over
  in-place update with undo): a node gets its parent's accumulator and
  board word and computes its own only when needed: at a leaf in one pass
  (parent + changed board rows + reserve rows), or before expanding.
  Nodes that return early (table hits, game over, immediate wins) compute
  nothing, and a leaf no longer copies twice. `AB_CHECK_INCREMENTAL=1`:
  every leaf equal to the full int8 evaluation. 1.57M -> 1.63M nodes/s,
  0.64 -> 0.61 us per node; evaluation is now 68% of the timed cycles.

## Faster evaluation without changing the network

The developer chose four of the options (and asked about two others:
branch-free sparsity in layer 2 helps only with very sparse activations,
and here only 13.7% of 4-groups are all zero though 60% of activations
are; below 8 bits there is no hardware support, layer 2 is already at
x86's int8 floor). Measured on 200k test positions first: layer 1's sums
reach at most 11896 in their integer units (int16 holds 32767); the
pre-tanh output spans -4.7..7.2.

Each step checked exact with `AB_CHECK_INCREMENTAL=1` (every leaf equal
to the full 32-bit int8 evaluation, 18 games); profiled (TSC cycles per
node, 36 games, h256-clip):

| step | evaluation cycles/node | nodes/s |
|---|---|---|
| before (lazy accumulators) | 751 | 1.63M |
| drop the tanh: search scores are the raw output / 8 (alpha-beta only compares; / 8 keeps them within +-0.9, below win/loss scores) | | |
| int16 accumulators (wrap-around in intermediate sums cancels) | | |
| reserves in the sums: two views, one per side to move, each with bias, board rows and reserve rows; a placement updates one reserve row in each | 480 (the three together) | 1.88M |
| fused leaf: activations computed from parent + changed rows in 64-wide blocks, no accumulator stored | **438** | **1.96M** |

Evaluation 751 -> 438 cycles per node (-42%), nodes/s +20%.

Same 200 games against MCTS: **106-1-93, score 0.532**, pairs 16-10-74;
AB 1.72 s per move (depth 10.2, 1.96M nodes/s), MCTS 1.72 s. Within
noise of the 0.527 and 0.510 before it; three runs in a row at or above
0.5.

## Next options

- A larger match (or longer time controls) to settle the result.
- Move ordering is now the next cost (21% before the incremental layer).
- A better small network: more data, longer training (the learning curve
  had not flattened, entry 21).
- Alpha-beta self-play with this engine (randomized openings, labels from
  each position's own search score).
