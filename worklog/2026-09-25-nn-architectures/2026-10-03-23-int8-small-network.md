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

## Next options

- A larger match (or longer time controls) to settle the result.
- Incremental first layer (now a larger share of the time).
- A better small network: more data, longer training (the learning curve
  had not flattened, entry 21).
- Alpha-beta self-play with this engine (randomized openings, labels from
  each position's own search score).
