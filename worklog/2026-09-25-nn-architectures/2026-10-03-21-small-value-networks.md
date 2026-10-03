# 21 — Small value networks for an alpha-beta engine

2026-10-03. Script `pipeline/arch/nnue_distill.py`; results
`pipeline/runs/nnue/results.jsonl`.

## Why

The developer asked whether self-play with alpha-beta (AB) instead of AZ
is worth it, since AZ seems to plateau. Discussion:
- for AB, the evaluation is the only learned part, but it must be good
  per unit of time (depth beats a slightly better evaluation), and move
  ordering and search features still matter;
- the developer: "why not just train an NNUE on whatever current self-play
  data, the bootstrap doesn't have to be too good"; AB self-play then
  needs randomized openings (a few plies only, to stay on-policy) and
  labels from each position's own search score, not the game outcome;
- "who cares about the endgame [...] even the no-model AB search bashes
  through the endgame fast enough": roots at P <= ~29 can be solved
  outright (tens of ms), and leaves below ~P 20 solved on the spot, so
  the evaluation matters at about P 20-40.

Step 1 (this entry): can a small, fast network match the AZ network's
values where AB would use them?

## Setup

- Training positions: `runs/relabel-27` gens 1-5 (3,728,261 positions;
  entry 20), whose labels are exact at P <= 27.
- Targets: the exact value at P <= 27 (35% of positions), else the value
  of the teacher, `runs/solve-0/gen_5` (AZ, 512x4 residual).
- Test set: `runs/solve-0b/gen_5` data (747k positions; another run, other
  seeds), with the teacher's values, the game outcomes, and exact values
  for 6000 positions at P 10-27 (`bench/solver_bench`).
- Small networks: 70 -> H -> 32 -> 1, clipped ReLU, tanh; AdamW lr 3e-3,
  one-cycle, batch 4096, a random board symmetry per sample, MSE. The
  checkpoint kept is the best on the test set's P >= 28 teacher error
  (checked every 2000-8000 steps; a mild selection on the test set).

## Accuracy

| network | positions | P 23-27 exact (MSE / sign) | P 28-35 vs teacher | P 36+ vs teacher | P 10-22 exact |
|---|---|---|---|---|---|
| teacher (AZ solve-0 g5) | | 0.324 / 89.1% | 0 | 0 | 0.154 / 95.6% |
| another AZ (relabel-27 g5) | | 0.242 / 92.2% | 0.055 | 0.022 | 0.170 / 94.9% |
| h256, 20k steps | 3.7M | 0.360 / 88.4% | 0.070 | 0.018 | 0.298 / 90.8% |
| h256, 80k | 3.7M | 0.324 / 89.7% | 0.060 | 0.015 | 0.275 / 91.8% |
| h512, 80k | 3.7M | 0.323 / 89.7% | 0.056 | 0.014 | 0.266 / 91.9% |
| **h1024, 80k** | 3.7M | **0.291 / 91.2%** | **0.051** | **0.011** | 0.239 / 92.3% |
| h512, 80k | 100k | 0.423 / 86.0% | 0.089 | 0.024 | 0.327 / 90.2% |
| h512, 80k | 300k | 0.398 / 86.9% | 0.076 | 0.019 | 0.308 / 90.9% |
| h512, 80k | 1M | 0.337 / 89.3% | 0.061 | 0.015 | 0.295 / 91.0% |

- Where AB would use it (P ~20-40), h1024 is as close to the teacher as a
  second AZ network of the same strength is, or closer (0.051 vs 0.055;
  0.011 vs 0.022), and between the two AZ networks on exact values at
  P 23-27.
- Weakest in the late endgame (P 10-22), which AB would solve instead.
- The learning curve still falls at 3.7M positions, and the full-data
  runs were still improving at their last step: more data and longer
  training should help.

## Speed

One thread, float, full evaluation (no int8 quantization, no incremental
updates; sparse input rows skipped), random weights, inputs with ~10 board
bits set:

| hidden | per position |
|---|---|
| 256 | 0.72 us |
| 512 | 1.77 us |
| 1024 | 2.56 us |

The AZ network: ~11.5 us per call on CPU (2026-09-24 probe).

## Training only where AB evaluates (P >= 20)

The developer's point: positions below ~P 20 are solved, so the network
need not learn them. `--min-p 20` (3,255,440 positions):

| network | P 23-27 exact | P 28-35 vs teacher | P 36+ vs teacher |
|---|---|---|---|
| h512, all P | 0.323 / 89.7% | 0.056 | 0.014 |
| h512, P >= 20 | 0.317 / 90.1% | 0.056 | 0.013 |
| h1024, all P | 0.291 / 91.2% | 0.051 | 0.011 |
| h1024, P >= 20 | 0.301 / 90.5% | 0.050 | 0.011 |

No real difference: the late endgame was not taking capacity from the
bands that matter. P >= 20 is fine to use (13% fewer positions).

## Next

1. An AB engine with a small network (the solver's search with a depth or
   node limit, the network at the leaves, exact solves at P <= ~27),
   against MCTS with the AZ network at equal time per move, both sides
   adjudicated at P 27.
2. Speed: int8 quantization and incremental first-layer updates.
3. Only if 1 is competitive: AB self-play generations (randomized
   openings, search-score labels).
