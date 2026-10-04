# 25 — Alpha-beta self-play from a blank network

2026-10-04. `pipeline/cpp/ab_selfplay.cpp`, `pipeline/cpp/alphabeta.h`,
`pipeline/arch/nnue_distill.py` (`init`, `mix`, `train --init --select`),
`pipeline/arch/ab_loop.sh`.

## Why

Alpha-beta (AB) with the 8-bit small network is about even with MCTS and
the AZ network (entries 23-24). The developer: try AB self-play,
**from scratch**, to start from a clean slate without the AZ network's
biases; the exact solver anchors the endgame, and a deep search over a
random evaluation already prefers positions with more options (Beal and
Smith's random-evaluation effect).

Settled in discussion (entry 24 and before):

- Network: h256, int8, monotone in the reserves (64/16), random orientation
  per sample in training (no tying).
- Labels: exact where solved; elsewhere lambda x search value + (1 -
  lambda) x outcome, lambda starting low and rising. Whether the second
  player's advantage of the AZ runs returns decides how much the outcome
  is worth.
- No parity input for now.
- Openings deduplicated up to symmetry; move choice sampled from the
  search scores for the first moves, a best-effort source of variety.
- Small generations first (~200k positions, ~20 minutes each), larger once
  progress slows.

## The pieces

- `alphabeta.h`: entry 22-23's search, moved out of `ab_match.cpp` (no
  change to it), plus a depth limit (`choose(game, seconds, max_depth)`)
  and `rootScores`: after a fixed-depth search, every other root move is
  searched with a window just below the best, so moves within a margin get
  exact scores and the rest fail low.
- `ab_selfplay`: games on --threads threads. --opening random moves
  (default 4), redrawn if the position up to symmetry was already used;
  then a fixed-depth search (--depth, default 6) for both sides. For the
  first --temp-plies moves after the opening (10) the move is sampled with
  probabilities exp((s - best) / T) over the moves within 4T of the best,
  T falling linearly from --temp (0.05 in search-score units, 0.4 before
  the tanh) to 0. Every position at P <= 27 is solved (5M-node cap). The
  outcome label is the exact result of the game's first solved position,
  so a later mistake does not change it; games that never reach P 27 use
  the result as played. Rows for P >= 20: inputs, search value, outcome,
  exact value (NaN if none), P, game, ply.
- `nnue_distill.py init` (a random network, projected to monotone), `mix`
  (targets from generations at a given lambda), `train --init PREV.pt
  --select last` (warm start; the last step, not a checkpoint chosen on the
  AZ teacher).
- `ab_loop.sh`: generation g plays with network g-1, mixes the last WINDOW
  (4) generations, trains network g from g-1 (20k steps), and every EVAL (5)
  generations plays entry 23's match against MCTS. Resumable step by step;
  POSITIONS and the lambda schedule can change between generations.

## First measurements (cloud container, 4 threads, random network)

3,000 positions, depth 6:

| setting | wall | positions/s |
|---|---|---|
| default (temperature 10 plies, solver at P <= 27) | 26 s | 118 |
| no solver | 26 s | 118 |
| no temperature | 10 s | 306 |

The solver costs nothing measurable; sampling costs 2.6x (scoring the
near-best moves in the first 10 plies). At depth 4: 3 s. Nodes/s per thread
3.3-3.4M. Expected on the developer's machine (18 threads): ~450
positions/s at depth 6, so ~7-8 minutes per 200k positions.

Random network, depth 6: 24 plies per game, 87% of games reach a solved
position, 28% of positions (P >= 20) have exact values. The first player
scores 84-4-79 with sampling and **64-1-104 without**: the second
player's advantage appears even with a random evaluation.

Search value against outcome, correlation by P: 0.61 (P 20-27), 0.31
(28-35), 0.17 (36+). Outcome equals the exact value on 84% of solved
positions (play after the first solved position is imperfect).

A two-generation loop at toy size (1,500 positions, depth 4, 200 steps)
ran end to end. `ab_match` compiles unchanged against ONNX Runtime's
headers (also with `-DAB_PROFILE`).

## To run

```
make build/ab_selfplay build/ab_match
arch/ab_loop.sh                       # runs/ab-1, 30 generations
POSITIONS=500000 arch/ab_loop.sh      # later generations larger
```
