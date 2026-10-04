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
  then a fixed-depth search (--depth, default 5; 6 at first) for both sides. For the
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

## Depth 5, progress by time

The developer: depth 6 is too slow on the real run; progress lines by
time, not by games. Same 3,000 positions, 4 threads: depth 4 3 s, depth 5
9 s, depth 6 26 s. Default now depth 5 (~3x faster than 6, ~350
positions/s on 4 threads). `ab_selfplay --report-seconds` (30) prints
positions, rate, time left and the first player's results so far;
`nnue_distill.py train --report-seconds` prints step and loss; `ab_loop.sh`
sends both to progress.log (REPORT, default 30 s).

## To run

```
make build/ab_selfplay build/ab_match
arch/ab_loop.sh                       # runs/ab-1, 30 generations
POSITIONS=500000 arch/ab_loop.sh      # later generations larger
```

## Run ab-1 (developer's machine, 18 threads), first generations

Defaults: 200k positions per generation, depth 5, window 4, 20k steps,
lambda 0.3 + 0.05 per generation. ~1,000-1,240 positions/s (faster as the
network improves), ~3.5-4 minutes per generation with training (~45 s).

| gen | first player W-D-L (self-play) | first player share |
|---|---|---|
| 1 | 6017-53-6715 (partial) | 47% |
| 2 | 5008-95-6194 | 45% |
| 3 | 4150-65-5881 | 41% |
| 4 | 3741-74-5607 | 40% |
| 5 | 3790-80-5693 | 40% |
| 6 | 4080-81-6127 | 40% |
| 7 | 4133-105-6180 | 40% |

The second player's advantage returns as the network learns, then holds
at ~40% for the first player (the AZ runs reached ~9%).

**Generation 5 against MCTS** (entry 23's match, the same 200 paired games):
107-6-87, **score 0.550**, against 0.490 (plain) and 0.495 (monotone) for
the networks distilled from AZ (entry 24). From a blank network in 21
minutes; about 1.4 standard errors above even, and above the distilled
networks by ~0.055 on identical openings. To be confirmed by generations
10 and 15.

Generations 8-11: the first player at 39-40% (gen 10: 4275-116-6615);
self-play 1,200-1,320 positions/s; lambda reaches its cap (0.8) at
generation 11.

**Generation 10 against MCTS: 115-3-82, score 0.583** (gen 5: 0.550;
distilled: 0.490-0.495). About 2.4 standard errors above even (~+58 Elo),
47 minutes from a blank network, and still rising.

Generations 12-17 (lambda 0.8): the first player at 39-40%, loss flat at
0.16-0.17. **Generation 15 against MCTS: 108-6-86, score 0.555.** With 5:
0.550 and 10: 0.583, within noise of each other (a difference of two
200-game matches is +-0.05): likely a plateau at ~0.55-0.58 with 200k
positions per generation. Next: more positions per generation (500k), and
finer measurement (longer matches, or generation against generation).

**Generation 20 against MCTS: 117-8-75, score 0.605** (~+74 Elo, ~3
standard errors above even). With 5-15 (0.550, 0.583, 0.555): still rising,
slowly (~+0.003 per generation); the "plateau" at 15 was noise. Run
continues to generation 30 at 200k positions.

**Generation 25 against MCTS: 120-1-79, score 0.603**, the same as 20
(0.605): at 200k positions per generation the run levels off around 0.60
(5-25: 0.550, 0.583, 0.555, 0.605, 0.603).

## Growing generations (overnight)

The developer: double the training and test games, and scale again when
progress stalls. `ab_loop.sh` ADAPT=1: after each match against MCTS
(MATCH_GAMES), a score that does not beat the best so far by STALL_MARGIN
(0.01) counts as a stall; after STALL_EVALS (2) in a row, POSITIONS and
STEPS double (up to MAX_POSITIONS, 3.2M). The schedule is kept in
`$RUN/schedule` across restarts. KEEP_DATA=0 deletes positions that have
left the training window. Checked at toy size with a stand-in match
(constant score): doubles after 2 stalls, continues after a restart.

`arch/ab_run.sh start|stop|status|log`: one command (the developer: too many
steps to restart). `start` builds, stops any running loop, and starts
ab_loop.sh in its own process group (setsid), so `stop` ends it and all its
children; defaults are the overnight settings (ADAPT=1, 400k positions,
40k steps, 400-game matches, 200 generations, KEEP_DATA=0, a progress line
every 60 s). `status` prints the schedule, every match and the latest
line. Checked here at toy size (stand-in match): start, status, stop with
no process left, restart continuing the schedule.
