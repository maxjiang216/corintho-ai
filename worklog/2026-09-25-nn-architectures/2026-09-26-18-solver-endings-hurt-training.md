# 18 — Ending self-play games by solver hurts training

2026-09-26. Runs `pipeline/runs/solve-27`, `pipeline/runs/solve-0`
(entry 16), continued to gen 5 by `arch/paired_loop.sh` (`bbc5a5e`: one
generation of each in turn). Data: `data/solver/solve-27-vs-solve-0-gen5.txt`.

## Setup

Both runs start from `runs/night-2/it1/net.pt` (the best anchored network,
70% vs full-2 gen 1). Entry 16 was written in a session that believed
the start was a random network; both `config.txt` files say otherwise.
Default loop config (25k games per generation, window 4, 2 epochs, lr
1e-2). Only `SOLVE_P` differs: 27 vs 0. Matches (1600 games) are played
out to the end with the solver off, as always in `loop.sh`.

## Result at gen 5

| gen 5 vs | solve-27 | solve-0 |
|---|---|---|
| gen0 (its start, night-2 it1) | **0.241** | **0.571** |
| n1it0 | 0.355 | 0.645 |
| gen1 (full-2) | 0.392 | 0.697 |

(decisive win rate). Without the solver, the run improves on its start.
With it, the network lost roughly 200 Elo in 5 generations. The
validation value MSE shows it too: solve-27 0.39 -> 0.61 over gens 1-5,
solve-0 flat at ~0.38.

## Endgame only, or everything?

The solve-27 network is never trained on a position at P <= 27 (games end
there), yet the matches make it play those endgames. Replayed gen 5 vs
gen 0 (same seed; a test match is deterministic, and the played-out rerun
reproduced 381-21-1198 exactly) with both sides' games adjudicated at P
27 (`test --solve-p 27`):

| | decisive win rate |
|---|---|
| played out | 0.241 |
| adjudicated at P 27 | **0.399** |

So most of the loss comes from endgame play, which drifts without data. But
even judged only up to P 27, gen 5 is clearly worse than its start. The
middlegame got worse too, plausibly because the search from roots at P
28-35 relies on the network's values at leaves with P <= 27 (entry 15's
open question), and those values now come from a network with no data
there. Its errors then flow into the policy and value targets at P >= 28.

## Fair matches: the solver for both sides

The developer: matches should give the solver to both sides, since the
user-facing engine would use one. Every network against its run's start,
1600 games, seed 7005, `test --solve-p 27` (both sides adjudicated alike):

| gen vs its start | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| solve-27 | **0.212** | 0.392 | 0.388 | 0.373 | 0.399 |
| solve-0 | 0.518 | 0.555 | 0.560 | 0.592 | 0.566 |

Head to head, solve-27 gen 5 vs solve-0 gen 5: 596-18-986, **0.377**
(~10 standard errors). Most of the damage is in the first generation.

## Not a label bug

3000 P 28 positions from each run's gen 1 data, solved exactly
(`bench/solver_bench`, cap 50M, all solved). A label can never beat the
exact value (a win cannot come from a lost position):

| | solve-27 | solve-0 |
|---|---|---|
| label > exact | **0** | 186 (the opponent's later mistakes) |
| label == exact | **93.2%** | 85.3% |

The solver's labels are correct and more accurate than game outcomes.

## The mechanism: endgame values collapse

Value MSE by horizon P (/ sign correct on decisive positions), 6000
positions at P 18-27 from solve-0 gen 3 data with exact solver labels,
6000 each at P 28-35 and 36+ with outcome labels:

| network | P 18-22 exact | P 23-27 exact | P 28-35 | P 36+ |
|---|---|---|---|---|
| solve-27 g0 (start) | 0.256 / 93% | 0.378 / 88% | 0.564 | 0.539 |
| solve-27 g1 | **0.680 / 66%** | 0.497 / 84% | 0.576 | 0.539 |
| solve-27 g2 | 0.370 / 88% | 0.382 / 88% | 0.505 | 0.517 |
| solve-27 g3 | 0.345 / 90% | 0.372 / 88% | 0.495 | 0.517 |
| solve-27 g4 | 0.339 / 90% | 0.366 / 88% | 0.492 | 0.518 |
| solve-27 g5 | 0.367 / 89% | 0.392 / 87% | 0.507 | 0.536 |
| solve-0 g1 | 0.219 / 93% | 0.304 / 90% | 0.501 | 0.518 |
| solve-0 g3 | 0.188 / 94% | 0.250 / 93% | 0.458 | 0.510 |
| solve-0 g5 | 0.168 / 95% | 0.229 / 94% | 0.443 | 0.505 |

(solve-0 g3-g5 trained on these positions, with outcome labels; g0, g1
of both runs did not.) One training step without P <= 27 data takes the
network's late-endgame values from 93% to 66% correct in sign. Search leaves
from roots at P 28-35 land there, which matches gen 1 being the weakest in
every match. Later generations recover partly but stay flat, while solve-0
improves at every P. (Why gen 2 recovers is not known.)

So exact values are not the problem; losing the data at P <= 27 is. The
search keeps evaluating those positions with the network.

## Decisions

- Stopped both runs after gen 5 (resumable with `arch/paired_loop.sh`).
- `arch/loop.sh` defaults to `SOLVE_P=0`.
- Entry 16's loss analysis stands (different position mix), but its
  rising value error was a real signal.

## If the solver comes back

It has to keep training data at P <= 27. Options, cheapest first:
1. Keep playing solved games out, using the solver only for the value
   labels (no speed gain; tests whether exact labels help).
2. Play the rest of a solved game with the solver's moves: exact value
   labels, and policy targets from the solver (winning moves), at almost
   no cost.
3. Value-only solver samples at P 18-27 (policy loss masked).
