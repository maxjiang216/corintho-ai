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
