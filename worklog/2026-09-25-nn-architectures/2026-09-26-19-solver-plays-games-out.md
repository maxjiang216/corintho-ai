# 19 — The solver plays solved games out, for samples

2026-09-26. Commits `fdffce1` (playout), `b89830e` (shared-tree fix
found on the way), `5dfaac0` (format), `314f9c2` (`MATCH_SOLVE_P`). Bench:
`bench/solver_playout.cpp`.

## The developer's design

After entry 18: "we should use exact labels... but there shouldn't even be
a distinction. that's just the outcome of the game [...] but yes, I
suppose we should train on these moves since they're still searching them
internally." Move choice: the winner plays the shortest mate, the loser
the longest, approximately ("choose from the nodes we did search if
there's pruning, getting the exact best mate is not that important"). For
draws, if easy: drawing moves that leave the opponent more losing moves.
Matches: the solver for both sides, since the user-facing engine would
use it.

## Implementation

`Solver::playOut(game, cap, line)` plays a position to the end of the
game, one `LineStep` (position, move, exact value) per ply:
- **Winner:** an immediate win if any; else the table's move for the
  position (the winning move the solve proved; `Solver::tableMove`),
  checked by solving its child (a table hit). A stand-in for the
  shortest win: the solver tries line-making moves first.
- **Loser:** every child is solved; the move whose refutation visited the
  most positions (a stand-in for the longest defence).
- **Draw:** every child is solved; among drawing moves, the one leaving
  the opponent the most replies that lose (grandchildren solved).
- Any capped solve: the line is dropped (the game keeps its result).

In training (`SelfPlayer`, solver mode), a game at the solve horizon, or
proven by the search (PROVEN, P 28-44), is sent to the pool with
`play_out`. `finalize()` appends the line as samples: the position, a
one-hot policy on the move played, and values that keep flipping sign
(`last_mover_value_` = the line's last value). A proven position whose
solve is capped keeps its proven value (no 20x retry). Test matches only
adjudicate, as before.

## Checks

- `bench/solver_playout`, 4000 P 27 positions from solve-0 gen 2 data, 6
  threads, warm tables: every line consistent (first value = the solve,
  values alternate, the last move ends the game with the promised
  result), none capped. Lines: won 6.0 plies, lost 4.9, drawn 12.7.
- Cost after the solve (10.5 ms mean): 31.3 ms mean when the winner also
  solved every child; **0.56 ms** (p90 0.02 ms) with the table's move.
- Self-play, 4000 games, solve-0 gen 5, 2 groups x 2000, 14 + 6 threads:

  | | wall | samples | waiting for solves |
  |---|---|---|---|
  | solver off | 56.0 s | 119.1k | |
  | end at P 27 (entry 14) | 47.6 s | 76.5k | 0.1 s |
  | **play out** | **49.5 s** | 98.3k | 0.1 s |

- Labels, the same run's compacted data (`arch/dataset.py`: symmetry and
  legality checks pass), solved exactly: 4000 positions at P <= 27 (all
  from solver lines, all one-hot policies): **label == exact 100%**;
  2000 at P 28: label == exact 94.25%, label > exact 0.
- Solver off: digest unchanged (ae0be3ec2a3dcb5b); test matches with
  `--solve-p 27`: `game_scores.txt` identical to the old binary.
- Assert build: clean in train mode (shared tree on and off) and test
  mode. It caught `SelfPlayer` nulling `players_[1]`, which a shared tree
  (entry 17) never gives a root (harmless in release); fixed in
  `b89830e`. Unit tests pass (50, GCC and clang).

## `loop.sh`

`MATCH_SOLVE_P` (default 0) adjudicates the matches at P <= it, both
sides alike. `SOLVE_P` stays 0 by default until the run below says
otherwise.

## Run

`runs/solve-line`: `NAME=solve-line GENS=5 SOLVE_P=27 MATCH_SOLVE_P=27`,
otherwise as solve-0 (start night-2 it1). ~310 s self-play per
generation (solve-0: ~355 s); 25.0 turns per game (solve-0 29.9; with
the solver line counted). Validation value MSE falls 0.329 -> 0.276
(solve-27 of entry 18 rose 0.39 -> 0.61).

Every generation against its start, 1600 games, seed 7005, both sides
adjudicated at P 27 (solve-0 and solve-27 from entry 18):

| gen vs its start | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| solve-line | 0.478 | 0.603 | 0.535 | 0.551 | 0.557 |
| solve-0 | 0.518 | 0.555 | 0.560 | 0.592 | 0.566 |
| solve-27 | 0.212 | 0.392 | 0.388 | 0.373 | 0.399 |

Head to head, solve-line gen 5 vs solve-0 gen 5, adjudicated at P 27,
seeds 7005 / 7105 / 7205: 746-15-839, 730-24-846, 719-29-852; together
2195-2537 decisive, **0.464** (z ~ -5, ~25 Elo weaker).

Value error by P (entry 18's test set: positions from solve-0 gen 3's
games; P <= 27 exact labels, MSE / sign-correct on decisive):

| network | P 18-22 | P 23-27 | P 28-35 | P 36+ |
|---|---|---|---|---|
| start | 0.256 / 93% | 0.378 / 88% | 0.564 | 0.539 |
| solve-line g1 | 0.277 / 92% | 0.329 / 90% | 0.517 | 0.526 |
| solve-line g3 | 0.295 / 91% | 0.326 / 90% | 0.505 | 0.522 |
| solve-line g5 | 0.307 / 90% | 0.332 / 90% | 0.499 | 0.524 |
| solve-0 g1 | 0.219 / 93% | 0.304 / 90% | 0.501 | 0.518 |
| solve-0 g5 | 0.168 / 95% | 0.229 / 94% | 0.443 | 0.505 |

(solve-0 g3-g5 trained on these positions; g1 is clean.)

## Reading

The collapse is gone, but late-endgame values still drift slowly down
instead of improving. The solver lines give ~5.5 positions of near-perfect
play per game. The search's leaves at P 18-27 come from varied, imperfect
play, which MCTS play keeps producing (~11 searched plies per game below
P 27 in solve-0). Exact labels on a few "perfect" positions do not cover
what the search evaluates.

Next test of exact labels on the positions the search needs: play games
out with MCTS as in solve-0, and relabel every sample with its exact value
(the P 27 solve for earlier positions, one solve per sample below it;
~1 ms each at P <= 27). No self-play time saved; it isolates the labels.
