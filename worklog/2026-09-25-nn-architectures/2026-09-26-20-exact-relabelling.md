# 20 — Exact value labels on games played out as usual

2026-09-26. Commits `5b8531d` (`--relabel-p`), `4ac5d77` (`RELABEL_P`),
`7bc065d` (`SEED`). Run `pipeline/runs/relabel-27`; replicate
`pipeline/runs/solve-0b`.

## Why

Entry 19: training on the solver's own lines put off-policy positions
below P 27 (near-perfect play), and the network's values drifted on the
positions its search actually visits. The developer asked whether the
solver's labels were like off-policy training. Two parts: *which positions*
(off-policy for solver lines, which is what the measurements showed
hurting) and *which value target* (the perfect-play value instead of the
outcome of this self-play's play; untested). This run isolates the second.
The developer: "let's test the relabelling". Node solving (`--node-p`)
was discussed and set aside: it could only cover P <= ~19 (P 15-27:
4-35% of samples, 5-50% of network rows; half of all rows are leaves at
P <= 27, from roots at P >= 28).

## Implementation

`--relabel-p 27` (training; not with `--solve-p`): games are played to
the end by the search, unchanged. When a move is chosen at a root with P
<= 27, that position is submitted to the solver pool. In `finalize()`,
each such sample gets its exact value, and the samples before the first
one get its value flipped back (the result had both sides played
perfectly from there). A capped solve (retried at 20x) keeps the outcome
label. Policy targets: the visit counts, unchanged.

## Checks

- 4000 games, solve-0 gen 5, seed 5: positions and policies identical to
  a solver-off run; values differ on 9.0%; 117.8k of 119.1k samples
  relabelled (the rest: games won before P 27); 41.8k solves, none
  capped; wall 49.5 -> 52.8 s (+6.7%), no waiting for solves.
- Solved exactly (solver_bench): 3000 samples at P <= 27, label == exact
  100%; 1500 at P 28, label never above the exact value (94.1% equal).
- Solver off: digest unchanged. Assert build clean. Unit tests pass.
- `runs/relabel-27` gen 1 data vs `runs/solve-0` gen 1 (same start, same
  seed): the same 746,245 positions and policies; 9.0% of values differ.
  The first training step differs only in its labels.

## Results

`NAME=relabel-27 GENS=5 RELABEL_P=27 MATCH_SOLVE_P=27`. Self-play ~365 s
per generation (solve-0 ~355 s). Validation value MSE 0.292 -> 0.258
(solve-0: 0.369 -> 0.384, against its noisier outcome labels).

Every generation against its start, 1600 games, seed 7005, adjudicated at
P 27 for both sides:

| gen vs its start | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| relabel-27 | 0.477 | 0.541 | 0.557 | 0.574 | 0.580 |
| solve-0 | 0.518 | 0.555 | 0.560 | 0.592 | 0.566 |
| solve-line (entry 19) | 0.478 | 0.603 | 0.535 | 0.551 | 0.557 |

Head to head, adjudicated at P 27:
- relabel-27 gen 1 vs solve-0 gen 1 (paired data): 772-21-807, 0.489
  (z -0.9).
- relabel-27 gen 5 vs solve-0 gen 5, seeds 7005 / 7105 / 7205:
  729-19-852, 740-24-836, 742-19-839; together 2211-2527, **0.467**
  (z -4.6).
- relabel-27 gen 5 vs solve-line gen 5 (seed 7005): 832-22-746, 0.527.

Value error by P (entry 18's test set; P <= 27 exact labels, MSE /
sign-correct on decisive; P >= 28 outcome labels from solve-0's own
games, which favour solve-0):

| network | P 18-22 | P 23-27 | P 28-35 | P 36+ |
|---|---|---|---|---|
| relabel-27 g1 | 0.219 / 93% | 0.290 / 91% | 0.513 | 0.526 |
| relabel-27 g3 | 0.201 / 94% | 0.241 / 93% | 0.494 | 0.527 |
| relabel-27 g5 | 0.187 / 94% | 0.228 / 93% | 0.491 | 0.525 |
| solve-0 g1 | 0.219 / 93% | 0.304 / 90% | 0.501 | 0.518 |
| solve-0 g3 | 0.188 / 94% | 0.250 / 93% | 0.458 | 0.510 |
| solve-0 g5 | 0.168 / 95% | 0.229 / 94% | 0.443 | 0.505 |

The endgame values are fine now (gen 1, a clean comparison: 0.290 vs
0.304 at P 23-27).

## Open: is solve-0 gen 5 just a good draw?

Both solver variants lose to solve-0 gen 5 by about the same margin
(0.464, 0.467), and every run so far shared its self-play and training
seeds, so run-to-run noise was never measured. `loop.sh` now takes
`SEED` (offsets both; 0 reproduces earlier runs). Replicate:
`NAME=solve-0b GENS=5 SEED=1 MATCH_SOLVE_P=27`: solve-0's settings,
other seeds. Its gen 5 against its start: 0.557 (solve-0: 0.566).

Head to head at gen 5, adjudicated at P 27, seeds 7005 / 7105 / 7205:

| | per seed | decisive | score | z |
|---|---|---|---|---|
| **solve-0b vs solve-0** | 747-22-831, 755-20-825, 753-11-836 | 2255-2492 | **0.475** | -3.4 |
| relabel-27 vs solve-0 | (above) | 2211-2527 | 0.467 | -4.6 |
| **relabel-27 vs solve-0b** | 813-19-768, 822-35-743, 814-20-766 | 2449-2277 | **0.518** | +2.5 |

**solve-0 gen 5 is a good draw.** The same settings with other seeds lose
to it by as much as the solver variants do. Run-to-run noise at 5
generations is about +-25 Elo, well above a 4800-game match's resolution
(~5 Elo). A head to head between two single runs measures mostly the
runs' luck, not the treatments.

## Conclusions

- Exact relabelling is **not measurably different** from outcome labels
  at 5 generations: -23 Elo against one solve-0 run, +12 against the
  other. It fixes nothing that was broken (the endgame values were fine
  with outcomes), and costs 6.7% self-play time.
- solve-line (entry 19, 13% faster self-play per generation) also sits
  inside this noise at gen 5 (0.464 against solve-0), but its endgame
  values drift, which may compound over longer runs.
- Only solve-27 (entry 18) is clearly worse (0.377 against solve-0, far
  outside the noise).
- **Method:** comparing training treatments needs several runs per arm
  (different `SEED`), or much longer runs, before a head to head means
  anything. Matches between generations of one run and matches against
  fixed anchors do not show run-to-run noise either.
