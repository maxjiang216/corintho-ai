# 24 — Symmetry-tied and reserve-monotone small networks

2026-10-03. `pipeline/arch/nnue_distill.py` (`--tie`, `--mono-m`,
`--mono-m2`, `export`), `pipeline/arch/nnue_sweep.sh`. Code only: the
distillation comparison runs on the developer's machine (the data and the
GPU are there).

## Why

Planning alpha-beta (AB) self-play from a blank network, the developer
asked for two structural priors in the small value network, to test by
distillation first (the data of entry 21):

- **Symmetry.** The board has 8 symmetries; training so far uses a random
  symmetry per sample, and the network must learn the invariance. The
  developer chose weight tying over canonical orientation (which would break
  incremental updates when the orientation changes between parent and
  child).
- **More pieces is better.** Having more reserve pieces only adds moves.
  The exception is a draw by having no legal move (`Game::getLegalMoves`:
  no moves and no line is a draw), which a side with more pieces may be
  unable to claim. The developer: draws are rare and the network is used
  only above P ~27, where the solver does not reach, so a hard constraint
  is acceptable there.

Also decided in the discussion, not done here: AB self-play from scratch,
starting with a low lambda (more weight on game outcomes than on search
scores); no explicit parity input for now (if the second-player bias of
the AZ runs returns, a later run can add one); openings deduplicated up to
symmetry; small generations that grow when progress plateaus. "Merge" is
the proposed name for a move move (a stack moved onto another).

## The constraints

Inputs 64-66 are the side to move's reserves, 67-69 the opponent's
(count / 4). Clipped ReLU and tanh are non-decreasing, so the value is
monotone in an input when every path from it to the output has
non-negative weights. Options considered (diagrams in the session's
artifact "Monotone reserve networks"):

- A: every weight after layer 1 non-negative: exact, but layer 2 can never
  subtract a board feature.
- **B (chosen):** layer 1 split into board-only units (reserve weights 0)
  and reserve-aware units M (own reserves >= 0, the opponent's <= 0);
  layer 2 split into board-only units B2 (no input from M) and M2 (inputs
  from M >= 0, from board-only units any sign); output weights from M2
  >= 0. Every path from a reserve input is non-negative; board features
  stay free. `--mono-m 64 --mono-m2 16`: the last 64 layer-1 units and
  the last 16 layer-2 units are reserve-aware.
- C: an additive reserve term: no board-reserve interaction.

Kept by projection after each optimizer step (`Small.project`). Inference
is unchanged: the same dense network, the forced zeros simply stored. In
`small_net.h`'s reserve rows (`row[slot][count]`), B means rows that never
decrease with count for the side to move's slots, never increase for the
opponent's, and are equal for board-only units.

## Weight tying

Layer-1 units come in orbits of 8, one per symmetry: unit (g, k) applies
group g's weights to the board under symmetry k. Layer 2 is equivariant:
`W2[(q, l), (g, k)]` depends only on `l^-1 k`, so units keep their
relative orientation (summing each orbit would lose it). The output weight
is shared by an orbit, so the value is exactly invariant. 256 layer-1
units = 32 groups, 32 layer-2 units = 4 groups. Free parameters: 3,305
instead of 26,433. The network is built dense from the base parameters
(`Small.dense`); training saves the dense weights, so evaluation, int8
quantization and incremental updates do not change.

The group table comes from the engine's own symmetry tables
(`features.symmetries`): `prod[m][k]` is the symmetry `sym[m][sym[k]]`.

## Checks (CPU, random weights and synthetic data)

- Exact invariance: tied networks give the same value for all 8 symmetric
  copies (max difference 1e-5 in float, 2e-7 after export); untied: up
  to 2.0.
- Monotone: adding one reserve piece to the side to move never lowers the
  value, and one for the opponent never raises it, after random
  perturbation of all weights and projection (minimum change 0.0);
  unconstrained: -1.9.
- `export` writes `small_net.h`'s file (int32 hidden, W1 input-major,
  b1, W2 [32][h], b2, W3, b3): 105,736 bytes for h256, values equal to the
  trained model's within 1.5e-7.
- **Tied layer 2 can saturate for good.** A tied layer-2 or output weight
  moves its 8 copies together, so a step moves the output ~8x as far: at
  lr 3e-3 the 4 layer-2 groups saturated in 1 of 6 seeds (all 32 units dead,
  loss stuck at the target's variance). Tied layer-2 and output weights now
  train at lr / 8: 0 of 6 seeds dead, lower final loss on the synthetic
  target. Untied networks are unchanged (their rate is lr / 1).
- On a synthetic target (material plus a board count, 50k positions,
  1000 steps), tied networks converge more slowly than untied ones; this
  says nothing about the real data at 80k steps.

## To run

```
arch/nnue_sweep.sh TARGETS.npz TEST.npz
```

3 seeds each of plain, tied, monotone and both (h256, `--clip-w2 1.98`,
`--min-p 20`, 80k steps), into `runs/nnue/results.jsonl`. Compare P 28-35
and P 36+ against the teacher and P 23-27 against exact values with entry
21's table. Then `nnue_distill.py export` and `ab_match --small` for the
variants worth a match.

## Results (developer's machine, `nnue_sweep.sh`, 3 seeds each)

Training targets as in entry 21: exact values at P <= 27, the AZ teacher's
values above. The real labels in the test set are the exact values (P 23-27)
and the game outcomes (all P); outcomes are single noisy +-1 results, so
most of their ~0.49 error is irreducible. Means of 3 seeds:

| variant | P 23-27 exact MSE / sign | P 28-35 outcome | P 36+ outcome | P 28-35 vs teacher | train |
|---|---|---|---|---|---|
| plain | 0.339 / 89.2% | 0.4944 | 0.4916 | 0.0605 | 153 s |
| mono (64/16) | 0.354 / 88.7% | 0.4996 (+1.1%) | 0.4923 (+0.1%) | 0.0655 | 162 s |
| tie | 0.367 / 88.6% | 0.5078 (+2.7%) | 0.4961 (+0.9%) | 0.0744 | 188 s |
| tie + mono | 0.379 / 88.1% | 0.5134 (+3.8%) | 0.4961 (+0.9%) | 0.0796 | 200 s |

Seeds agree closely (plain P 28-35 outcome 0.4940-0.4948, mono
0.4984-0.5004), so the gaps are real.

- **Tying loses everywhere.** A tied h256 has 32 independent layer-1
  features against 256; the plain network already trains on a random
  symmetry per sample (positions stored once), so it learns most of the
  invariance and tying only removes capacity. The developer: keep the
  random orientation per sample, no tying.
- **Monotone costs ~1% on outcomes at P 28-35 and nothing at P 36+;** 4% on
  exact values at P 23-27 (solved in play). Against the teacher it looks
  like 8%, but the teacher is not monotone itself, so that measures
  disagreement with AZ, not error. Open: whether it costs strength
  (`ab_match` plain vs mono), and whether a larger reserve-aware share
  (`--mono-m 128/192`) closes the gap.

## Reserve-aware share, and the match

Larger reserve-aware shares (seed 0) do not close the gap: the cost is the
constraint, not the split.

| mono split | P 23-27 exact | P 28-35 outcome | P 36+ outcome |
|---|---|---|---|
| 64/16 (3 seeds) | 0.354 | 0.4996 | 0.4923 |
| 128/16 | 0.353 | 0.4998 | 0.4920 |
| 192/24 | 0.361 | 0.5005 | 0.4927 |
| plain (3 seeds) | 0.339 | 0.4944 | 0.4916 |

`ab_match` against MCTS (`runs/solve-0/gen_5`), entry 23's settings (200
games, seed 1, AB 2 s, MCTS 11,000 searches, 18 threads), seed-0 networks
in int8:

| network | AB vs MCTS | score | pairs (AB-MCTS-split) | depth | nodes/s |
|---|---|---|---|---|---|
| plain h256 | 97-2-101 | 0.490 | 14-16-70 | 10.4 | 2.22M |
| mono 64/16 | 97-4-99 | 0.495 | 15-17-68 | 10.4 | 2.13M |

Equal within noise (+-0.035): the monotone constraint costs no measurable
strength, so it stays (64/16) for AB self-play, for its guarantee. Both are
within noise of entry 23's 0.532 (that network trained on all P, these on
P >= 20).
