# 03 — Supervised test bench, masking, and the symmetry-table bug

2026-09-25. Commits `9c69a00` (bench), `873988e` (the fix). Data:
`data/sweep1-broken-symmetries.log`; later sweeps in `data/`.

## The bench

- `Game::lineSpaces` (spaces in a line, by top type) and
  `build/libcorintho_features.so` (`pipeline/cpp/features.cpp`, via ctypes in
  `arch/features.py`): legal-move masks and line spaces computed by the
  engine's rules from a 70-float network input. The input's reserves are
  canonized, so the position is rebuilt with player 0 to move.
- `arch/dataset.py`: `runs/full-2` gens 1–4 -> `runs/sup/full2.npz`, one row
  per position (the stored identity copy), 2.84M positions, ~47 s. Checks:
  the 8 stored copies equal the table-transformed identity (0 mismatches:
  consistent with the table, which turned out not to mean correct); every
  move with visits is legal by the mask (0 violations in 2.84M).
- Facts: 29.5 legal moves per position on average, so **69% of the 96
  policy outputs are illegal**; 21.7 legal moves per position get no visits
  (the targets are sparse); 9.2% of positions have a line on the board;
  values are game results, -1/0/1.
- `arch/sup.py`: trains from scratch on gens 1–3, validates on gen 4 (a later
  generation, so no copies straddle the split). The policy is scored as the
  search uses it, renormalized over legal moves. AdamW, lr 2e-3 with warm-up
  and cosine decay, batch 4096, 20 epochs (~70 s for 130k parameters).
  Random symmetry per sample per step; switches for masking, line inputs,
  legal-mask inputs, model type/size/norm.

References on gen 4 (711k positions): gen_93 value MSE 0.5763, policy CE
1.6660, top-1 52.8%, loss 0.9928; gen_1 0.5741 / 1.6538 / 53.0% / 0.9875
(gen 4 was played by gen 1, so gen 1 has a home advantage).

## Masking (the developer's point 3)

Same 12x100 architecture, 20 epochs: softmax over all 96 (today's training)
loss 1.0698 (value 0.5916, policy 1.9130, top-1 45.8%); **softmax over legal
moves only: 1.0309 (0.5832, 1.7907, 48.4%).** Both heads improve.

## Sweep 1 (under the bug, see below)

All masked, 20 epochs. Loss: no random symmetries 0.9799; +16 line inputs
1.0179; +48 line inputs by type 1.0159; +96 legal inputs 0.9999; residual
256x4 BatchNorm **0.9285** (policy 1.5262, top-1 58.6%), LayerNorm 0.9474,
no norm 0.9562; MLP 64x6 1.0817; MLP 256x6 0.9464. Width helps most; line
inputs a little; the legal mask as input more.

## The bug

"No random symmetries" beat the augmented baseline and gen_93. Scoring both
models on each of the 8 orientations of the validation set: the augmented
model was worse on orientations 2 and 6 only (policy CE 1.827 vs 1.791,
top-1 41% vs 48%), which a symmetry-trained model on symmetric data should
not be.

Test: rotate states, recompute legality by the engine's rules, and compare
with the stored legality mapped through `move_symmetries`. Orientations 0,
1, 3, 4, 5, 7 agree in every position; **2 and 6 disagree in 96.2%.** With
rows 2 and 6 of `move_symmetries` exchanged, all 8 agree exactly (200k
positions). Row 2 of `space_symmetries` turns the board one way and row 2
of `move_symmetries` the other, so the two copies' policy targets were
rotated 180 degrees from correct: in 96% of positions they put visits on
illegal moves.

The tables are used only by `SelfPlayer::writeSamples`, and date from at
least June 2023. **Every generation trained with 2 of every 8 copies (25%)
of policy targets wrong.** Values were unaffected. Possibly part of the old
plateau; not measured.

Fix `873988e`: rows swapped, and `GameTest.SymmetryTablesAgree` checks all 8
symmetries against recomputed legality over seeded random play (fails on
the old table at the first ply; 46/46 pass). The driver was rebuilt, so
self-play from now on writes correct copies. The existing samples in
`runs/full-2` still hold the wrong copies; `arch/dataset.py` keeps only the
identity copy, which is unaffected.

After the fix (20 epochs, masked): baseline 0.5886 / 1.7579 / 51.0% / loss
1.0281; residual 256x4 BN 0.5558 / **1.4587 / 60.2% / 0.9205**.

## Sweep 2 (after the fix; `data/sweep2-fixed-symmetries.log`)

All masked, 20 epochs, lr 2e-3, pre-activation order. Loss (value MSE /
policy CE / top-1): baseline 12x100 1.0281 (0.5886 / 1.7579 / 51.0%); +16
line inputs 1.0157; +48 line inputs by type 1.0158; +96 legal-mask inputs
0.9969 (policy 1.6990); MLP 64x6 1.0773; MLP 256x6 0.9370; residual 256x4
BN 0.9205, LN 0.9341, no norm 0.9397. Train = val everywhere: the networks
are too small to overfit 2.1M positions. The developer's reading: extra
inputs matter less than size; line inputs are cheap and plausible, the
legal mask as input is not needed conceptually (it adds computation, not
information) — see entry 04 for the decision.

## Open: orientation carries value information

With the fix, "no random symmetries" still has the lower loss (0.9799),
almost all in the value (0.5455 vs 0.5886), and only in the stored
orientation (0.62–0.65 in the other seven). So orientation predicts the
result in this data. Likely lineage habits (these games were played by
networks trained on the broken copies, which may behave differently by
orientation), or the augmented task needing more epochs. The developer's
decision: randomize orientation regardless ("any biases in orientation are
a feature of some randomness in the initial training").
