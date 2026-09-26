# 11 — Legal-move generation, profiled and tightened (engine-wide)

2026-09-26. Commits `000f2cc`, `f5f7aaa`, `5df391e`. These change engine
functions that self-play also uses, so every step was gated as an engine
change.

## Gate (every change)

- solver: identical results and node counts at P <= 24 and P <= 27;
- 50/50 unit tests;
- self-play sample digests (gen_93 on the CPU network, 64 games, seeds 1
  and 2): de6300d58e8b4c75, 3ccc9044f1e27859 before and after;
- for generator changes also `bench/rulecheck` (59,405 comparisons, 0
  mismatches) and `bench/verify` (200,000 positions identical to the
  reference implementation);
- timing: the solver at P <= 24, one thread pinned, three interleaved
  rounds against HEAD; instruction counts where timing noise (~10% between
  gate runs) is comparable to the effect.

## Profile (solver, P <= 24, no-inline build)

Full legal-move generation (for line-making children: the win check and
reply counts need it): 2.91M calls, ~750 instructions each, 41% of all
instructions: lineBreakers ~245, findLines ~239, computeSpaceInfo ~118 (its
bit gathering alone 5.6% of everything), basicLegalMoves ~108.

## Changes

| change | effect |
|---|---|
| computeSpaceInfo: four pext instead of the gathering routine | 0.30 -> 0.28-0.29 s |
| basicLegalMoves: left/right stack moves via pext (0x7777, 0xEEEE) instead of walking set bits | 0.33-0.34 -> 0.28-0.31 s |
| lineBreakers: extend check as a bit of top_plane instead of top()'s branch chain | 0.32-0.33 -> 0.30-0.31 s; instructions 2.647G -> 2.594G |
| findLines with the three types in one word (as hasLine) | reverted: 0.30 -> 0.31-0.32 s (the per-type loop skips types with no tops, which is common) |

Net (no-inline build, instructions): 5.30G -> 4.58G (-13.5%). Non-BMI2
builds keep the previous code paths.

Remaining (same build): findLines 14%, the child classification loop 13%,
hasLine 12%, search / table probes 10%, basicLegalMoves 8%, doMove 7%.

A pinned single-thread solve of the P <= 24 set now takes ~0.28-0.31 s on
a quiet machine; earlier ~0.9 s figures in entry 08 were taken with a
stray process and a training loop running.
