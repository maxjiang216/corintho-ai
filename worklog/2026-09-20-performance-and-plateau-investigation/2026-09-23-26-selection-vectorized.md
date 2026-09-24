# 26 — Selection scores children branch-free and vectorizes: −4.5% at 20 threads, bit-identical

This is target 3 from entry 23. Selection (`TrainMC::chooseNext`) was ~35% of
engine instructions and ~35% of mispredicts in the real-network callgrind
profile. Commit `8f20319`. Raw data is in
`data/selection-vectorized/runs.txt`.

## What the loop does, and who calls it

PUCT, per child: `score = −evaluation/visits + P·c·√N / (visits + 1)`. A
skipped child (solved win or loss, or exhausted) is left out, and a known draw
scores only the prior term. The highest score wins, ties going to the first
child.

Temporary instrumentation (100 games, real network):

- 21.0M `chooseNext` calls scanned 350M children, **16.6 per call**;
- **70% of the children scanned belong to nodes with ≥20 children**, the
  high-traffic nodes near the root. Entry 15's histogram counted nodes, which
  are mostly narrow, not selection work;
- 7.1% of children scanned were skipped, and drawn children almost never
  appear (5,117).

So selection is a hot loop over ~30 children, which is worth vectorizing.

## Before

One loop per child, with:

- three data-dependent branches: skip, draw, and `u > max_eval`, which is
  effectively random;
- two **double-precision** divisions. The `-1.0` and `1.0` literals promote
  the arithmetic to double; the score is rounded to float afterwards.

## After

```cpp
// Pass 1: every child, no branches
weighted = float(edges[i].probability()) * denominator * v_sqrt;   // = probability(i) * v_sqrt
normal   = -1.0 * evaluation[i] / visits[i] + weighted / (visits[i] + 1.0);
u        = selectFloat(flags[i] & kDrawnChild, weighted, normal);
score[i] = selectFloat(flags[i] & kSkipChild, -inf, u);
// Pass 2: first index holding the maximum (the old strict > scan)
```

Two things blocked the vectorizer. They were found with `-fopt-info-vec` at
the LTO link, and in standalone variants (`vectorizer_variants.cpp`):

1. **A float `?:` stays a branch.** Under GCC's default `-ftrapping-math` it
   is not converted to a select: "control flow in loop". `selectFloat`
   performs the same selection on the bit patterns with an all-ones or
   all-zeros mask, and that becomes a vector blend. `-fno-trapping-math`
   would also work, but it would have to be set on every build of the engine
   (bench, Cython, CMake). A local helper cannot be forgotten.
2. **Bitfields.** `Edge` was `uint16_t move_id : 7, probability : 9`, and the
   vectorizer rejects bitfield reads. It is now one `uint16_t` with
   `move_id() = bits & 0x7F` and `probability() = bits >> 7`: same size and
   layout, and still uninitialized by default as before (zeroing 48 edges per
   node would cost time). Selection reads the edges and the denominator
   through `ChildStats`, and `kMaxEdges` became public for the score buffer.

The scoring loop now compiles to 4-wide `vdivpd`: 32 packed divides appear in
`TrainMC::doIteration` across the vector, epilogue and versioned paths.

## Why it is bit-identical

- **Same arithmetic.** Each score uses the same operations in the same order
  and precision: float weight × denominator × √N, promoted to double for the
  divisions, rounded to float. IEEE vector division is correctly rounded per
  lane, like scalar division. `-ffp-contract` is off under `-std=c++17`, so
  nothing is fused.
- **Same winner.** Pass 2 is the same strict `>` scan over the same values,
  so it still picks the first maximum.
- **Skipped children** now get a score computed and then replaced by −∞.
  That is harmless: visits are always ≥1, so the divisions are defined.

Checks: golden digests unchanged; `verify` PASS; `selfplay_nn` sample digests
identical to the `5a5616e` baselines on 4 seeds × 300 games (real network,
1600 searches); assert build clean with the same digest.

## Measurements

Callgrind, stub, 3 games:

- instructions 587.0M → 541.9M, **−7.7%**;
- conditional branches **−17.2%**;
- mispredicts 2.17M → 1.92M, **−11.5%**;
- L1 read misses −2.8%, L1 write misses +5% (the score buffer).

Timing, interleaved, identical turns in every pair:

| | engine time |
|---|---|
| real network, 20 threads, 6 × 1,000 games | **−4.5%** (95% CI −5.1 to −3.9) |
| stub, 1 thread pinned to a P-core, 12 × 200 games | **−13.6%** (CI −14.5 to −12.6) |

It is the entry-22 pattern again. Single-threaded stub runs overstate what 20
threads on real trees get; at 20 threads the memory side and the other work
weigh more.

## Next, discussed with the developer: float instead of double

The developer asked whether float would be enough. Probably yes for play:

- the score is already stored as a float, so double only affects the
  rounding of intermediate steps;
- float and double pick different children only at near-ties, agreeing to
  6–7 digits, where either choice is equally good by the formula's own
  standard.

It would give 8 lanes instead of 4, faster divides, and no conversions; the
guess is a further 2–5% of engine time. But it is **not bit-identical**, and
the games diverge after the first differing choice. So it needs the entry-14
style of evidence: a decision mismatch rate (a temporary build computing both
versions), behaviour statistics over many seeds, and a timing A/B. It is a
separate change, so that any effect on behaviour can be traced to it.
