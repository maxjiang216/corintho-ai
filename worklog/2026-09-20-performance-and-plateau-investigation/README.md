# Performance and plateau investigation

Started 2026-09-20. Branch `perf/training-overhaul`, then
`perf/engine-optimizations` for the engine work.

**Goal: improve the playing strength of the trained network**, not raw speed.
Speed is instrumental — it buys more games and more searches per game. Keep that
distinction in view, because most of what is recorded here is speed work and
none of it has yet been shown to move strength.

## Entries

| # | entry | what |
|---|---|---|
| 01 | [audit, plateau and benchmark harness](2026-09-20-01-audit-plateau-and-benchmark-harness.md) | Initial audit, the training plateau, and the `bench/` harness. **Read this first.** |
| 02 | [targeting the move generator](2026-09-20-02-targeting-the-move-generator.md) | Choosing the first target with profiles rather than intuition |
| 03 | [enabling LTO](2026-09-20-03-enabling-lto.md) | `-flto`, -23% engine time; why `-march=native` was noise |
| 04 | [the line-breaking fix](2026-09-20-04-line-breaking-fix.md) | **Two rule bugs live since 2023**, through all 95 generations |
| 05 | [optimizing the engine](2026-09-20-05-optimizing-the-engine.md) | Bitboards, arena allocation; 1.83x ST / 1.69x MT |
| 06 | [the vector&lt;bool&gt; race](2026-09-21-06-vector-bool-race.md) | Real race, never observed to fire, and why |
| 07 | [applying moves, and what counters cost](2026-09-21-07-applying-moves-and-what-counters-cost.md) | `doMove`; **cost model for Ir / cache / mispredicts** |
| 08 | [writeGameState](2026-09-21-08-writegamestate.md) | 12.7x; the ISA the training module actually builds with |
| 09 | [to_eval_ sizing](2026-09-21-09-to-eval-sizing.md) | 100x oversized *and resident*; 3x peak RSS |
| 10 | [the last column, and colour imbalance](2026-09-21-10-the-last-column-and-colour-imbalance.md) | **Second player wins ~74% of training games.** (Its claim about the gate is retracted — see 12.) |
| 11 | [where the 74% comes from](2026-09-21-11-where-the-74-percent-comes-from.md) | Not the rules bug, not strength. Structural, found by generation 6. |
| 12 | [the promotion gate](2026-09-21-12-the-promotion-gate.md) | **Retraction.** The gate was not passing noise; I misread it. What changed anyway, and why. |
| 13 | [lround in setProbs](2026-09-22-13-lround.md) | Bit-identical (checked over every float in range). −6.94% Ir, −3.8% engine time. |
| 14 | [Dirichlet noise RNG](2026-09-22-14-dirichlet-rng.md) | splitmix64 for noise only. Reproducible, not bit-identical; checked statistically. −3.7% time per request. |
| 15 | [lazy selection](2026-09-22-15-lazy-selection.md) | chooseNext scores one unvisited edge. Same search (2 float-tie mismatches in 4.6M). −17% engine time on stub, −12.5% with the real network. |
| 16 | [child stats in the parent](2026-09-22-16-parent-stats.md) | Selection reads contiguous per-parent arrays mirrored by each child. Bit-identical digest. −18% engine time (real network), −9% (stub); +18% peak memory. |

## The two entries to read if you read nothing else

- **10** is the one that most plausibly explains the plateau: a ~74%
  second-player win rate, which makes the value target nearly constant. Read it
  with **12**, which retracts its claim about the promotion gate.
- **04** is the other one that plausibly affects strength. The engine allowed
  9,686 forbidden moves and forbade 5,673 legal ones per 6.1M positions, for
  every generation trained so far.
- **07** holds the measurement framework everything else is judged by: what an
  instruction, a cache miss and a branch mispredict each cost here, why they
  must not be summed, and why `Ir` and `Bcm` have to be read together.

## Standing rules learned the hard way

- **Derive tables, never transcribe them.** `line_breakers` acquired 13 errors
  by hand-transcription (entry 04). Every table since is `constexpr`-generated.
- **Build a positive control before believing a null result** (entry 06).
- **Read the disassembly before optimizing.** Twice now the obviously-wasteful
  source was already gone at `-O3` (entry 07).
- **Measure share before optimizing.** `doMove` and `writeGameState` looked
  equally optimizable; one was 0.23% of the program and one 4.4% (entries 07-08).
- **Wall clock on this machine is untrustworthy with a browser open.** Arm
  spreads reach 13-16%. Prefer exact counters; pin to a P-core; interleave arms.
- **Match the gate to the change.** Digests catch behaviour changes; they cannot
  catch an out-of-bounds store. Shrinking a buffer needs memcheck (entry 09).
- **A differential reference that is kept passing is not a record of old
  behaviour.** `game_reference.cpp` was re-frozen by the fix commit; comparing
  against it compares the fix with itself. Use a `git worktree` (entry 11).
- **Suspiciously exact agreement is a bug signal.** Two 200,000-game arms that
  match to five decimals did not agree; they were the same code (entry 11).
- **Eager work on the evaluation path is paid by every leaf.** Two thirds of
  evaluated nodes never get a child; measure who uses the work first (entry 15).
- **A change to the random stream changes the workload.** Raw counter totals
  from the two arms are then different games; normalize per request over many
  seeds (entry 14).
- **Read what was recorded before reconstructing it.** The promotion threshold
  and the full best-generation chain were on disk the whole time. Replaying them
  from the ratings produced a wrong answer that then motivated a change (entries
  10-12). A replay also diverges permanently after one wrong decision, so it
  cannot be spot-checked at the end.

## Still open

- **Confirm the colour imbalance with a real network** under the fixed rules
  (entry 11). Needs a TF or tflite runtime, which is not available locally. This
  is the one measurement that would settle whether 0.74 survives the entry-04
  fix in the regime that matters.
- ~~Fix the promotion gate~~ — done in entry 12, though for a weaker reason than
  entries 10-11 gave. It now adapts to the test-game count instead of being a
  fixed 0.52.
- Strength impact of the entry-04 rules fix: needs a fixed-weights match.
  Generation 92 was trained under the buggy rules.
- `web/engine.js` and `web/line_breakers.js` still carry every line-breaking
  defect; `web/corintho.js`'s rules overlay is also wrong (RULES-CHECKLIST #3).
- `setup.py` passes no `-march`; decide deliberately rather than by omission
  (entry 08).
- `scripts/bash/build.sh` has never been build-verified with `-flto` — no
  cython or TensorFlow available locally.
- Stub tree shape: 18 turns/game against 28.4 real, but children-per-node
  distribution matches the real network closely (entry 15). `bench/selfplay_nn`
  now measures with the real network directly.
- Optional: split `syncStats` into value/flag paths; 4-byte child handles
  (+2% speed, overcommit risk) (entry 16).
- Transposition table unevaluated; distinct-position instrumentation not built.
- S3 for large benchmark artifacts not set up.
