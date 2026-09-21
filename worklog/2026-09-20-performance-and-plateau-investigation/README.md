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

## The two entries to read if you read nothing else

- **04** is the only one that plausibly affects strength. The engine allowed
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

## Still open

- **The spreadsheet's last column is unidentified** (0.574 -> 0.935 monotonic
  over 78 generations). Open since entry 01 and never answered. If it is the
  second-player self-play score, colour imbalance outranks every performance
  item here.
- Strength impact of the entry-04 rules fix: needs a fixed-weights match.
  Generation 92 was trained under the buggy rules.
- `web/engine.js` and `web/line_breakers.js` still carry every line-breaking
  defect; `web/corintho.js`'s rules overlay is also wrong (RULES-CHECKLIST #3).
- `setup.py` passes no `-march`; decide deliberately rather than by omission
  (entry 08).
- `scripts/bash/build.sh` has never been build-verified with `-flto` — no
  cython or TensorFlow available locally.
- Stub tree shape uncalibrated: 18.2 turns/game against 28.4 real.
- Transposition table unevaluated; distinct-position instrumentation not built.
- S3 for large benchmark artifacts not set up.
