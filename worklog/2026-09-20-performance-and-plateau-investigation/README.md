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
| 17 | [terminal propagation](2026-09-22-17-terminal-propagation.md) | **Correctness fix.** Drawn positions were deduced as losses (typo, audit CPP-1). 1% of real-network moves were made from a wrong root result; full-tree audit now 0 wrong, 0 missed. Golden digest blind to it. |
| 18 | [the network in-process](2026-09-23-18-in-process-network.md) | Start of the inference plan (laptop GPU, PyTorch, C++ owns inference). Hand AVX2 MLP matches tflite to 8e-7 on 3.5M real states; ~450 GFLOP/s on 20 threads, ~9× the pipe. **Hybrid CPU: static OpenMP schedules wait on E-cores.** |
| 19 | [the engine at scale](2026-09-23-19-engine-at-scale.md) | **~0.9 GB per 1k games in flight** (25k at once needs ~23 GB): keep only enough in flight to saturate the GPU. `schedule(dynamic, 1)` on the game loops: −20% engine time at 1k games, −32% at 2k, same games. |
| 20 | [GPU throughput](2026-09-23-20-gpu-throughput.md) | fp32 saturates at 16k–32k rows, ~175 ns/row end to end, so a generation is ~2.6 GPU-min against ~3.4 engine-min: **the engine is the bottleneck**. TF32/fp16 are off by up to 0.18 in probability, so not usable. tflite matches Keras to 7e-5. E-cores add 22%. |
| 21 | [orchestration reasoning](2026-09-23-21-orchestration-reasoning.md) | Synthesis of 18–20: the developer's decisions, the self-play driver design, and why the engine (not the GPU) sets every orchestration tradeoff. **Read before building the driver.** |
| 22 | [computeSpaceInfo](2026-09-23-22-space-info.md) | SWAR plane extraction replaces the 16-space loop (finishes entry 02's Stage 3). Same games. −12% stub, −4.9% real network single-thread, **−2.4% at 20 threads: stub instruction counts overstate production gains.** |
| 23 | [long profile](2026-09-23-23-long-profile.md) | 5.5 h, real network. **Serial request gather = 16% of engine time at 20 threads** (the data-copying suspicion, confirmed). syncStats 19% of engine CPU, selection ~35% of instructions, tree teardown ~48% of LL misses. Scaling 9.1× on 20 threads ≈ hardware capacity. Ranked targets inside. |
| 24 | [batch slots](2026-09-23-24-batch-slots.md) | Games write network inputs into fixed slots of the Trainer's batch, compacted as games finish. **Serial gather gone: −16% engine+gather at 20 threads**, same games, +0.56% rows. The first version without compaction was +48% rows. Makefile header dependencies fixed. |
| 25 | [child stats updates](2026-09-23-25-child-stats-updates.md) | syncStats: store only the changed field, flags only on change, one update per level per pass. **−2.6% at 20 threads**, bit-identical. The parent-only-copy design (B) was slower: **it turned store misses into load misses.** Why 18.8% of CPU became 2.6%. |
| 26 | [selection vectorized](2026-09-23-26-selection-vectorized.md) | chooseNext scores all children branch-free (bit-mask selects, `Edge` without bitfields), then takes the first max: the loop vectorizes. **−4.5% at 20 threads** (−13.6% stub single-thread), bit-identical. Float instead of double is next, not bit-identical. |
| 27 | [PGO](2026-09-24-27-pgo.md) | GCC PGO trained on real-network games: bit-identical but **+3.2% slower**. The CPU network is 97% of profile counts and set the "hot" cutoff above once-per-search code, so hot calls were un-inlined ("call is unlikely"). Without the network profile: **−0.1%, neutral**. Stub 12–15% slower in every arm. Not adopted; the harness stays for compiler comparisons. |
| 28 | [compilers](2026-09-24-28-compilers.md) | g++-13 vs g++-14 vs clang-20, all bit-identical. **clang-20: engine −4.4% at 20 threads (−12.6% single-threaded), wall −26%** (the CPU network is much faster). g++-14 same as 13. clang needs libomp (`-fopenmp=libgomp` silently disables OpenMP) and a VLA fix (`488a74a`). **Adopted** (`019ee6d`). Found on the way: unit tests segfaulting since entry 16 (null parent for a new root, fixed in `87533d8`); **unit tests are now part of the gate**. |

## Session 2026-09-23: the network in-process, the long profile, and three engine changes (entries 18–26)

Training moves to this laptop (RTX 3060 Laptop GPU, i7-12700H), with PyTorch
for fitting and C++ owning inference. Measured:

- **In-process CPU network** (`Mlp`): matches tflite to 8e-7 on real positions
  (entry 18).
- **The engine's game loops now use a dynamic OpenMP schedule:** −20 to −32%
  engine time. The static split waited on staggered starts and on E-cores
  (entry 19).
- **Memory:** ~0.7 GB per 1k games in flight. The staggered start is real,
  −30% peak (entry 19).
- **GPU fp32:** ~175 ns/row end to end, saturating at 16k–32k rows. TF32 and
  fp16 are not accurate enough with this network (entry 20).
- **Per generation, the engine (~3.4 min) outlasts the GPU network
  (~2.6 min).** The engine is the bottleneck again, so work returns to it
  before the driver is built (entry 21).
- **`computeSpaceInfo` without its per-space loop:** −12% engine time on the
  stub, but −4.9% with the real network and −2.4% at 20 threads. **Stub
  instruction counts overstate production gains.** Rank targets on
  real-network multithreaded time (entry 22).
- **Long profile (entry 23):**
  - the serial request gather is 16% of engine time at 20 threads;
  - `syncStats` is 19% of engine CPU;
  - selection is ~35% of instructions;
  - thread scaling is close to the hardware's capacity.
- **Fixed batch slots (entry 24):** the serial gather is gone, −16%
  engine+gather at 20 threads.
- **syncStats (entry 25):** −2.6% at 20 threads. Making the parent's
  arrays the only copy was slower, because it turned store misses into
  load misses.
- **Selection (entry 26):** branch-free and vectorized, −4.5% at 20
  threads, bit-identical.

Engine changes this session, real network at 20 threads, each measured
against the commit before it, with the same games in every pair:

| entry | change | engine-side time |
|---|---|---|
| 19 | `schedule(dynamic, 1)` on the game loops | −20% (1k games) to −32% (2k) |
| 22 | `computeSpaceInfo` without the per-space loop | −2.4% |
| 24 | fixed batch slots: the serial gather is gone | −16% (engine + gather) |
| 25 | store only the changed child statistic | −2.6% |
| 26 | branch-free, vectorized selection | −4.5% |

New verification tool: `SAMPLE_DIGEST=1 selfplay_nn …` hashes every training
sample at full depth with the real network (`5a5616e`). Baselines are in
`data/child-stats-updates/runs.txt`: seeds 1–4, 300 games, 1600 searches, 20
threads. Entries 25 and 26 match them exactly, and any bit-identical change
must too.

## Session 2026-09-22: selection and search engine (entries 13–16)

Engine time per NN request, each step measured against the one before it:

| entry | change | stub | real network (`model_93`) |
|---|---|---|---|
| 13 | `lround` → add-and-truncate (bit-identical) | −3.8% | not measured |
| 14 | splitmix64 for Dirichlet noise (new stream, reproducible) | −3.7% | not measured |
| 15 | `chooseNext` scores only the best unvisited edge | −17.2% | −12.5% |
| 16 | child statistics mirrored in the parent (bit-identical) | −9.0% | −18.3% |

Roughly 30% less engine time per request compounded on the stub, likely more
with the real network. Peak memory +18% (entry 16). In training the network
dominates wall time, so end-to-end gains are smaller. None of this has yet
been shown to change strength.

New tool: `bench/selfplay_nn` runs the engine against a real tflite model
(`pip install ai-edge-litert` in a throwaway venv). It reproduces real game
length (28.7 turns vs 28.4 recorded) where the stub gives ~18. Use it before
trusting any stub-only result.

**Where to resume:**
- **(2026-09-23) Engine first, then the self-play driver (entries 18–23).**
  Entry 23 has the ranked targets from the long profile:
  1. ~~The serial gather (16% at 20 threads)~~: done in entry 24 with fixed
     batch slots.
  2. ~~`syncStats`~~: done in entry 25 (−2.6%). Most of its 19% was the
     unavoidable first touch of the parent's cold stats line in backup.
  3. ~~Selection~~: branch-free and vectorized in entry 26 (−4.5%). Next:
     float instead of double (not bit-identical; needs a mismatch rate and
     statistics).
  4. Tree teardown in bulk.
  5. `receiveEval` and Dirichlet cache misses.

  Then ~~PGO~~ (entry 27: neutral at best, not adopted), ~~compilers~~ (entry
  28: clang-20 −4.4% engine, bit-identical, adopted; a Nix flake for the
  toolchain is planned), manual hints (cold paths, prefetch in backup), then
  the driver (entry 21). Gate now includes the CMake unit tests (entry 28).
  - Rank on real-network, 20-thread engine time with paired seeds.
  - Scaling is near hardware capacity (9.1× on 20 threads), so entry 22's
    memory-bound guess is weakened.
  - Environments: uv venvs, recreated from `bench/README.md`.
    `bench/long_profile.sh` reruns the whole profile unattended.
- The strength questions are unchanged and more important: colour imbalance
  with a real network under the fixed rules (entries 10–11), and the strength
  effect of the entry-04 rules fix. `selfplay_nn` runs the real network
  locally now. It does not report results yet, but adding the winner per game
  would answer the first question.
- Entry 17 (draw deduction, fixed) left two behaviour questions open: proven
  draws stop gaining visits but move choice goes by visits, and proven results
  do not back up into ancestors' values (MCTS-Solver). Both need a strength
  test. Gens 79–94 were trained with the bug; gens ≤78 were not.
- `main.pyx` / `wrapper.py` gate changes (entry 12) are still unbuilt: no
  Cython or TensorFlow here.

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
- **Check the generated code before explaining a speedup.** Twice the source
  suggested a mechanism (a reload, a cmov) the compiler had already removed or
  never emitted (entry 15).
- **An unchanged digest only covers what its workload reaches.** The golden
  engine run (20 games, 200 searches) never deduces a draw, so the entry-17
  fix left it unchanged while changing 1600-search games (entry 17).
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
