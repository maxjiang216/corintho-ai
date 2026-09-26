# 2026-09-20 — Audit, the plateau, and a benchmark harness

Branch: `perf/training-overhaul`, off `main` at `c6539bc`.

Started as "audit the game simulation, training logic and MCTS engine for performance
optimizations". Ended up establishing that the last training run's plateau is probably not
a performance problem at all, and building the measurement infrastructure needed before
touching any code.

**Nothing has been optimized yet.** Every commit on this branch so far is documentation or
measurement tooling. That is deliberate: the audit's original priority ordering was wrong
in two places, and both errors were only caught by measuring.

## TL;DR

- **The last run plateaued at generation ~61** (rating flat 5477–5537 across gens 62–93),
  **but the training loss plateaued at generation ~13** and never moved again. Those are
  48 generations apart and the gap is the central mystery.
- **The rating number is self-referential** and cannot be trusted. Late-run promotions are
  statistically consistent with noise.
- **Self-play games are ~93% decided by which player moves first.** This may be the actual
  root cause and would make most performance work irrelevant to strength.
- **NN prediction was 62% of training wall clock and 92% of testing** on the CPU-only
  cloud box. The C++ engine was never the bottleneck there.
- **Found several correctness bugs that cost playing strength**, most importantly Dirichlet
  noise applied at every node rather than only the root, and a replay buffer that silently
  discards its input.
- **Built a standalone C++ benchmark harness** with golden digests, before/after
  comparison, and single- plus multi-threaded profiling. No Python or TensorFlow needed.
- **`getLegalMoves` is 928 ns and ~80% of the cost of constructing a node.** This reversed
  the audit's priority ordering: the bitboard rewrite now outranks the arena allocator.
- **`-flto` is worth 18%; `-march=native` is noise.** The audit had predicted the reverse.

---

## Details

### 1. What the last training run actually shows

Source data is in `generations/gen_*/` — 95 generations, run ending 2023-07-22, plus a
per-generation spreadsheet the developer supplied covering gens 1–78.

**Rating curve** (from `generations/gen_N/rating.txt`):

| Gens | Rating | Behaviour |
|---|---|---|
| 0–13 | 100 → 1388 | Fast climb |
| 14–61 | 1414 → 5497 | Steady climb |
| 62–93 | 5477 – 5537 | Flat. 32 generations, no net progress |

**Loss curve** (spreadsheet): 1.75 → 1.00 by gen 13, then flat at 1.04–1.13 for the next
55 generations.

This is the key observation and it is easy to miss: **the network stopped fitting its own
data 48 generations before the rating stopped moving.** Two readings, with very different
remedies:

- (a) The data kept improving but the network could not represent it — a capacity or
  trainability ceiling.
- (b) The network was already at the data's noise floor, and the gen 14–61 rating climb
  was substantially an artifact of the rating chain.

Not yet resolved. See *Next steps*.

**Why the rating cannot be trusted.** `update_rating` in `corintho_ai/python/main.pyx:263`
computes `new_rating = best_gen_rating - 400*log10(1/score - 1)` — a cumulative chain
against the previous best, with no fixed anchor. A run of noisy 0.52 promotions inflates it
monotonically. "5537 Elo" is not an absolute quantity.

Corroborating: test scores for gens 79–93 were 0.492, 0.493, 0.514, 0.509, 0.502, 0.521,
0.508, 0.511, 0.525, 0.502, 0.483, 0.489, 0.493, 0.528, 0.501. At 1600 test games
(σ ≈ 0.0125) against a promotion threshold of 0.52, promotion fires at roughly 1.6σ.

**The champion is gen 92, not 93.** `generations/gen_94/metadata.txt` records
`best_generation: 92`; gen 93 scored 0.5009 and was not promoted. Gen 92 is the model any
new run must beat.

**Second-player dominance.** From `generations/gen_93/testing_logs/score_verbose.txt`:

```
First player  wins 198/800 (24.75%)  draws 10/800 (1.25%)  losses 592/800 (74%)
Second player wins 596/800 (74.5%)   draws  5/800 (0.63%)  losses 199/800 (24.9%)
```

The spreadsheet's final unlabeled column runs 0.574 → 0.935 monotonically over 78
generations. **If that is the second-player score in training self-play**, it is very
likely the root cause: by gen 78 ~93% of training games are decided by parity, the value
head is predicting a near-constant, and almost every position the first player sees is
objectively lost. Learning would stall not because the optimizer failed but because
self-play stopped producing informative games.

> **Still unconfirmed.** The developer has not yet said what that column is. The priority ordering in
> `PLAN.md` depends on the answer.

### 2. Where the wall clock went

From `generations/gen_93/training_logs/play_time.txt` and `fit_time.txt`:

| Phase | Time | Share |
|---|---|---|
| Self-play total | 40m21s | — |
| — NN prediction | 24m58s | **62%** |
| — C++ self-play | 14m47s | 37% |
| Network fitting | 9m23s | — |
| Testing total | 6m54s | — |
| — NN prediction | 6m20s | **92%** |
| Generation total | ~48 min | |

78 generations ≈ 62 hours.

The run was **CPU-only** (the developer confirmed; `num_threads=0` in `corintho_ai/toml/train.toml`
resolves to `2 * cpu_count()`, and gen-90 metadata records 64, so `cpu_count()` was 32).

**No inference optimization of any kind was in use.** Grep of `main.pyx`, `wrapper.py` and
`main.py` found zero references to tflite, quantization, `jit_compile`, XLA, oneDNN or any
thread configuration. The training path is `load_model()` → `.predict()` on a full float32
Keras graph with `verbose=1` (a progress bar on every one of 4,539 calls).
`tflite_runtime` *is* installed by `scripts/bash/startup_script.sh` and
`corintho_ai/python/tflite.py` converts all 94 generations, but that is for the web app and
the tournament code — the training loop never imports it.

Rough efficiency estimate for the predict phase: ~250k rows per call × ~266 kFLOP per row
(133k params) over 0.533 s ≈ **200 GFLOP/s**, against an AVX2 FMA peak of roughly
1.4 TFLOP/s for ~16 physical cores. That is 10–15% of peak. The likely cause is that TF
executes op-by-op across the whole batch, streaming a 250k×100 float32 activation tensor
(100 MB) through DRAM per op — about 7.2 GB of DRAM traffic per predict call with no
fusion.

### 3. Correctness bugs found

Not performance items. Several plausibly cost playing strength directly.

| Location | Bug | Consequence |
|---|---|---|
| `trainmc.cpp:299` | `int32_t max_prob = 0` compared against a float probability in (0,1]; the assignment truncates to 0, so the condition is always true | `chooseHighProbMove` returns the **last** legal move, never the highest-probability one. Used as fallback at `:368` and `:436` |
| `trainmc.cpp:518` | `if (cur->drawn())` should be `cur_child->drawn()` | Draw deduction wrong; corrupts `kDeducedDraw` vs `kDeducedLoss` |
| `main.pyx:198-201` | `np.concatenate(...)` return value never assigned | **The replay buffer does nothing.** With `num_old_gens: 2`, the effective window is a single generation |
| `trainer.cpp:192,226` | `is_done_[i] = true` on a `std::vector<bool>` inside `#pragma omp parallel for` | Bit-packed concurrent writes: data race, lost updates |
| `trainmc.cpp:268-283` | Dirichlet noise applied to **every** evaluated node | AlphaZero applies it only at the root. ε=0.25 of noise on every interior node's priors corrupts every playout. Probably the largest strength bug in the engine |
| `wrapper.py` | `ModelCheckpoint(save_frequency=1)` — the real argument is `save_freq` | Silently ignored |
| `play_setup.py:19` | No `-O` flag; stale source paths (`../cpp/node.cpp` vs `../cpp/src/`) | Builds unoptimized, probably does not build at all |

Also flagged: `regularizers.L1L2()` is used with **default arguments** (`l1=0.01,
l2=0.01`) on all 12 Dense kernels. For a ~133k-parameter network that is heavy L1 —
plausibly enough to put a floor under the loss, which would explain §1's reading (a) vs
(b). A one-line test settles it.

### 4. Hardware, and why it changed the plan

The developer is moving development to a local laptop: RTX 3060 Laptop (6 GB VRAM), i7-12700H
(6 P-cores + 8 E-cores, 20 threads), **15 GB RAM with ~9 GB available**, 627 GB disk.

**RAM is the binding constraint, not the GPU.** And it makes one bug a blocker rather than
an optimization: `SelfPlayer` allocates `kGameStateSize * max_searches` floats per game
(`selfplayer.cpp:24`) = **448 KB per game**, of which only `searches_per_eval` (16) slots
are ever used = 4.5 KB. At 25,000 games that is ~11 GB of buffer alone — more than this
laptop has. A 25,000-game generation **cannot run locally** until it is fixed. After the
fix the same config needs ~112 MB.

**The GPU flips the bottleneck back to C++.** NN prediction was 62% of the cloud run; on a
3060 that collapses and the 14-core self-play engine dominates. Consequences: the C++ work
is promoted for local iteration speed, pipelining predict against self-play becomes
genuinely valuable (separate silicon), and TFLite/quantization becomes irrelevant locally
while still mattering if the final run returns to a CPU cloud box.

⚠️ **Timing conclusions do not transfer between the laptop and a CPU cloud box.** Strength
conclusions do. Keep the two kinds of result separate.

**The old training samples are gone.** `train_94.zip` (2 GB) has a valid PK header but is
truncated — Python's `zipfile` rejects it. No `.npz` sample files survive anywhere in the
repo except a toy run under `corintho_ai/python/logs/`. Models and logs survived; training
data did not. Any offline re-fitting experiment needs a dataset regenerated first.

### 5. The benchmark harness

Commits: `3c9b2a2`, `7f02c1e`, `410c2b9`, `2bb6d28`, `df8b058`, `c70cae2`, and the
dual-thread commit.

**The enabling discovery:** the entire engine compiles standalone from
`corintho_ai/cpp/src` against the gsl headers alone, with `g++ -std=c++17 -O3 -fopenmp`.
No Python, no Cython, no TensorFlow, no gtest. Verified by compiling each of the seven
core `.cpp` files individually. This matters because the system Python has neither cython
nor tensorflow installed, and it means all engine work can proceed without fighting a TF
install.

It also had to be separate from CMake: `CMakeLists.txt:14` sets
`CMAKE_CXX_FLAGS = "--coverage -fopenmp -g -pg -O3"`, forcing coverage and gprof
instrumentation into every build, so the existing build system cannot produce a binary
whose timings mean anything.

**Components**, all under `bench/`:

- `bench_util.h` — timing helpers, a `keep()` optimization barrier, SplitMix64, and the
  **stub evaluator**. The stub hashes the game state to derive an evaluation in (−1,1) and
  strictly positive priors. Deterministic, so identical positions always receive identical
  values — which also makes the harness usable for measuring position sharing later.
- `micro.cpp` — game-layer microbenchmarks over a corpus built by random legal playouts.
- `selfplay.cpp` — drives `Trainer` exactly as `main.pyx` does, substituting the stub for
  the network, and reports the same split as `play_time.txt`.
- `golden.cpp` — behavioural digests. `game` fingerprints `getLegalMoves` and
  `writeGameState` over the fixed corpus; `engine` fingerprints the entire training-sample
  stream of a fixed-seed single-threaded run. These make "this is a pure optimization" a
  checked claim rather than an assertion.
- `run_suite.sh` — records a labelled run with every parameter pinned, 5 reps reduced to
  median/min/max, and **rejects the run outright if the digests are not reproducible**.
- `compare.py` — compares two records, enforcing exact digest equality and printing any
  delta inside the noise band as `noise` rather than as a win.
- `profile.sh` — gprofng at 1 and 14 threads plus callgrind, with provenance headers.

**Two properties worth knowing about the harness:**

1. **The engine is deterministic across thread counts.** `st_turns == mt_turns == 971` and
   `st_requests == mt_requests == 1,248,785` exactly. Games are independent, each with its
   own RNG and tree, so thread count can be varied freely without disturbing the digests.
2. **The stub is not the network.** It produces ~18.2 turns/game against 28.4 in the real
   gen-93 run. Random priors are flat, so the search spreads wide; a real peaked policy
   goes deep and narrow. Tree shape sets the ratio of node *creation* (dominated by
   `getLegalMoves`) to node *revisiting* (dominated by `chooseNext`), so **a too-flat tree
   would overstate the value of the bitboard work.** This is the largest known weakness in
   the current numbers.

### 6. Profiler research

`perf` is installed but unusable: `/proc/sys/kernel/perf_event_paranoid` is `4`, which
blocks user-space sampling. Lowering it to `1` needs root and widens what unprivileged
processes can observe — a deliberate decision, not taken.

**Adopted: `gprofng`.** Ships with binutils, already installed, and does **not** use
`perf_event_open`. Unlike callgrind it samples the real multi-threaded run at full speed
and reports wall-clock CPU rather than instruction counts.

**Caveat discovered the hard way:** gprofng undersamples short runs severely. A 3-second
run yielded ~35 samples and percentages that swung 10 points between runs; `-p hi` made it
worse, not better. Both profiles in `profile.sh` are now sized for ~300+ samples
(`GAMES_ST=300`, `GAMES_MT=400`). Its absolute seconds are unreliable — it appears to
capture roughly a tenth of true CPU time — so **read only the percentages**.

**Rejected, with reasons** (recorded so the question does not get reopened):

| Tool | Why not |
|---|---|
| samply, hotspot, nperf | All wrap `perf_event_open`; blocked by the same setting. If it is ever lowered, samply is the nicest — Firefox Profiler view, no setup |
| Intel VTune | Would genuinely help — this CPU is hybrid and VTune understands OpenMP imbalance and P/E asymmetry — but its hardware sampling also wants driver or perf access |
| Coz (causal profiling) | Conceptually the best fit for the parallel-scaling question, since it estimates what speeding up a region would do to end-to-end throughput rather than merely attributing time. Also built on `perf_event_open` |
| heaptrack | Nicer UI than massif, but massif and dhat are already installed |

**Already installed and underused:** `cachegrind` (models cache behaviour — the right
instrument for judging the bitboard rewrite and for settling the arena-allocator question,
since locality is precisely what callgrind cannot see), `massif` and `dhat` for the heap
questions.

**Recommended pairing:** gprofng for wall-clock reality on the threaded run, callgrind to
verify the instruction delta of a specific change, cachegrind when the change is about
memory layout.

### 7. Prior art: big2-ai

`~/projects/big2-ai` solves the same move-generation problem and is the model to follow.

- `src/core/util.h:59` — `HandBits`: **threshold bit-planes** (`at1/at2/at3/at4`), one bit
  per rank, so a predicate over all ranks is a single mask operation.
- `src/core/util.cpp:60` — `hand_bits_tables()`: **superset-indexed lookup tables** built
  once, enumerating supersets with the `sup = (sup-1) & complement` trick.
- `compute_legal_moves` **emits** moves via `__builtin_ctz` rather than testing every
  candidate; `compute_legal_moves_into` avoids the allocation entirely.
- `move_to_cards.inc` — auto-generated `constexpr` tables.
- `Makefile:4` — `-O2 -march=native -fopenmp`; `Makefile:182-193` — a dedicated
  `benchmark` target under `-DBENCHMARK_MAIN`, separate from the instrumented test build.
- `scripts/profile_pimc_selfplay.sh` — a scripted profiling driver with perf/callgrind
  modes.

Corintho's `Game::getLegalMoves` does the opposite of all of this: it sets all 96 bits,
then **filters** by constructing a `Move` from each ID and testing it, recomputing
`top()`/`bottom()` from scratch on every query. The derivation of the bitboard replacement
is in `PLAN.md` §13; the short version is that all 48 place moves collapse to three mask
expressions and all 48 move-moves to eight shift-AND operations.

---

## Measurements

Machine: i7-12700H (6P + 8E, 20 threads), 15 GB RAM, g++ 13.3.0, Ubuntu 24.04, kernel
6.8.0-57.
Build: `-std=c++17 -O3 -DNDEBUG -flto -march=native -fopenmp -Wall -Wextra`.
Record: `bench/results/baseline.tsv` at commit `c70cae2`.
Reproduce: `cd bench && ./run_suite.sh <label> all 5`

### Microbenchmarks

Corpus of 20,000 positions: mean 26.02 legal moves, range [0, 48], 4.9% terminal.

| Function | ns/op (median of 5) |
|---|---|
| `Game::getLegalMoves` | **928.5** |
| `Node` ctor+dtor | 1153.9 |
| `Game::writeGameState` | 126.0 |
| `Game::doMove` (incl. copy) | 10.7 |

**`getLegalMoves` is ~80% of the cost of constructing a node.** Allocation accounts for
~225 ns of the 1154 ns. (Pre-LTO these were 1211 ns and 1352 ns, i.e. ~90%.)

### Engine, single vs multi-threaded

50 games × 1600 searches × 16 per eval, seed 12345:

| Metric | 1 thread | 14 threads |
|---|---|---|
| engine seconds | 4.089 | 0.531 |
| requests / engine second | 305,378 | 2,350,968 |
| turns | 971 | 971 |
| requests | 1,248,785 | 1,248,785 |

**Speedup 7.70×, parallel efficiency 55%.**

200 games at 14 threads (headline): 2.42 s engine, 3715 turns, 4,733,369 requests,
1,955,473 requests/engine-second.

### Thread scaling sweep

200 games × 1600 searches, engine time:

| Threads | Seconds | Speedup | Efficiency |
|---|---|---|---|
| 1 | 24.07 | 1.00× | 100% |
| 4 | 6.85 | 3.51× | 88% |
| 8 | 4.16 | 5.78× | 72% |
| 14 | 3.10 | 7.76× | 55% |
| 20 | 2.72 | 8.85× | 44% |
| 28 | 2.89 | 8.33× | 30% |

Scaling saturates near the thread count and **regresses under oversubscription**. The
production run used `2 * cpu_count()` = 64 threads on a 32-vCPU box.

### Profiles

`bench/results/profiles/baseline-{st,mt,callgrind}.txt`.
Reproduce: `cd bench && ./profile.sh <label> 14`

gprofng, exclusive CPU:

| Function | 1 thread | 14 threads |
|---|---|---|
| `TrainMC::doIteration` (search + chooseNext, LTO-inlined) | 48.33% | 52.05% |
| `Node::initializeEdges` (= `getLegalMoves`, inlined) | 25.23% | 27.21% |
| `main` (stub evaluator — harness artifact) | 9.73% | — |
| libc internals (malloc/free) | ~4.6% | ~4.6% |
| `Game::applyRowColLines` | — | 3.70% |

**The two profiles are strikingly similar.** Parallel overhead does not appear as a
distinct hotspot — no libgomp spike, no futex, no malloc explosion. That suggests the 45%
efficiency loss is memory bandwidth or this CPU's P/E core asymmetry rather than lock
contention. Worth confirming with cachegrind before acting on it.

callgrind instruction attribution (single-threaded, `main` = stub artifact at 12.34%):

| Cluster | Share | Components |
|---|---|---|
| Move generation | ~33.9% | `getLegalMoves` 13.0, `Move::Move` 8.1, `canMove` 6.9, `initializeEdges` 3.8, `applyRowColLines` 2.0 |
| Search + accessors | ~25.2% | `chooseNext` 12.0, `move_id` 3.7, `num_legal_moves` 3.0, `probability` 2.6, `drawn` 1.2, misc 2.1 |
| Evaluation handling | ~16.4% | `getFilteredProbs` 6.1, `generateDirichlet` 3.9, `setProbs` 3.0, `lround` 2.6 |
| Allocator | 2.5% | `_int_malloc` 1.9, `_int_free` 0.6 |

**gprofng and callgrind disagree in an informative way.** callgrind puts move generation
above search by instruction count; gprofng puts search above move generation by time. The
reconciliation is that search is pointer-chasing over a linked tree and is memory-latency
bound, so it costs more time per instruction. **This makes the arena allocator's locality
benefit more plausible than callgrind's 2.5% suggests** — and it is exactly what
cachegrind should be used to settle.

### Compiler flags

200 games × 1600 searches × 14 threads, 3 runs each:

| Flags | Seconds | vs production |
|---|---|---|
| `-O3` (current production) | 3.108 / 3.067 / 3.051 | — |
| `+ -march=native` | 2.991 / 3.047 / 3.100 | **no effect** |
| `+ -flto` | 2.505 / 2.535 / 2.512 | **−18%** |
| `+ both` | 2.445 / 2.526 / 2.845 | −18% |

The reason LTO wins: the one-line `Node` accessors (`move_id`, `probability`,
`num_legal_moves`, …) are defined in `node.cpp`, not `node.h`, so without LTO they are
cross-translation-unit calls that cannot inline. They are ~12.7% of instructions. Moving
them into the header would capture most of this without depending on LTO — which matters
because the production build via `setup.py` has no LTO.

---

## Corrections

Beliefs held earlier in this session that measurement overturned. Recorded because the
pattern is more useful than the conclusions: **both errors came from reasoning about code
structure instead of measuring it.**

1. **"The arena allocator is the biggest C++ win."** Asserted in the initial audit on the
   grounds that two mallocs per node under OpenMP contention would dominate. Measurement:
   allocation is ~225 ns of a 1154 ns node construction, and callgrind puts malloc+free at
   2.5% of instructions. `getLegalMoves` is ~80% of node cost. The bitboard rewrite
   outranks it. *Caveat:* gprofng's time-vs-instruction disagreement (above) means the
   allocator's locality benefit is still unmeasured, so this is a demotion, not a
   dismissal.

2. **"`-march=native` first, then `-flto`."** The audit listed vectorization flags ahead of
   LTO. Measurement: `-march=native` is within noise; `-flto` alone is 18%. The code is
   branch- and pointer-heavy, not vectorizable. The real win was inlining across
   translation units, which the audit had not connected to the accessors living in
   `node.cpp`.

3. **"Pipelining predict against self-play is ~1.6×."** Stated while assuming the cloud
   profile. On a CPU-only box both phases contend for the same cores, so the gain is much
   smaller; the claim is only correct with a GPU. Corrected in `PLAN.md` §7.1.

Two harness bugs of my own, both found and fixed:

4. **`#dirty` always reported `yes`.** The provenance header was written into the output
   file, so truncating an existing tracked record dirtied the tree *before*
   `git diff --quiet` ran. Fixed by capturing provenance before opening the file
   (`2bb6d28`).

5. **gprofng profiles were taken on runs far too short**, giving ~35 samples and
   percentages that moved 10 points between runs. The first ST/MT comparison drawn from
   them was noise. Re-sized to ~300+ samples.

---

## Research

- `~/projects/big2-ai` — bit-plane move generation, superset tables, benchmark target
  layout. See §7.
- `~/projects/through-the-ages-ai/worklog/` — the worklog convention this file follows.
- [samply](https://github.com/mstange/samply) — Linux/macOS sampling profiler, Firefox
  Profiler output. Needs `perf_event_paranoid ≤ 1`.
- [Intel VTune: profiling without sampling drivers](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-0/profiling-hardware-without-sampling-drivers.html)
- [Intel VTune: OpenMP imbalance and scheduling overhead](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-0/openmp-imbalance-and-scheduling-overhead.html)
- [Coz: finding code that counts with causal profiling](https://arxiv.org/pdf/1608.03676)
- [OpenMP compilers and tools](https://www.openmp.org/resources/openmp-compilers-tools/)

---

## Carried forward

First entry in this epic; nothing to sweep.

---

## Next steps

**Blocked on a decision from the developer:**

- **What is the last column of the spreadsheet?** (§1.) If it is the second-player score in
  training self-play, then §5.4 of `PLAN.md` — fixing the colour imbalance — becomes the
  highest-value work in the project, ahead of all performance work.
- Whether to lower `perf_event_paranoid` to 1, which would unlock samply and Coz.
- Whether the final full run goes to a GPU cloud instance (matching the laptop's profile)
  or a CPU one (matching the old run's). They favour different architectures.

**Ready to start, in order:**

1. **Bitboard `getLegalMoves`** (`PLAN.md` §13). Top target by both profilers. Must be
   validated by `digest_game` holding exactly.
2. **Move the `Node` accessors into `node.h`**, and add `-flto` to `setup.py`. Worth 18%
   measured; the header change makes it robust without LTO.
3. **Fix `to_eval_` oversizing.** Gates all local running (§4).
4. `getFilteredProbs` O(96) → O(legal); `lround` → `lrintf`; `writeGameState` nibble LUT.
5. **Root-only Dirichlet noise.** A strength fix that also removes ~4% of instructions.
   This one *will* move `digest_engine`, deliberately.

**Harness work still owed:**

- **Calibrate the stub's tree shape.** Currently 18.2 turns/game vs 28.4 real. Add a
  policy-concentration knob and tune until it matches. Until then, treat the split between
  move generation and search as approximate.
- **Eventually, run the real network in the benchmark.** Gen 92 is a 12×100 MLP — ~50 lines
  of C++ for the forward pass. Extract the weights once in a TF venv, dump to a flat
  binary, and the harness becomes authoritative rather than approximate.
- **Use cachegrind** to settle whether the arena allocator is worth it (see Corrections 1).
- **Instrument distinct canonical positions vs total evaluation requests**, by depth
  (`PLAN.md` §13.7). One number that sizes the entire position-sharing opportunity —
  recall that the starting position alone is evaluated 25,000 times per generation.
- **Set up S3** for benchmark and training data before anything outgrows the repo.
