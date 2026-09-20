# Corintho benchmark harness

Measurement infrastructure for the work described in `../PLAN.md`. Nothing here
changes engine behaviour; it exists so that optimizations can be shown to work.

## Why it is separate from CMake

The top-level `CMakeLists.txt` forces `--coverage -g -pg -O3` into every build,
so it cannot produce a binary whose timings mean anything. This harness uses its
own Makefile with clean release flags.

It also has **no dependency on Python, Cython, TensorFlow or gtest** — the whole
engine compiles from `corintho_ai/cpp/src` against the gsl headers alone. Engine
work therefore needs no working TF install and no GPU.

## Building and running

```
make            # build both benchmarks
make run        # build and run both at default sizes
make clean
```

### `micro_bench [corpus_size] [reps]`

Times the game-simulation layer against a corpus built by random legal playouts
from the starting position, so the position distribution resembles what search
actually visits. Covers `getLegalMoves`, `writeGameState`, `doMove`, and
`Node` construction/destruction.

### `golden [game|engine|all]`

Behavioural digests. `game` fingerprints `getLegalMoves` and `writeGameState`
over the fixed corpus; `engine` fingerprints the entire training-sample stream
of a fixed-seed, single-threaded run. Both are verified reproducible by
`run_suite.sh` before anything else is recorded.

### `selfplay_bench [games] [max_searches] [searches_per_eval] [threads] [seed]`

Drives `Trainer` exactly as `corintho_ai/python/main.pyx` does, but replaces the
neural network with a deterministic hash-based stub (`bench_util.h`). This
isolates the C++ engine and reports the same split as the generation logs in
`generations/*/play_time.txt`.

Because the stub hashes the game state, identical positions always receive
identical values — so this harness can also be used to measure position sharing
(`PLAN.md` §13.7).

**Caveat:** the stub is not the real network, so the search tree shape differs
from production (observed ~18 turns/game here vs 28.4 in the real run). Use this
for *relative* engine measurements, never as a prediction of generation time.

## The optimization workflow

Every optimization follows the same loop, so that each commit carries its own
evidence and can be assessed on its own.

```bash
./run_suite.sh before-<name>     # record, on a clean tree
#   ... make the change ...
./run_suite.sh after-<name>
./compare.py results/before-<name>.tsv results/after-<name>.tsv
```

`run_suite.sh` fixes every parameter that affects the result (game count, search
count, thread count, seeds, corpus size) so two records are always comparable.
It runs each benchmark 5 times and reports the **median** with min/max, because
single runs on this machine vary by a few percent.

`compare.py` enforces the two things that are easy to get wrong by hand:

- **Digests are compared exactly.** A changed digest on a run labelled as a pure
  optimization is a failure, not a curiosity.
- **Deltas are judged against the observed spread.** Anything inside the noise
  band is printed as `noise` and must not be claimed as a win.

### Which suite to run

| Component changed | Suite |
|---|---|
| `game.cpp`, `move.cpp` (move generation, state encoding) | `game` — plus `engine` to confirm it carries through |
| `node.cpp`, `trainmc.cpp`, `selfplayer.cpp`, `trainer.cpp` | `engine` |
| Build flags, allocator, anything cross-cutting | `all` |

When in doubt run `all`; it takes about a minute.

### Commit convention

One commit per optimization, or per class of mechanically identical one-line
changes. Each commit message states:

1. What changed and why it is faster.
2. The before/after table from `compare.py`.
3. Whether the digests held. If a digest moved, the commit must say so
   explicitly and justify it — a behaviour change disguised as an optimization
   is the failure mode this harness exists to prevent.

Records live in `results/` and are committed alongside the change.

## Profiling

`perf` is installed but unusable: `/proc/sys/kernel/perf_event_paranoid` is `4`,
which blocks user-space sampling. Lowering it to `1` needs root and widens what
unprivileged processes can observe, so it is a deliberate decision, not a
default. Everything below works without it.

```
./profile.sh <label> [quick|deep|heap|all] [threads]
```

| Tier | Tools | Run size | Cost | Answers |
|---|---|---|---|---|
| `quick` | gprofng, 1 and 14 threads | 300–400 games | ~90 s | Where does wall-clock time actually go |
| `deep` | callgrind, cachegrind | 3 games | ~15 s | Exact instructions; branch and cache behaviour |
| `heap` | dhat | 2 games | ~5 s | Allocation counts and peak heap |

### Why tiers, and what transfers between them

Heavyweight tools run 20–100× slower, so they get tiny runs; sampling tools cost
nothing, so they get large realistic runs. The question is whether the tiny run
measures the same thing as the big one. Measured:

| Config | requests/turn |
|---|---|
| games=3, searches=400 | 341.7 |
| games=10, searches=400 | 341.8 |
| games=40, searches=1600 | 1280.6 |
| games=300, searches=1600 | 1271.9 |

**Tree shape is governed by search count, not game count.** Changing games by
100× moves requests/turn by under 1%; changing searches by 4× moves it by 3.7×.

Hence the rule every tier follows: **fix the search count at 1600, vary only the
game count.** A tier that changed the search count would be measuring a
different workload, not the same workload more precisely.

**Transfers across run sizes:** instruction counts and attribution, branch
prediction behaviour, allocation counts and sizes.

**Does not transfer:** cache miss rates, especially last-level. These depend on
the size of the live search tree, which scales with games × threads. The 3-game
single-threaded run has a working set that fits in L2, so cachegrind reports
near-zero LL misses regardless of how bad real locality is. **Treat cachegrind's
D1/DL numbers as a lower bound.**

### Reading each tool

- **gprofng** — the only tool that sees the real threaded run. Undersamples short
  runs badly (a 3-second run gave ~35 samples and percentages that swung 10
  points), hence the large game counts. Its absolute seconds capture roughly a
  tenth of true CPU time, so **read only the percentages**. With `-flto` many
  functions inline into their callers: `Node::initializeEdges` carries
  `getLegalMoves`, and `TrainMC::doIteration` absorbs `search` and `chooseNext`.
- **callgrind** — exact and perfectly repeatable. The right tool for confirming a
  change removed the instructions it was meant to remove.
- **cachegrind** — needs `--cache-sim=yes --branch-sim=yes`; both default to off
  and without them it reports instruction counts only. `Bcm` is the interesting
  column for move generation, `D1mr` for the search.
- **dhat** — allocation counts. Relevant to the node arena and to the `to_eval_`
  oversizing (`PLAN.md` §10.2).

### Considered and rejected

- **samply**, **hotspot**, **nperf** — all wrap `perf_event_open` and are
  blocked by the same setting. If it is ever lowered, `samply` is the nicest of
  these: a Firefox Profiler view with no setup.
- **Intel VTune** — would genuinely help, since this CPU is hybrid (6P + 8E) and
  VTune understands OpenMP imbalance and per-core-type behaviour. Its hardware
  sampling also wants driver or perf access, so it does not dodge the
  restriction.
- **Coz** (causal profiling) — conceptually the best fit for the parallel
  scaling question, since it estimates what speeding up a region would do to
  end-to-end throughput rather than merely attributing time. Also built on
  `perf_event_open`.
- **heaptrack** — nicer UI than massif, but massif and dhat are already here.

---

## Baseline — 2026-09-20

Machine: i7-12700H (6P + 8E, 20 threads), 15 GB RAM, g++ 13.3.0.
Build: `-O3 -DNDEBUG -flto -march=native -fopenmp`.

### Microbenchmarks

Corpus of 20,000 positions: mean 26.02 legal moves, range [0, 48], 4.9% terminal.

| Function | ns/op |
|---|---|
| `Game::getLegalMoves` | **1211.4** |
| `Node` ctor+dtor | 1351.9 |
| `Game::writeGameState` | 123.7 |
| `Game::doMove` (incl. copy) | 13.1 |

**`getLegalMoves` is ~90% of the cost of constructing a node.** Allocation is
only ~140 ns of the 1352 ns. This is the headline result: it reorders the C++
priorities in `PLAN.md` §7.2, moving the bitboard rewrite (§13) ahead of the
arena allocator.

### Instruction attribution (callgrind, single-threaded)

`main` is 12.34% and is the stub evaluator — a harness artifact, excluded below.
Percentages are of total instructions.

| Cluster | Share | Components |
|---|---|---|
| **Move generation** | **~33.9%** | `getLegalMoves` 13.0, `Move::Move` 8.1, `canMove` 6.9, `initializeEdges` 3.8, `applyRowColLines` 2.0 |
| **Search + accessors** | **~25.2%** | `chooseNext` 12.0, `move_id` 3.7, `num_legal_moves` 3.0, `probability` 2.6, `drawn` 1.2, misc 2.1, `search` 0.5 |
| **Evaluation handling** | **~16.4%** | `getFilteredProbs` 6.1, `generateDirichlet` 3.9, `setProbs` 3.0, `lround` 2.6, `set_probability` 0.9 |
| Game state write | 2.2% | `writeGameState` |
| Allocator | 2.5% | `_int_malloc` 1.9, `_int_free` 0.6 |

### Thread scaling (200 games, 1600 searches, engine time)

| Threads | Seconds | Speedup | Efficiency |
|---|---|---|---|
| 1 | 24.07 | 1.00× | 100% |
| 4 | 6.85 | 3.51× | 88% |
| 8 | 4.16 | 5.78× | 72% |
| 14 | 3.10 | 7.76× | 55% |
| 20 | 2.72 | 8.85× | 44% |
| 28 | 2.89 | 8.33× | 30% |

Scaling saturates near the thread count and **regresses under oversubscription**.
The production run used `num_threads = 2 * cpu_count()` = 64 on a 32-vCPU box
(`wrapper.py`), which this suggests was leaving performance on the table.

Note this CPU is heterogeneous (6 performance + 8 efficiency cores), so a static
OpenMP schedule is especially poor here — which makes `schedule(dynamic)`
(`PLAN.md` §7.2) more valuable on this machine than on the uniform cloud box.

### Compiler flags (200 games, 1600 searches, 14 threads, 3 runs each)

| Flags | Seconds | vs production |
|---|---|---|
| `-O3` (current production) | 3.108 / 3.067 / 3.051 | — |
| `+ -march=native` | 2.991 / 3.047 / 3.100 | **no effect** |
| `+ -flto` | 2.505 / 2.535 / 2.512 | **−18%** |
| `+ both` | 2.445 / 2.526 / 2.845 | −18% |

**`-flto` alone is worth ~18%; `-march=native` is within noise.** This corrects
`PLAN.md` §7.3, which listed `-march=native` first.

The reason LTO helps so much: the one-line `Node` accessors (`move_id`,
`probability`, `num_legal_moves`, …) are defined in `node.cpp`, not `node.h`, so
without LTO they are cross-translation-unit calls that cannot inline. They
account for ~12.7% of instructions. Moving them into the header would capture
most of this without relying on LTO.

### Engine throughput reference

500 games, 1600 searches, 16 per eval, 14 threads: 7.6 s engine time,
9,095 turns, 11.5M stub evaluations, 1.52M requests/engine-second.

## What the profilers say, and the resulting priority order

The three tools disagree in a way that turns out to be informative rather than
contradictory.

**callgrind** (instructions) ranks move generation above search. **gprofng**
(time) ranks search above move generation. **cachegrind** explains why:

| Function | Branch mispredicts | Rate | L1 data read misses |
|---|---|---|---|
| `Node::initializeEdges` (move gen) | **6,455,211** (54.8%) | 14.8% | 1,723 |
| `TrainMC::doIteration` (search) | 3,948,907 (33.5%) | 7.9% | **1,180,440** (47%) |
| `Game::applyRowColLines` | 613,581 (5.2%) | 14.7% | 4,975 |

Program-wide branch misprediction rate is **9.2%** (11.8M of 127.5M branches).

**These are two different problems needing two different fixes:**

- **Move generation is branch-bound.** It accounts for 60% of all mispredicts at
  a ~15% rate — unsurprising for a 96-iteration loop of data-dependent branches
  that constructs a `Move` per ID and early-returns out of `canPlace`/`canMove`.
  At roughly 17 cycles per mispredict this is on the order of a quarter of all
  cycles. **Branchless bitmask code (`PLAN.md` §13) should eliminate nearly all
  of it**, which is a far stronger argument for the rewrite than instruction
  count alone.
- **Search is cache-bound.** It owns 47% of L1 data read misses but only 8%
  branch mispredicts — pointer-chasing over a linked tree. **This rescues the
  arena allocator**, which callgrind's 2.5% instruction share had demoted:
  locality is exactly what callgrind cannot see, and these numbers are a lower
  bound because the 3-game working set fits in L2.

dhat: **87,010 allocations for 2 games** (9.2 MB churn, 1.5 MB peak in 6,390
blocks — about 1,600 nodes × 2 allocations, matching `max_searches`).
Extrapolated to 25,000 games that is roughly **1.1 billion allocations per
generation**.

### Priority order

1. **Bitboard move generation** (`PLAN.md` §13) — ~34% of instructions, ~25-27%
   of time, ~90% of node construction cost, and 60% of all branch mispredicts.
2. **Inline the `Node` accessors into the header**, and/or ship `-flto` — ~13%
   of instructions, measured at 18% end-to-end.
3. **Arena allocator** — promoted back up on the cachegrind evidence. Confirm
   with cachegrind at a larger working set before committing to it.
4. **`getFilteredProbs` O(96) → O(legal)** — 6.1%, trivial fix.
5. **Root-only Dirichlet noise** (`PLAN.md` §5.1) — 3.9%, and a strength fix too.
6. **Replace `lround`** (a libm double call in `setProbs`) with `lrintf` — 2.6%.
7. **`writeGameState` nibble LUT** — 2.2%.
