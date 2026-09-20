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

`perf` is unavailable on this machine (`/proc/sys/kernel/perf_event_paranoid` is
`4`, which blocks user-space sampling; lowering it needs root). Use callgrind:

```
valgrind --tool=callgrind --callgrind-out-file=cg.out ./build/selfplay_bench 3 400 16 1
callgrind_annotate --auto=no cg.out | head -30
```

Callgrind counts **instructions, not cycles**, and the run above is
single-threaded. It therefore under-represents allocator contention, cache
misses and false sharing — all of which are multi-threaded effects. Treat it as
attribution, not as a cost model.

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

## Revised C++ priority order

Evidence-based, replacing the ordering in `PLAN.md` §7.2:

1. **Bitboard move generation** (§13) — ~34% of instructions, ~90% of node cost.
2. **Inline the `Node` accessors into the header**, and/or ship `-flto` — ~13%,
   measured at 18% end-to-end.
3. **`getFilteredProbs` O(96) → O(legal)** — 6.1%, trivial fix.
4. **Root-only Dirichlet noise** (§5.1) — 3.9%, and it is a strength fix too.
5. **Replace `lround`** (a libm double call in `setProbs`) with `lrintf` — 2.6%.
6. **`writeGameState` nibble LUT** — 2.2%.
7. **Arena allocator** — only 2.5% single-threaded. Downgraded from its original
   top ranking, but callgrind cannot see the multi-threaded contention that
   motivated it, so re-measure at 14+ threads before dismissing it.
