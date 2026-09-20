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

## Measurement doctrine: decide on single thread, watch multi

**Measure and decide on single-threaded. Record multi-threaded as a guardrail
and as the production number.**

Single thread is the precise instrument. On a quiet machine on AC,
`st_engine_seconds` reproduces to the fourth decimal (3.9141 – 3.9174, a 0.08%
spread), and interleaved A/B arms come in at 0.4 – 1.8%. That resolves a 2%
change. Multi-threaded cannot approach it: it adds scheduling order, load
imbalance across workers, and this CPU's P-core/E-core asymmetry on top of
whatever is being measured.

Nearly everything on the optimization roadmap — bitboard move generation, the
`Node` accessors, `getFilteredProbs`, `lround` — is per-thread algorithmic work
with no mechanism to behave differently under threading. Measuring it there only
adds variance.

**But keep multi-threaded running**, because it catches one failure mode single
thread is blind to by construction: a change that is neutral or better per
thread while being worse under contention — extra memory traffic that only
saturates when 14 cores demand it at once, added allocator pressure, false
sharing on a new field. Parallel efficiency here is only 55%, and the ST and MT
profiles are nearly identical with no libgomp or futex hotspot, which points at
memory bandwidth as the limiter. A change trading instructions for memory
traffic would look like a win in ST and a loss in production. MT also costs
almost nothing: 0.53 s against ST's 3.9 s for the same work.

### The single-thread number overstates the production win

Measured on the LTO change, interleaved, on AC:

| Configuration | no-LTO | with-LTO | Delta |
|---|---|---|---|
| 1 thread, 50 games | 5.169s | 3.973s | **−23.2%** |
| 14 threads, 200 games | 2.758s | 2.240s | **−18.8%** |

**Multi-threaded captured 81% of the single-threaded win.** The parallel run is
closer to memory-bandwidth-bound, so removing compute pays less there. Expect a
similar discount on any change that removes instructions rather than memory
traffic, and quote the MT figure when claiming what a training run will see.

### Rules

- **Quote and decide on ST.** It is the number whose error bars mean something.
- **Watch MT for divergence.** ST improving while MT does not is a finding;
  investigate before committing.
- **Profile MT only** when the hypothesis is about threading — the arena
  allocator, the OpenMP schedule, the `vector<bool>` race.
- **Discount for production.** Training runs multi-threaded.

## Machine state invalidates cross-session comparisons

An identical build measured **77% slower** forty minutes after its baseline
(`st_engine_seconds` 4.08 → 7.22). Nothing in the code or the flags had changed:
the laptop had moved from AC to battery, where `intel_pstate` on `powersave`
caps turbo hard, and a browser was taking most of a core.

Consequences, which apply to every number in this directory:

- **Records are only comparable when `#ac_power`, `#governor` and `#loadavg`
  match**, and even then only loosely. `run_suite.sh` records all three.
- **Absolute timings are not portable across sessions.** Treat
  `bench/results/*.tsv` timings as valid within their own session and compare
  counters, which are simulated and unaffected by clock speed, across sessions.
- **For any A/B where the delta might be under ~20%, use `ab.sh`.**

```
./ab.sh <labelA> "<flagsA>" <labelB> "<flagsB>" [reps]
```

It builds both arms once into separate directories and then **alternates runs**
A,B,A,B,…, so slow drift lands on both arms equally instead of entirely on the
comparison. It reports each arm's median and spread, and refuses to endorse a
delta smaller than the worst arm's own spread.

`run_suite.sh` remains the right tool for recording a labelled point in time and
for the exact counters. `ab.sh` is the right tool for deciding whether one build
is faster than another.

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

### What each tool actually does

They are not interchangeable. Two of them do not even run your program.

**gprofng — statistical sampling.** Runs the real binary at full speed and
interrupts it on a timer, recording which function was executing. Attribution is
statistical, so accuracy grows with the square root of the sample count, and it
cannot reliably go below function granularity. What it *does* see is real time:
a function that stalls on cache misses or branch mispredicts simply accumulates
more samples. That makes it the only tool here that reflects the actual machine.
Use it to answer **"where does the wall clock go, really"**.

Caveats: it undersamples short runs badly (a 3-second run gave ~35 samples and
percentages that swung 10 points between runs), hence the large game counts. Its
absolute seconds capture roughly a tenth of true CPU time, so **read only the
percentages**. With `-flto` functions inline into their callers, so
`Node::initializeEdges` carries `getLegalMoves` and `TrainMC::doIteration`
absorbs `search` and `chooseNext`.

**callgrind — simulation, not measurement.** Valgrind executes the program on a
synthetic CPU and counts every instruction. The result is exact, deterministic
and repeatable to the instruction, with full caller/callee attribution — but it
counts *instructions, not time*. It has no model of superscalar execution, cache
or branch prediction, so a cache-missing pointer chase and an L1-hot arithmetic
loop look identical if they issue the same instructions. Costs 50–100× slowdown.
Use it to answer **"did my change remove the work I thought it did"**.

**cachegrind — the same simulator plus a cache and branch-predictor model.**
Adds simulated I1/D1/LL caches and a branch predictor, giving miss and mispredict
counts per function or per line. Still a *model*: a generic LRU cache sized from
the host CPU, with no prefetching, no out-of-order execution and no notion of
cores sharing a cache. Directionally reliable, not quantitatively exact. Needs
`--cache-sim=yes --branch-sim=yes` — both default to off, and without them it
silently degrades to instruction counts. Use it to answer **"is this code
branch-bound or cache-bound"**, which is the question that decides *which kind*
of rewrite helps.

**dhat — heap instrumentation.** Intercepts every `malloc`/`free` and tracks each
block's size, lifetime, and how many bytes of it are actually read and written.
That last part matters here: it can prove memory is allocated and never touched,
which is exactly the `to_eval_` case (`PLAN.md` §10.2). Use it to answer **"how
much am I allocating, and am I using it"**.

### Precise versus coarse, and which to reach for

| Question | Tool | Granularity |
|---|---|---|
| Where is the time going? | gprofng | function, statistical |
| Which *lines* are hot? | callgrind/cachegrind `lines` tier | source line, exact |
| Did my change remove the work? | callgrind | instruction, exact |
| Branch-bound or cache-bound? | cachegrind | function or line |
| How much am I allocating? | dhat | allocation site |
| **Is it actually faster?** | **`run_suite.sh`, no profiler** | **wall clock** |

The last row is the one that decides anything. Profilers tell you *where to
look*; only an unprofiled timed run tells you whether the change worked.

### Line-by-line needs a build without LTO

`-g` is in the default flags and costs nothing measurable, but **`-flto`
collapses the line tables**, so `cg_annotate --auto=yes` can only reach function
granularity on the normal build. The `lines` tier therefore builds into
`build-lines/` with `-O2 -g` and no LTO.

Inlining consequently differs from the production build. Use that tier to find
**which source lines are hot**, never to measure how fast anything is.

### Single thread by default; multi-thread on purpose

Profiling under threads buys nothing for most changes and costs clarity: samples
scatter across workers, the scheduler adds noise, attribution blurs. Nearly
everything on the roadmap — move generation, the `Node` accessors,
`getFilteredProbs`, `lround` — is per-thread algorithmic work that costs the same
on one thread as on fourteen. So `quick` is single-threaded and `all` does not
include `mt`.

Profile multi-threaded only when the hypothesis is *about* threading: allocator
contention, false sharing, load imbalance, memory bandwidth saturation. Here that
means the arena allocator, the OpenMP schedule, and the `vector<bool>` race.

**Benchmarking is different and always measures both.** A change can be neutral
single-threaded and harmful at fourteen — extra memory traffic that only
saturates under load, say — so `run_suite.sh` records `st`, `mt` and `big` every
time, whatever was profiled.

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

Line-level attribution (from the `lines` tier) puts the mispredicts on specific
statements:

| Mispredicts | Line | Function |
|---|---|---|
| 1,131,458 (9.3%) | `if (empty(move.space_from()) \|\| empty(move.space_to()))` | `canMove` |
| 1,086,686 (8.9%) | `if (board(space, piece_type))` | `top()`/`bottom()` |
| 936,741 (7.7%) | `if (edge_index < cur_->num_legal_moves() && ...)` | `getFilteredProbs` |
| 894,269 (7.3%) | `if (legal_moves[i])` | `initializeEdges` |
| 724,476 (5.9%) | `if (empty(move.space_to()))` | `canPlace` |

**About 37% of all branch mispredicts sit on five lines that the bitboard
rewrite deletes outright** — the per-space predicates become single mask
operations and the bitset scans become `ctz` iteration.

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
