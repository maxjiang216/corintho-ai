# 23 — A long, realistic engine profile: the serial gather, syncStats and selection are where production time goes

The developer asked for a long, unattended profile "in a more realistic run", to
see which parts of the engine take the most time, and specifically whether
copying data between the classes costs more than it should. It ran for 5.5
hours on 2026-09-23 via `bench/long_profile.sh` (commit `52b2e82`, which
includes entry 22's change). Everything is in
`bench/results/long-profile-2026-09-23/`:

- `progress.log`, `provenance.txt`
- `scaling.tsv`
- `temps.tsv`: package temperature and mean clock every minute
- `gp-mt-*.out`: per-run metrics
- `reports/`: gprofng function reports, and `callgrind-merged.txt`
- the raw `callgrind.out*` files

The gprofng experiment directories (~580 MB) stay local and are git-ignored.

All runs use the real network (`model_93.mlp`, in-process `Mlp` on the CPU),
1600 searches and 16 per evaluation. The package ran at 85–95 °C under load,
so every number here is thermally constrained. So is an overnight training
run.

## 1. Thread scaling on identical games (1,000 games, 2 reps, unpinned)

| threads | engine s (mean) | speedup vs 1 | gather s | gather / engine |
|---|---|---|---|---|
| 1 | 78.09 | 1.00 | 1.28 | 1.6% |
| 2 | 37.65 | 2.07 | 1.18 | 3.1% |
| 4 | 20.42 | 3.82 | 1.19 | 5.8% |
| 6 | 14.63 | 5.34 | 1.15 | 7.9% |
| 8 | 12.41 | 6.29 | 1.21 | 9.7% |
| 12 | 10.21 | 7.65 | 1.50 | 14.7% |
| 14 | 9.53 | 8.19 | 1.50 | 15.7% |
| 20 | 8.59 | 9.09 | 1.40 | **16.3%** |

- Requests are identical in all 16 runs (35,903,101), so these are the same
  games.
- Peak memory: 0.80–0.90 GB per 1,000 games.
- Repetitions agree within ~3%.

**Scaling is close to what the hardware offers.**

- 1 → 6 threads is 89% efficient, which is the P-cores.
- Past 6, threads land on hyperthreads and E-cores. Those are worth well
  under a P-core each (entry 18: an E-core is ~⅓ of a P-core on dense
  arithmetic, and hyperthreads share a core), so 9.1× on 20 threads is
  roughly the machine's capacity.

This **weakens entry 22's guess** that the 20-thread engine is heavily
memory-bound. The halving of that change's gain from single-thread to 20
threads is not explained here. Plausible causes are the different mix of
core types and clock speeds at 20 threads, but it is not established.

**The serial gather does not scale at all.** `gather_seconds` covers
`Trainer::num_requests` and the `writeRequests` copy, on one thread between
the engine and the network. It stays at ~1.2–1.5 s whatever the thread count,
so as the engine gets faster it becomes a bigger share: **16% of engine time
at 20 threads**. At 2,000 games (the gprofng runs) it was 2.88 s against
18.4 s of engine (15.6%), and very steady across 8 runs.

This is the developer's "data copying between classes" suspicion, confirmed.
Each game's TrainMC writes its rows into its own `to_eval_` buffer, and the
Trainer then copies them all, serially, into the batch buffer.

## 2. Where engine CPU time goes at production shape (gprofng, 8 × 2,000 games × 20 threads)

Clock sampling (`-p hi`) is statistical, but it covers all threads with real
memory pressure: 8,835 CPU-seconds sampled in total. The network (`Mlp`)
takes 73.5% of that; this CPU backend will be replaced by the GPU. The engine
functions, ~2,206 CPU-s:

| function (LTO inlines much of the engine into `TrainMC::doIteration`) | CPU s | share of engine |
|---|---|---|
| `TrainMC::doIteration`, self: selection, backup, receiveEval, noise | 1,451 | 65.8% |
| **`Node::syncStats`** | **414** | **18.8%** |
| `Node::initializeEdges`, including move generation | 178 | 8.1% |
| `Node::~Node`, tree teardown | 112 | 5.1% |
| `Node::Node` | 51 | 2.3% |
| OpenMP runtime (libgomp) | ~30 | ~1.4% |

- **Load balance is fine.** libgomp waiting is ~1.4% of engine CPU, so the
  dynamic schedule from entry 19 is doing its job.
- **`syncStats` is 7.5% of instructions (below) but 18.8% of production CPU
  time.** It is a memory-latency cost: it writes a child's statistics into the
  parent's stats block, a line that is often not in cache.
- **`~Node` is 0.7% of instructions but 5.1% of CPU time**, and 48% of
  simulated last-level misses (below). Deleting the discarded part of the
  tree after every move walks cold nodes.

gprofng 2.42 recorded **no line numbers** for this LTO build ("Source
location not recorded"). Its single-thread run sampled only 0.36 s of a 35 s
run and is not usable. Both are noted in the script. Callgrind supplied the
line-level view.

## 3. Exact instruction and cache attribution (callgrind, 48 games, 1 thread, 4.9 h)

The run is engine-only (`--toggle-collect='Trainer::doIteration*'`) with
cache and branch simulation: 1,314 turns, 1.66M requests, 11.0B instructions,
~6,600 instructions per request.

Each 30-minute checkpoint dump zeroed the counters, and later dumps refer to
function names defined in earlier ones. `callgrind_annotate` cannot read them
individually, so `bench/callgrind_merge.py` (new) sums the 10 files in order.

| function | Ir | L1 data misses | LL misses | mispredicts |
|---|---|---|---|---|
| `TrainMC::doIteration` (inlined) | 75.8% | 69.0% | 31.3% | 74.4% |
| `Node::initializeEdges` | 13.9% | 4.9% | 17.0% | 17.3% |
| `Node::syncStats` | 7.5% | 14.5% | 1.0% | 3.4% |
| `Node::Node` | 1.5% | 2.6% | 2.1% | 2.5% |
| `Node::~Node` | 0.7% | 6.7% | **48.1%** | 2.3% |

The simulated LL cache is 24 MB against ~35–40 MB of trees for 48 games. So
last-level misses are only indicative of what 2,000 games (1.4 GB) do in
production.

Hot lines, from `reports/callgrind-merged.txt`:

- **Selection (`chooseNext`)**, `trainmc.cpp:613–630` plus
  `Node::probability` at `node.cpp:104`: ~35% of all instructions and ~35% of
  all mispredicts.
  - Per visited child it computes two float divisions
    (`evaluation / visits` and `/ (visits + 1)`) and reads the prior from the
    edge array as a `uint16` times the denominator.
  - It branches unpredictably on the skip flags and on `u > max_eval`.
- **`receiveEval` → `getFilteredProbs`**, `trainmc.cpp:225`: 11.9% of all L1
  misses, from reading each request's 96-float network output row and the
  edge move IDs.
- **`generateDirichlet`**, `trainmc.cpp:258`: 7.6% of L1 misses. The 4 KB
  gamma table is evicted by tree traffic between uses.
- **`syncStats`**, `node.cpp:271–273`: 13.6% of L1 misses on two lines.
  - The backup loop calls it **twice per level**: `increase_evaluation` and
    then `set_all_visited(false)` each re-sync the same child.
  - The descent calls it again through `increment_visits`.
  - Each call recomputes the parent's stats layout and writes all three
    fields.
- **`~Node`**, `node.cpp:24` and `node.h:191`: 44% of LL misses on two lines,
  the recursive delete touching each cold node and its edge block.
- Move generation (`initializeEdges`, `node.cpp:390`, `util.h:163–168`): 15%
  of LL misses and 11% of mispredicts, mostly `forEachMove` and the
  `EdgeBlock` allocation.

## Ranked targets for production engine time

Estimated from the 20-thread shares above, largest first:

1. **The serial gather: ~16% of engine wall time at 20 threads.** It grows as
   the engine gets faster. The fix belongs in the self-play driver (entry 21):
   games write their rows straight into the batch buffer from inside the
   parallel loop, at offsets known in advance, and counts are tracked
   incrementally rather than by walking every game. Removing the copy entirely
   also removes a pass over ~5–9 MB per iteration.
2. **`syncStats`: ~19% of engine CPU.** One sync per node per update instead
   of two or three (merge `increase_evaluation` with `set_all_visited`, and
   `increment_visits` on the way down). Keep the stats pointer and layout
   instead of recomputing them. The mirrored statistics are entry 16's design,
   so the digest must stay bit-identical.
3. **Selection: ~35% of instructions and mispredicts.**
   - Keep the float prior in the child-stats block next to evaluation and
     visits, so it is not a separate edge array.
   - Avoid the divisions, for example by keeping `1 / (visits + 1)`.
   - Make the max branch-free.

   This must be bit-identical, or checked like entry 15.
4. **Tree teardown: ~5% of CPU and nearly half the LL misses.** Free
   discarded subtrees in bulk (per-game arena regions), or hand them to a
   deferred free, instead of a recursive walk over cold nodes.
5. **`receiveEval` and Dirichlet L1 misses: ~20% of L1 misses together.**
   Prefetching the next request's output row, or processing requests in
   memory order, might help. This is speculative.

Items 2, 3 and 4 are engine changes measurable with the existing tools on
real-network 20-thread paired A/Bs. Item 1 is a driver change.

## Tooling lessons

- Checkpoint dumps plus string compression make callgrind's files unreadable
  one at a time. Use `callgrind_merge.py`, or pass `--compress-strings=no`.
- gprofng gives function-level data only for this LTO build. For line-level
  sampling at production shape, profile a `-O2 -g` non-LTO build, or try
  `-gdwarf-4`. Untested.
- gprofng's "main thread" view does not isolate serial work under OpenMP.
  Explicit timers do (`gather_seconds`).
