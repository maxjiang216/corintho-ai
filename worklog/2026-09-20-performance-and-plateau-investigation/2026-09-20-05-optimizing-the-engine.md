# 2026-09-20 — Optimizing the engine, measured by line count

Branch: `perf/engine-optimizations`. Seven commits, `375a01a` through `cb59949`.

Picks up after the line-breaking fix, which was correct but 12–16% slower.
The goal was to recover that and go past it. It did, by a wide margin, but only
after a measurement technique that should have been used from the start.

## TL;DR

- **The engine is 1.83× faster single-threaded, 1.69× at 14 threads**, measured
  interleaved on a quiet machine.
- **`getLegalMoves` is 5.03× faster** — 922.3 ns → 183.3 ns on a fixed corpus.
- **Branch mispredicts down 65.8%**, instructions down 35.8% against the branch
  point.
- **The technique that mattered**: bucketing `getLegalMoves` by how many line
  shapes are present. Aggregate numbers said "slower"; the buckets said
  *which* path.
- **Three of my measurements this session were invalid** and one crashed
  program nearly got reported as a 94% win. See *Corrections*.

## What was done

| commit | change | effect |
|---|---|---|
| `375a01a` | decode moves from a `constexpr` table | −15% instructions |
| `8d46b43` | `MoveMask`; iterate edges by bit scan | −7.8% mispredicts |
| `45512aa` | line breakers as one mask per line | −2.8% instructions |
| `de6801e` | find lines by shifted AND | −19% on every position |
| `5dc42ac` | basic legality by mask arithmetic | **−78.7%** on the common path |
| `cf59113` | index edges directly; hoist edge count | −18.9% mispredicts |
| `cb59949` | per-thread arena allocator | −26.4% cache misses |

Every one holds both digests and passes `verify` and `rulecheck`.

## The measurement that unlocked it

Three commits went into optimizing line *handling* on the assumption that was
where the regression lived. The gains were real but small — 2.8% here, 7% there.

Bucketing `getLegalMoves` by the number of line shapes present showed why:

```
  lines      cost
      0     740.1 ns     <- 90% of positions, and the whole function cost
      1     978.6 ns        646 ns before the correctness fix
      2     974.8 ns
```

A position with **no lines at all** cost 740 ns, against 646 ns for the entire
pre-fix function. The regression was never in line handling. It was in line
*detection* (`findLines` scanned 34 shapes unconditionally) and in basic
legality (96 per-move calls), both paid by every position.

Aggregate counters could not show this: they said "13% slower" without saying
which path. **Bucket by the condition that distinguishes the paths.**

## Two structural facts that did the work

**All 34 line shapes are constant-stride runs.** 1 across a row, 4 down a
column, 5 along a4–d1, 3 along d4–a1. The short diagonals are not special:
`SD1 {1,6,11}` and `SD3 {4,9,14}` are stride-5, `SD0 {2,5,8}` and
`SD2 {7,10,13}` stride-3. So `p & (p>>s) & (p>>2s)` finds every run of a type at
once, and 90% of positions exit on a zero test.

**Only two move pairings can be legal.** `canMove` needs
`bottom(from) - top(to) == 1`, and with both spaces occupied each is in
{0,1,2}, so only column-bottomed-onto-base-topped and
capital-bottomed-onto-column-topped can satisfy it. All 48 moves become two
shifted ANDs per direction.

The developer spotted a third: **a legal move landing on a line space always breaks that
line.** `top(to) == type` forces `bottom(from) == type + 1`, and a stack's top
is at least its bottom, so the arriving top can never equal the line's type.
Verified over 315,385 such moves, zero exceptions. That removed a whole dynamic
term I was about to build a three-mask lookup for, two-thirds of it dead.

## Where the remaining cost is

After the arena, sorted by L1 data read miss, the allocator had been **33.9%**
of all misses — `malloc_consolidate` and `unlink_chunk`, glibc walking its own
free lists. No engine function came close; `chooseNext` did not appear at all.

The cause was churn, not footprint: ten games at 1600 searches allocate 527,076
blocks while peaking at 32,978 live, because `moveDown` frees a subtree every
move and the next turn rebuilds one. The arena cut malloc calls 99.5%.

`chooseNext` is now the largest single mispredict source at ~2.1M. Its cost is
the linked-list walk comparing `child_id()` against `move_id()`, and the running
maximum — both genuinely data-dependent. A fix means changing how children are
indexed, which needs the DAG/transposition redesign rather than a local edit.

## Corrections

Four measurement failures, each caught late, and the pattern is worth more than
the numbers.

1. **"74% more nodes."** Taken from a 3-game counter run. At 300 games it is
   −3.8%. The developer objected that a change touching 15% of positions cannot move total
   work by 74%, which was exactly right. The per-allocation argument built on
   top of it was unsound.
2. **A crashed program reported as a 94% win.** The first arena corrupted the
   heap: `Edge::operator new[]` passed the true size, so a short array came from
   the 64-byte class while `delete[]` returned it to the 128-byte one. The
   segfaulting run produced plausible counters. Both arms are now checked for
   identical turn and request counts before any comparison is believed.
3. **`run_suite.sh` mixed workloads.** Instructions came from a 3-game run and
   allocations from a 2-game run, so dividing them compared different runs. All
   counter families now share one workload and per-allocation figures are
   derived automatically.
4. **Every wall-clock measurement for most of the session was noise.** A stray
   benchmark process of mine had been pinning a core for hours. The counters
   were right throughout; the clock was unusable.

## Why single-threaded timing was bimodal, not noisy

Worth recording because it is not obvious and it invalidated a lot of work.

This CPU is heterogeneous: 6 performance cores at 4600–4700 MHz, 8 efficiency
cores at 3500 MHz. The same binary, pinned:

```
  cpu 0  (P-core, 4600 MHz)   0.520 s
  cpu 2  (P-core, 4600 MHz)   0.513 s
  cpu 12 (E-core, 3500 MHz)   0.732 s    +41%
  cpu 16 (E-core, 3500 MHz)   0.726 s    +40%
```

A single-threaded run is at the mercy of which type the scheduler picks, and it
re-rolls whenever the process is descheduled. A stray process does not slow you
by competing for a scarce core — there were 19 free. It slows you by occupying a
P-core, raising the chance your benchmark lands on an E-core.

`run_suite.sh` and `ab.sh` now pin single-threaded runs with `taskset -c 0`, and
record the pinned CPU and the top CPU consumer by name. Spread fell from 21–24%
to 2.3%. Multi-threaded runs are deliberately unpinned.

## Carried forward

From `2026-09-20-04`:

**Done since**

- ~~Bitboard Stages 2 and 3~~ — done, and past the original target.
- ~~Move the `Node` accessors into `node.h`~~ — not done; superseded, since LTO
  already inlines them and the remaining cost moved elsewhere.

**Still live**

- **The spreadsheet's last column is still unidentified.** Unchanged across five
  entries. Still the highest-value unknown in the project.
- **The `vector<bool>` data race in `trainer.cpp`** — a real correctness bug
  under OpenMP, untouched since the first audit.
- **`to_eval_` oversizing**, 448 KB per game against 4.5 KB needed. Gates
  running at scale locally, and now matters more since the arena raised peak
  heap 16.9%.
- **Strength impact of the rules fix** — gen 92 trained on the buggy rules and
  now faces a slightly different legal move space.
- **`web/engine.js` and `web/line_breakers.js`** still carry every line-breaking
  defect.
- **Stub tree shape uncalibrated**, 18.2 turns/game against 28.4 real.
- **Distinct-position instrumentation** not built. Note the starting position is
  only 0.002% of evaluations, not the 5% first estimated — the broader opening
  and transposition sharing is the part that might matter, and is unmeasured.
- **S3** not set up.

## Next steps

1. **The `vector<bool>` race.** It is the only known correctness bug left and it
   is cheap: `std::vector<uint8_t>`.
2. **`to_eval_` oversizing.** Gates everything at scale.
3. **Re-time the branch on a quiet, pinned machine** to confirm 1.83× — the
   figure is good but was taken once.
4. **Transposition table**, as an algorithmic change with its own strength
   measurement, not a C++ optimization. Its payoff is fewer NN evaluations.
