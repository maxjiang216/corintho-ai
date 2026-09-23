# 16 — Child statistics mirrored in the parent: −18% engine time (real network), bit-identical

## Why look here

After entry 15, `chooseNext` was still 60.6% of search time on the stub and
68.0% with the real network. It was measured with `__rdtsc` in a temporary
build, because gprofng sampling returns almost nothing on this machine. That
was 288–331 ticks per call for ~17 children, **~17–19 ticks per child**, and
branch mispredicts in the loop had become small (~0.4M of 2.7M).

Hypothesis: the list walk is latency-bound. Each child's address comes from the
previous child's `next_sibling`, and children are 64-byte nodes created at
different times, scattered through the arena.

`sqrt` (the developer asked) runs once per call, not per child: 0.37% of instructions.
Left alone.

## Microbenchmark first (`data/parent-stats/layout_micro.cpp`)

Scoring ~17 children, linked list of scattered nodes against the same numbers
in contiguous per-parent arrays:

| node pool | list | arrays | |
|---|---|---|---|
| 127 KB | 2.5 ns/child | 1.3 | 1.9× |
| 1 MB | 6.2 | 1.5 | 4.2× |
| 8.5 MB | 23.5 | 1.7 | 14× |

The first run used a 174 MB pool (every child a DRAM miss, 64 ns). It was
discarded as unrealistic: the engine's ~5 ns/child means it mostly hits cache.

**SIMD was not the win.** An explicit two-pass vector version was slower than
the plain scalar loop over arrays, because two divisions per child set the
pace. The gain is the layout.

## Why the stats were in the children

The textbook layout, plus memory: edges were kept at 2 bytes (Leela Zero's
idea) because most are never visited, and two thirds of evaluated nodes never
get a child. Per-edge stats would cost memory on all of them. The design below
pays only on expansion.

## The design: mirror, don't move

- **Source of truth stays in each child.** Every setter that changes visits,
  evaluation, result or `all_visited` calls `syncStats()`, which writes the
  child's copy into its parent's slot. Everything else — `chooseMove*`,
  logging, `propagateTerminal`, Python — is untouched.
- **Parent's block:** `Node *child[cap]`, `float evaluation[cap]`,
  `float visits[cap]`, `uint8_t flags[cap]` (skip; known draw), 17 bytes per
  child. Allocated on first expansion. Capacity grows 3 → 7 → 15 → legal move
  count, filling 64/128/256-byte slots, then up to 1 KB.
- **Where the pointer lives:** the edge array always took a 128-byte slot but
  used at most 96 bytes. It is now `EdgeBlock`: the edges, plus the stats
  pointer, child count and capacity in the old padding. `Node` stays 64 bytes
  (its one free byte, 63, holds the child's slot index; `static_assert` added).
- **Arena:** size classes generalised to 64 … 1024 bytes.
- **`chooseNext`:** reads only the parent's arrays, with the score expression
  unchanged character for character. An `#ifndef NDEBUG` block walks the list
  and asserts every mirrored value equals the child's field.

## Verification: exact, not statistical

- **Engine digest unchanged**: `d7581196f9d30591`, the same as HEAD. Same
  selections, same games.
- Asserts-on runs (mirror == child on every selection, handle/pointer ==
  list node): ~2.5M requests per version, stub and real network, for every
  capacity scheme committed or measured.
- Every A/B arm played identical games (same turns and requests per seed).

## Measurements

Four arms interleaved (`var_stats.py`):
- HEAD;
- C0: capacity 3 → 7 → legal;
- **A: C0 plus a 15 step (committed)**;
- B: 4-byte child handles, capacity 4 → 9 → legal.

Engine time, paired:

| | stub, 30 seeds | real network, 15 seeds |
|---|---|---|
| C0 vs HEAD | −8.9% | −17.7% |
| **A vs HEAD** | **−9.0%** (CI −9.5 .. −8.5) | **−18.3%** (CI −19.9 .. −16.7) |
| B vs HEAD | −9.8% | −19.8% |
| A vs C0 | 0.0% ± 0.9 | −0.7% ± 1.3 |
| B vs C0 | −0.9% ± 1.1 | −2.35% (CI −4.2 .. −0.5) |

Peak RSS, 2000 games × 8 threads (two runs each, agreeing within 1 MB):

| HEAD | C0 | A | B |
|---|---|---|---|
| 466 MB | 584 (+25.4%) | **552 (+18.4%)** | 567 (+21.7%) |

An earlier version with 7-child initial capacity was +32%
(`ab-cap7-*.txt`). One multi-threaded run, 2000 games × 8 threads, gave engine
time 28.7 → 22.4 s (−22%).

The real network gains about twice as much as the stub, plausibly because its
trees are larger and scatter further, which makes the list walk costlier.

## Instructions went UP

Callgrind, identical 3-game workload: 601.8M → 617.3M (**+2.6%**). `syncStats`
is ~20 instructions over ~1M calls, about 25M added in total. The array scan
saves only ~10M, because the list walk never cost many instructions: it cost
*waiting*. Same lesson as entries 13–14: an instruction count is not a time.

Possible follow-up, not done: split `syncStats` into value and flag paths
(`receiveEval` syncs twice per level). Worth perhaps 1–2%.

## Considered and deferred

- **B (4-byte handles).** 17 MB less than C0 and ~2% faster with the real
  network. It needs the arena carved from one `mmap(MAP_NORESERVE)` 256 GB
  reservation, which fails on machines with overcommit disabled. Its memory
  figure is also slightly confounded: mmap pages fault in lazily, where the old
  chunks were zero-filled up front.
- **A + B** (13-byte entries, 4 → 9 → 19 → legal): not measured.

The developer: memory does not matter much here; stop at A.
