# 25 — syncStats: two designs, the store-to-load trap, and a 2.6% win

This is target 2 from entry 23. `Node::syncStats` was 18.8% of engine CPU at
20 threads, but only 7.5% of instructions. The result is **−2.6% engine time
at 20 threads (−4.0% single-threaded), bit-identical**. That is much less than
the 18.8% suggested. Why is the lesson of this entry. Raw data is in
`data/child-stats-updates/runs.txt`.

## What syncStats did

Entry 16 gave each parent contiguous arrays of its children's selection
inputs (evaluation, visits, flags), so selection does not chase child
pointers. Each child also kept its own `evaluation_`/`visits_`. Every setter
(`increment_visits`, `increase_evaluation`, `set_all_visited`, …) ended with
`syncStats()`. That walked child → parent → `EdgeBlock` → stats, rewrote all
three entries, and recomputed the flags.

One search called it **four times per node on the path**:

- going down: `increment_visits()`, then `increase_evaluation(1.0)` (the
  provisional value that makes the batch's other descents avoid this path);
- coming up in `receiveEval`: `increase_evaluation(eval − 1)`, then
  `set_all_visited(false)`.

## The developer's questions, which shaped the design

- **Why not do all the syncing on the way up?** The down-update cannot wait.
  The 16 descents of a batch run before any network result, and the visits
  plus the +1 placeholder in the parents' arrays are what keep descent #2 off
  descent #1's path (a virtual loss). The up pass is required too, and it
  already synced. The waste was two syncs per level per pass, each rewriting
  everything.
- **What does a node need to know about itself?** Checked every reader of
  `visits()`/`evaluation()`:
  - selection uses the node's own visits as N in c·√N; its parent's array has
    it;
  - root move choice reads the root's children; the root's array has them;
  - root checks, logging, and the debug consistency check.

  A node never reads its own evaluation during search.
- **Visits just increment; nothing needs coordinating.** True. With one copy,
  every update is local arithmetic on the parent's entry.

That led to design **B: the parent's arrays are the only copy** of a non-root
node's evaluation and visits. Updates act on them in place. A node copies its
entry into its own fields when it becomes the root (`null_parent`, which runs
before the old root is deleted). In assert builds the child's fields are kept
as a shadow, and every read asserts that the two agree.

## B was correct but barely faster, and the counters said why

- **Correctness:**
  - golden digests unchanged;
  - `selfplay_nn` sample digests identical on 4 seeds × 300 games (a new
    digest, `5a5616e`: every training sample at 1600 searches with the real
    network, where any change to visit counts moves the policy targets);
  - no shadow assertion fired in 600 assert-build games at 20 threads.
- **Speed:** within noise at first. Callgrind on the stub (3 games), before →
  after:

| | original | B | A′ (below) |
|---|---|---|---|
| instructions | 600.7M | 588.0M (−2.1%) | 587.0M (−2.3%) |
| L1 read misses | 2.109M | **2.259M (+7.1%)** | 2.148M (+1.9%) |
| L1 write misses | 1.593M | **1.423M (−10.7%)** | 1.533M (−3.7%) |

**B converted store misses into load misses.**

- The original backup did `evaluation_ += d` on the child's own field (whose
  line is loaded anyway, to follow `parent_`), then **stored** the result
  blindly into the parent's array.
- A store miss is mostly absorbed by the store buffer. B's `parent_entry += d`
  is a read-modify-write, so the cold line must be **loaded** first, and a
  load miss stalls.

The "duplicate" child copy was quietly doing useful work: a cheap write-only
mirror on the cold backup path.

## A′: keep the child copy, store only what changed (committed)

- Every update changes the child's own field, then stores just that field
  into the parent's entry: visits, evaluation, or both for the descent's
  combined `add_visit`. It never reads the parent's entry, and there is no
  full rewrite and no flag recomputation.
- The flags are written only when `result_` or `all_visited_` actually changes.
  `set_all_visited(false)` in the backup returns early when the flag is already
  clear, which is almost always.
- Down: `increment_visits` + `increase_evaluation(1.0)` → one `add_visit(1.0)`.
- Accessors read the child's fields, as before, so there is no special case
  for the root or for a new root.
- The parent's entries always equal the child's fields, as in the original,
  so this is bit-identical by construction. It was checked anyway:
  - sample digests identical on all 4 seeds;
  - golden unchanged;
  - assert build (with the existing mirror-vs-child check in `chooseNext`)
    clean, with the same digest.

## Timing (three arms interleaved, identical turns in every pair)

| | real network, 20 threads, 6 × 1,000 games | stub, 1 thread pinned, 12 × 200 games |
|---|---|---|
| B vs original | −2.5% (CI −2.9 to −2.0) | −2.3% (CI −3.2 to −1.4) |
| **A′ vs original** | **−2.6% (CI −2.9 to −2.3)** | **−4.0% (CI −5.4 to −2.5)** |
| A′ vs B | −0.1% (CI −0.8 to +0.5) | −1.7% (CI −3.0 to −0.3) |

Early two-arm runs of B, with the browser and the mail client active, had
confidence intervals of ±4–10%. Only the later three-arm runs resolve these
effects.

## Why 18.8% of CPU became a 2.6% gain

gprofng charged `syncStats` with the latency of the **first touch of the
parent's statistics line in backup**. That touch is unavoidable in any
design, because the parent's evaluation must change. What was removable:

- the second sync per level;
- rewriting unchanged fields;
- the flag recomputation.

Those are instructions on lines already in cache, and they are what the 2.6%
is. The miss itself moved, and in B it got worse, because it became a load.

**Lessons**

- A profiler's time for a function that takes a cache miss is mostly the
  miss. Before estimating a gain, ask whether the miss is avoidable or only
  moves.
- Read-modify-write on a cold line costs more than a blind store to it. A
  "redundant" copy on a hot line can be what lets you store blindly.
- The remaining lever here is the miss itself: fewer cold lines per level in
  backup (the parent node, its `EdgeBlock`, the stats line). For example,
  have the child cache a pointer to its parent's stats, or co-locate the
  stats with the `EdgeBlock`. Not attempted.
