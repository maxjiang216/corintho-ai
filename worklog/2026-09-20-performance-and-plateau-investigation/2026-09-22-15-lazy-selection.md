# 15 — Selection scores one unvisited edge: −17% engine time (stub), −12.5% (real network)

## The observation

`TrainMC::chooseNext` was ~43% of instructions (no-LTO profile, entry 13) and
~3.0M of 5.85M branch mispredicts. Every call scored every legal move, 25.8 on
average, 39% of them unvisited.

An unvisited edge scores `prior * c_puct * sqrt(N)`. Within one call only the
prior varies, so the highest-prior unvisited edge beats every other unvisited
edge. The rest were scored and thrown away.

A consequence Max's question drew out: unvisited edges are therefore **always
expanded in descending prior order** (ties by lowest move ID). The visited
children at any moment are exactly the top-k edges by prior. The old code could
not exploit this, because edges and the child list were in move-ID order, which
scattered the top-k through the array.

## The change

- **Invariant:** visited edges are a prefix of the edge array, in expansion
  order. The edge right after them (index k = number of children) is the best
  unvisited edge.
- `chooseNext` walks the k children, scores edge k, and stops. The per-edge
  "is this edge visited?" test (`child_id == move_id`, 1.21M mispredicts, the
  largest single source) is gone.
- `Node::promoteBestEdge(first)` scans `[first, n)` and swaps the best edge
  into `first`. The best edge is the highest integer rank
  `probability << 7 | (127 - move_id)`. Ranks are unique, so this is a plain
  running max, and ties go to the lowest move ID, as the old scan's did.
- It runs on the **first descent** into a node (k = 0), and after each
  expansion for k + 1. It does not run when priors arrive.

Every other loop that pairs children with edges (`match.cpp`,
`selfplayer.cpp`, `printMainLine`, `propagateTerminal`) only needs list order
to equal edge order on the prefix, and that still holds. `getFilteredProbs`
still sees ascending move-ID order, because reordering happens later.

## Equivalence: logically the same search, no pruning

- An instrumented build (not committed) ran the old full scan beside the new
  one on every call: **2 disagreements in 4,597,668 selections**, both exact
  float ties between two scores.
- Asserts-on build (`child_id == move_id(edge_index)` on every child visited):
  ~3.8M requests on stub and real network, no failures. `__assert_fail` has 97
  call sites, so asserts were really compiled in.
- The engine digest changes (`983c2f3d4ef7f5e5` → `d7581196f9d30591`) and is
  reproducible. The game digest is unchanged.

**What does change:** move choice (`chooseMove*`) walks the child list and
breaks ties by list position. Ties now go to the earlier-expanded (higher-prior)
child instead of the lower move ID. `chooseMoveOpening`'s cumulative walk lands
on a different move for the same random number, but the distribution is the
same. Max accepted these.

## First attempt: a full sort was slower (+8.3%)

v0 stable-sorted the edges by prior in `setProbs` (insertion sort; `std::stable_sort`
would allocate every call). Games were identical to the final version (same
digest), but time per request was **+8.32%** (95% CI +7.79 .. +8.86).

The insertion sort alone was **18.4% of instructions**. It ran on every
evaluated leaf, and most leaves never get a child, so the ordering was almost
never used.

## The histogram that shaped the final version

Children per evaluated node at deletion, 20 games (instrumented destructor, not
committed; `data/lazy-selection/children-histogram.txt`):

| children | real network (model_93) | stub |
|---|---|---|
| 0 | 66.4% | 65.8% |
| ≤ 1 | 85.0% | 84.1% |
| ≤ 2 | 90.9% | 90.7% |
| ≤ 10 | 98.1% | 98.2% |
| mean legal moves | 26.2 | 26.0 |

- Full expansion, the O(n²) case of a lazy selection sort, is rare: 2% of nodes
  get 10 or more children. Those nodes are passed through far more times than
  they have edges, and the old code paid O(n) on every pass anyway.
- Two thirds of nodes are never descended into. v1 still did the first scan in
  `setProbs` for all of them: 11.6M of 18.5M compares. v2 moved it to first
  descent.
- **The tree shape is not a stub artifact** (Max's question). The real
  network's distribution matches the stub's closely. It seems to be set by the
  search mechanics (default +1 evaluation, 16 searches per batch), not by how
  peaked the priors are. That cause is a guess; the match itself is measured.

## Measurements

Stub, 40 interleaved pinned seeds × 20 games, time per NN request
(`ab-three-arms-40-seeds.txt`):

| | vs HEAD `eed8ec0` | vs v1 |
|---|---|---|
| v1: skip unvisited edges, first scan in `setProbs` | −12.99% (CI −13.62 .. −12.35) | |
| v2: lazy first scan, rank-based max | **−17.24%** (CI −17.86 .. −16.61) | −4.85% (CI −5.63 .. −4.08) |

v1 and v2 play identical games (same turns and requests on all 40 seeds), so
v1 → v2 is an exact comparison. Callgrind at the identical workload: 655.4M →
601.8M instructions (−8.2%).

Real network, `bench/selfplay_nn` with `model_93.tflite`, 30 interleaved seeds
× 10 games (`ab-real-network-30-seeds.txt`):

| | HEAD | v2 | change |
|---|---|---|---|
| engine ns / request | 1257.6 | 1098.3 | **−12.52%** (CI −14.42 .. −10.63) |
| turns / game | 28.66 | 27.83 | −2.9% (t = −1.94) |
| requests / turn | 1251.6 | 1258.2 | +0.5% (t = +1.04) |

The gain is smaller than on the stub because the engine does more per request
with the real network (1258 against ~820 ns). The other work dilutes the saving.

## A retracted explanation

On the stub, games got longer: +3.2% turns/game pooled over 80 seeds
(t = 2.85). I told Max this was probably the tie-break change in `chooseMove*`
favouring higher-prior moves in lost positions. With the real network the
shift is the other way: −2.9%, t = −1.94. Two borderline results with opposite
signs do not support a directional effect. I did not test the tie-break
explanation, and it should not be repeated as a finding. If game length
matters, measure it with the real network over many more seeds.

## What the rank change does in the machine code (checked afterwards)

Explaining v2 to Max, I said v1's scan re-read `edges_[best]` from memory every
iteration and that v2's update compiles to a conditional move. **Both were
wrong.** I had reasoned from the source. Compiling both loops in isolation
(`data/lazy-selection/scan_codegen.cpp`, `g++ -O3 -march=native -S`) shows:

- **v1 never reloads.** Nothing in the loop writes to `edges_`, so GCC keeps
  the current best's probability and move ID in registers.
- **v2 keeps a branch** (`cmp` + `jnb`), not a `cmov`. That is fine: a new best
  is rare after the first few edges, so the branch predicts well.

What actually differs:

- v1 compares twice: probability, then "equal probability and lower ID". The
  second path is ~12 extra instructions, because GCC rebuilds both fields.
- v2 compares once, on a rank whose high bits are the probability where the
  bitfield already stores it (a mask, not a shift).
- About 9–10 instructions per edge in v2, against 7 on v1's common path and ~20
  on its tie path.

The v1 → v2 gain (−4.85% time, −8.2% instructions) was not split between the
lazy first scan and the rank. The lazy scan removed ~40% of scan calls outright
and is most likely the bulk of it. Max chose not to measure the split.

Lesson: check the generated code before explaining a speedup by what the
source appears to do. The compiler had already removed the "reload".

## Reusable lessons

- **Eager preparation is paid by every node; most nodes are leaves.** Check how
  many nodes ever use the work (the histogram) before putting it on the
  evaluation path.
- **Same digest means same workload.** Two variants that play identical games
  can be compared with exact counters again, which is not possible across a
  change to the random stream (entry 14).
- `bench/selfplay_nn` (commit 47f7794) answers "is this a stub
  artifact?" in minutes. It needs only `pip install ai-edge-litert` in a
  throwaway venv.
