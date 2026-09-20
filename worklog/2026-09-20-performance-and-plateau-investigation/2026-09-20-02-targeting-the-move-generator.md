# 2026-09-20 — Targeting the move generator

Branch: `perf/training-overhaul`. Design only; no engine code changed yet.

Applies the measurement framework to choosing and planning the first real
optimization. The point of writing this before implementing is that the
predictions below are **falsifiable**: if the measured result misses them, our
model of the code was wrong, and that is worth knowing separately from whether
the change was a win.

## TL;DR

- **Highest impact per unit of effort is not an engine change at all:** adding
  `-flto` to `corintho_ai/python/setup.py` is one line, zero risk, and worth a
  measured **18%** in production. Our harness cannot show it as an improvement
  because the harness already builds with LTO.
- **Highest impact engine change is the bitboard `getLegalMoves` rewrite.**
- `getLegalMoves` costs **7,962 instructions per call** and is **26.9% of all
  program instructions**.
- Within it, the 96-move filter loop outweighs line detection **6.5 : 1**, so
  the filter loop is the first target and line detection can wait.
- `Move::Move(int)` alone is **6.87% of the whole program** — the single largest
  function — and is constructed **83.6 times per `getLegalMoves` call** purely to
  be thrown away.

## Decomposing the target

Measured on the no-LTO build, which keeps functions separate enough to attribute
(`build-lines/`, `-O2 -g`), 3 games × 1600 searches, callgrind:

| Component | Calls | Per `getLegalMoves` call | Instructions | % of program |
|---|---|---|---|---|
| `Move::Move(int)` | 4,468,909 | 83.6 | 108,613,055 | 6.87% |
| `Game::canMove` | 2,259,364 | 42.3 | 106,193,649 | 6.71% |
| `Game::canPlace` | 2,209,545 | 41.3 | 80,155,580 | 5.07% |
| `getLegalMoves` itself | 53,452 | 1 | 73,500,848 | 4.65% |
| `Game::applyLines` | 53,452 | 1 | 57,148,634 | 3.61% |
| **Total** | | | **425,611,766** | **26.9%** |

`Game::empty` is called 5,426,762 times, inside `canPlace`/`canMove`.

**Filter loop** (`Move::Move` + `canPlace` + `canMove` + self) = 368.5M =
**86.6%** of `getLegalMoves`.
**Line detection** (`applyLines`) = 57.1M = **13.4%**.

Two useful details fall out:

- `Move::Move` is called 83.6 times per call, not 96, because `applyLines` runs
  first and zeroes some bits, so `legal_moves[i]` is already false and
  `isLegalMove(i)` is skipped. The filter is already partially short-circuited
  and it is *still* the dominant cost.
- **7,962 instructions to decide the legality of at most 96 moves on a 4×4
  board** — roughly 83 instructions per move considered. The bitboard derivation
  in `PLAN.md` §13 needs about 20 bitwise operations plus `ctz` emission for the
  same answer.

## Why this target, over the alternatives

| Candidate | Evidence | Verdict |
|---|---|---|
| **Bitboard `getLegalMoves`** | 26.9% of instructions; 25–27% of time; ~80% of node construction cost; 60% of all branch mispredicts | **Chosen** |
| `-flto` in `setup.py` | 18% measured, one line, zero risk | **Do it anyway**, separately — it is not an engine change and the harness cannot measure it |
| Arena allocator | 47% of L1 read misses are in the search, 1.1e9 allocations/generation | Later. Cache evidence is a lower bound from a 3-game working set and needs confirming at scale first |
| `Node` accessors to header | 10.7% of instructions without LTO | Subsumed by `-flto` for production; do it anyway for robustness |
| `getFilteredProbs` | 6.1% of instructions, 7.7% of mispredicts | Good, small, safe. Queue behind the big one |
| Root-only Dirichlet | 3.9% plus a likely strength gain | Separate concern; it deliberately changes the engine digest |

Four independent lines of evidence agree on the move generator, which is the
strongest case available. Notably the evidence arrived in this order:
instruction count, then wall-clock time, then node-construction microbenchmark,
then branch mispredicts — and each one *raised* the estimate.

## Staging

The rewrite is large enough that one commit would be unreviewable and would
confound the measurements. Three stages, each independently committable,
measurable, and revertable:

**Stage 1 — cache per-space state.** Compute `{top, bottom, empty, frozen}` once
per position into a 16-entry array; have `canPlace`/`canMove`/`applyLines` read
it. Keeps the existing control flow entirely. Attacks the 5.4M `empty` calls and
the repeated `top`/`bottom` scans. Low risk.

**Stage 2 — emit instead of filter.** Replace the 96-iteration loop with the
three place-mask expressions and eight shift-AND move-mask operations from
`PLAN.md` §13, emitting move IDs with `__builtin_ctz`. This is where
`Move::Move`'s 6.87% disappears entirely. Highest risk and highest reward.

**Stage 3 — bit-plane board representation.** Replace the interleaved
`bitset<64>` with four `uint16_t` planes. Makes stages 1–2 natural rather than
bolted on, shrinks `Game`, and speeds `doMove` and `writeGameState`. Touches
`Node` layout, so it goes last.

Line detection (13.4%) is deliberately left for a fourth stage. It is a fifth of
the prize and it carries the fiddly capital-line special case at
`game.cpp:252-278`.

## How each stage is measured

Per `bench/README.md`, but specifically:

**Before implementing:** `./run_suite.sh before-stageN all 5` on a clean tree.

**Targeting — already done, not repeated per stage.** The `lines` tier
(`-O2 -g`, no LTO) localized the mispredicts to specific statements. Re-run only
if a stage lands somewhere unexpected.

**Quantifying — the exact counters do the real work here.** `exact_instructions`
and `exact_branch_mispredicts` are deterministic, so a 3% move is unambiguous,
whereas the same 3% is invisible in wall clock. Expect them to move first and
most cleanly.

**Confirming — `st_engine_seconds` and `big_engine_seconds`.** Single-threaded is
the honest number for this change: it is pure per-thread algorithmic work and
should improve identically at one thread and fourteen. If `mt` improves markedly
*less* than `st`, that means we have shifted the bottleneck to memory bandwidth
and the next target changes.

**Behaviour — `digest_game` must be byte-identical.** This is a pure
optimization; any digest movement is a bug, full stop. `digest_engine` must also
hold, since search consumes legal moves in edge order.

## Predictions

Recorded before implementation so they can be wrong. Baseline is
`bench/results/baseline.tsv` at commit `d562726`.

| Metric | Baseline | Predicted after all 3 stages | Reasoning |
|---|---|---|---|
| `getLegalMoves` instructions/call | 7,962 | **400 – 800** | ~20 mask ops + `ctz` emission + unchanged line detection (~1,070/call) |
| `exact_instructions` | 962,920,578 | **680M – 730M** | Removing ~24% of program instructions |
| `exact_branch_mispredicts` | 11,788,588 | **5.0M – 6.5M** | Move gen owns ~60%; branchless masks should remove nearly all of its share |
| `exact_branches` | 127,525,946 | **85M – 100M** | The 96-iteration loop disappears |
| `ns_game_getlegalmoves` | 928.5 | **60 – 150** | 6–15× |
| `ns_node_ctor_dtor_alloc_heavy` | 1,153.5 | **250 – 400** | Node cost is ~80% `getLegalMoves`; the ~225 ns allocation floor remains |
| `st_engine_seconds` | 4.089 | **2.6 – 3.2** | Amdahl on instructions gives ~1.34×; mispredict removal should push it further, since those cycles are disproportionately expensive |
| `speedup_14t` | 7.56 | **unchanged or slightly lower** | Less work per thread with the same memory traffic means a marginally worse parallel ratio — which would be fine |
| `digest_game` | `0f82fc99147482e4` | **identical** | Non-negotiable |

The interesting prediction is that **time should improve proportionally more
than instructions**. Instructions say ~1.34×; the mispredict removal is worth an
estimated 120M cycles on its own. If time improves only as much as instructions,
our model of why this code is slow is wrong.

## Risk controls

A move-generation bug corrupts every downstream result silently, so the digest
alone is not enough — it detects a divergence but does not localize it.

**Keep the old implementation.** Rename it `getLegalMovesReference`, retain it
compiled, and add a `verify` mode to the bench that runs both over a large
random corpus and, on the first mismatch, prints the board, both bitsets, and
their XOR. Delete the reference only once all stages are done and verified.

This is cheap and it is the difference between "the digest changed, somewhere"
and "position 14,332 disagrees on move 57, which is a capital placement on a
frozen space".

## Carried forward

From `2026-09-20-01`:

**Done since**

- ~~Add single- and multi-threaded profiling~~ — done; `st`/`mt`/`big` in
  `run_suite.sh`, opt-in `mt` profiling tier.
- ~~Decide which profilers to use~~ — done; gprofng adopted, tiers documented,
  rejected tools recorded with reasons.
- ~~Set up a worklog~~ — done.

**Still live**

- **The spreadsheet's last column is still unidentified.** This continues to be
  the single highest-value unknown in the project: if it is the second-player
  self-play score, colour imbalance outranks all performance work.
- **The stub evaluator's tree shape is uncalibrated** (18.2 turns/game vs 28.4
  real). Does not block this optimization — `getLegalMoves` cost per call is
  shape-independent, and the *relative* prize would only grow with deeper trees —
  but it does affect the headline engine percentages.
- **`to_eval_` oversizing** still gates running anything at realistic scale
  locally.
- **Distinct-position instrumentation** not yet built.
- **S3 not set up.**

## Next steps

1. Add `-flto` to `setup.py` and move the `Node` accessors into `node.h`. Small,
   independent, and worth 18% in production.
2. Build the `verify` mode and the `getLegalMovesReference` safety net.
3. Stage 1, measure, commit. Then stage 2, then stage 3.
4. Revisit the arena allocator with cachegrind at a larger working set, once
   move generation is no longer masking the search's cache behaviour.
