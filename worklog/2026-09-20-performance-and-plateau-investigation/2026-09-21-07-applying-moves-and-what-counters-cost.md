# Applying moves, and what each counter is actually worth

*2026-09-21 — branch `perf/engine-optimizations`, commits `5bb695d`, `7645d5f`*

## Why this entry exists

Every optimization so far targeted *finding* legal moves. The obvious next
question is *applying* them: `Game::doMove`. This entry answers that, and then
answers a more useful question it raised — when a profile reports instructions,
memory reads, cache misses and branch mispredicts, what does each one cost?

Short version: `doMove` was worth 0.244% and I got essentially all of it. The
cost model is the part worth keeping.

## Part 1 — doMove

### How often it runs

Exactly once per new node. There is one call site in the whole engine:

```
corintho_ai/cpp/src/node.cpp:36:  game_.doMove(move_id);
```

inside `Node::Node(game, parent, next_sibling, move_id, depth)`. Each node
stores a full `Game`, so a child is built by copying the parent's board and
applying one move. There is no make/unmake and nothing replays moves, so
`doMove` runs once per node created and never again.

That bounds the prize before any work starts. From cachegrind on a fixed
3-game run:

```
Game::doMove            2,360,980 Ir    0.23% of the program
Node::initializeEdges   ~118,745,223 Ir 11.6% (inclusive)
```

Deleting `doMove` entirely would save under a quarter of one percent.

### Fixing the benchmark first

`bench/micro.cpp` already had a `doMove` benchmark. It was biased and had to be
fixed before its numbers meant anything (commit `5bb695d`):

- It took the **lowest-indexed** legal move. IDs 0-47 are all move-moves and
  48-95 are all places, so it picked a move-move whenever one existed — the
  expensive branch, and the same move every repetition, so perfectly predicted.
- It folded the `Game` copy into the figure without ever measuring the copy.

Corrected: a uniformly chosen legal move, place and move-move drawn from
separate pools and reported separately, copy timed alone. The copy is **0.5 ns**,
so it was never much of the old number — but the branch mix was. The old
"7.2 ns" was essentially the move-move figure.

**`ns_game_domove_incl_copy` keeps its name but is not the same measurement.
Do not compare it across `5bb695d`.**

### What was actually slow

Read the disassembly before changing anything. Two things stood out, and one
expected thing turned out to be a non-issue:

```asm
call   _ZN4MoveC1Ei@PLT                          ; Move ctor, out of line
movabsq $8608480567731124087, %rcx               ; = 0x7777777777777777
andq   (%rbx), %rcx                              ; the frozen reset, already one AND
```

- **The sixteen-iteration frozen-reset loop was already gone.** GCC had reduced
  `for row { for col { set_frozen(false) } }` to a single `andq`. The most
  obviously wasteful-looking code in the function was costing nothing. This is
  the entry's main methodological point: the source-level shape of a loop tells
  you very little after `-O3`.
- **`Move`'s constructor was a PLT call.** It lives in `move.cpp`, so without
  LTO the call could not inline — a function call for what is one array read.
- **The move-move branch looped over the three piece types**, reading source and
  destination and OR-ing per type, which unrolled into a chain of
  test/or/andn/cmove.

### The change

The board is four bits per space — three piece bits then a frozen bit — packed
into one `uint64_t`, so space `s` occupies bits `[4s, 4s+4)`. The whole
operation is a few word ops:

```cpp
const MoveInfo &move = kMoveTable[move_id];      // no PLT call
uint64_t b = board_.to_ullong() & kUnfrozenMask; // clear all frozen bits
const int32_t to_shift = move.to * 4;
if (move.is_place) {
  --pieces_[to_play_ * 3 + move.piece];
  b |= UINT64_C(1) << (to_shift + move.piece);
} else {
  const int32_t from_shift = move.from * 4;
  const uint64_t stack = (b >> from_shift) & kStackMask;  // lift the nibble
  b &= ~(kStackMask << from_shift);                       // clear the source
  b |= stack << to_shift;                                 // deposit it
}
b |= UINT64_C(1) << (to_shift + kFrozen);        // freeze; both branches did this
```

`kStackMask` and `kUnfrozenMask` are derived by `constexpr`, not transcribed —
a hand-written constant table is how `line_breakers` acquired thirteen errors
(entry 04).

### Results

Micro, interleaved and pinned to a P-core, 7 reps each arm, copy subtracted
(`data/counter-cost-model/domove-micro-ab.txt`):

| | old | new | delta |
|---|---|---|---|
| doMove move-move | 7.4 ns | 4.7 ns | **-36%** |
| doMove place | 5.4 ns | 4.8 ns | -11% |
| doMove uniform mix | 6.0 ns | 5.2 ns | -13% |

End to end, exact instructions over a fixed 3-game run:

```
1,027,085,956 -> 1,024,579,944    -0.244%
```

The micro and the counter agree rather than independently confirming each
other: ~2.5M fewer instructions over ~86k calls is ~29 instructions per call,
which is what the rewrite removes. Against `doMove`'s 0.23% program share,
-0.244% is close to the ceiling.

**Wall clock is deliberately not quoted.** A browser was active; the arms' own
spreads were 13-16%, far wider than the effect.

### A trap this run exposed

The micro also reported `getLegalMoves` improving 140.5 -> 127.4 ns (-9%),
with tight non-overlapping ranges. That change is not real. A genuine 9% cut in
`getLegalMoves` would have moved total instructions far more than 0.244%, and
the function was not touched. It was code layout — alignment shifting under an
unrelated edit.

Tight spreads are not evidence of a real effect. Only the exact counter settled
it. Any micro delta with no instruction-count change behind it should be assumed
to be layout until proven otherwise.

## Part 2 — what each counter costs

### Measuring the clock

`/sys/.../scaling_cur_freq` reports **400 MHz while the core is at full load**
on this chip — stale HWP reporting on Alder Lake. It is unusable for converting
counts to time.

Instead, `data/counter-cost-model/effective_clock.c`: a loop-carried dependent
add chain retires exactly one per cycle, so wall time gives the clock directly.

```
effective 4.49 GHz  (0.445 s for 2000000000 dependent adds)
```

Stable across three runs. Use this, not cpufreq, and re-measure it rather than
assuming a nominal figure — it was taken under browser load, so it is a lower
bound.

### The model

3-game run, 0.0862 s (min of 7), 4.49 GHz -> 387M cycles, 1,024,579,944
instructions, **IPC 2.65** against Golden Cove's 6-wide issue: the core is idle
in ~56% of its issue slots.

| event | count | cyc/event | cycles | share | exposed? |
|---|---|---|---|---|---|
| branch mispredict | 7,338,468 | ~17 | 125M | 32% | **fully** — pipeline flush |
| L1 miss -> L2/L3 | 5,485,986 | ~14 | 77M | 20% | mostly hidden by OoO |
| LL miss -> DRAM | 51,313 | ~250 | 13M | 3% | partly hidden |
| instruction issue | 1,024,579,944 | 1/6 | 171M | 44% | throughput floor |

**These rows must not be added.** They total ~99%, which looks like a clean
decomposition and is not one: an out-of-order core overlaps them, so each row
is an upper bound on its own contribution, not a share of a partition. The true
split needs hardware counters, and `perf_event_paranoid=4` blocks those here.

What is trustworthy is cost per event and, more importantly, **exposure**:

- **Branch mispredict: ~17 cycles, all of them paid.** A flush discards
  everything in flight. There is nothing to overlap it with, because the
  speculative work is exactly what gets thrown away.
- **L1 miss hitting L2: ~14 cycles, almost none paid.** The reorder buffer is
  ~512 entries; the core executes past it. This is why raw cache-miss counts are
  the most over-read number in a profile.
- **DRAM miss: ~250 cycles, but only 51,313 of them** — 0.029% of accesses.

### Why branch mispredicts dominate in *this* program

Not a universal law — a property of this workload:

```
working set       a Game is 16 bytes, a Node is 64    -> effectively L1-resident
LL miss rate      0.029% of accesses                  -> cache nearly free
branch density    11.1% of instructions
mispredict rate   6.56% of conditional branches       -> bad
```

The old move generator was near worst-case for a predictor: test 96 moves one
at a time on data-dependent conditions, scan 34 line shapes. Mask arithmetic
does not predict those branches better — it removes them.

### The rule for choosing a metric

**Instructions retired is the right primary number**: deterministic, unaffected
by whatever else is running. But it systematically *undervalues* removing an
unpredictable branch (1 instruction, 17 cycles) and *overvalues* removing
predictable ones. Read Ir and Bcm together — which is what `run_suite.sh`
records.

Evidence from this branch: the line-breaking rewrite cut instructions 35.8% but
mispredicts 65.8%, and the wall-clock speedup tracked the mispredict figure.

## Next

`Game::writeGameState` — **88.6 ns**, untouched, ~4.4% of program instructions,
19x `doMove`. It is a 64-iteration scalar loop testing one bit and storing one
float, which is a textbook bit-to-float expansion. Called once per node that
requests an evaluation, which is most nodes.

Carried forward unchanged: `to_eval_` oversizing; `build.sh` never verified with
`-flto`; strength impact of the line-breaking fix; `web/engine.js` defects; and
**the spreadsheet's last column**, open since entry 01.
