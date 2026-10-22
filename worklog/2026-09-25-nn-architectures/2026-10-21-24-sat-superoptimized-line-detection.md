# 24 — SAT superoptimization of line detection

2026-10-21. `bench/superopt/` (new), `corintho_ai/cpp/src/game.cpp`
(`boardHasLine`, `findLines`, new `topNibbles`).

## TL;DR

- A small z3-based superoptimizer (`bench/superopt/`): CEGIS synthesis
  of shortest straight-line programs with a minimality proof, equivalence
  proofs over all 2^64 board words, and sketch queries.
- Sketch query: K = 3 masked run terms cannot detect lines (UNSAT); at
  K = 4 the solver rediscovered the four hand-written directions.
- Line detection moved into the board's own nibble layout. All three
  piece types are tested at once, with no `pext`. Proven equal to the old
  code for every board word.
- The top-of-stack kernel is the synthesizer's: 5 instructions, proven
  minimal (no 4-instruction program exists). My hand-written version had 6.
- Position key, `packRest` and the reserve sums: loops replaced with
  `pext` and multiply tricks, all proven. Together with the line change,
  the solver at P <= 24 goes 2.973G -> 2.622G instructions (-11.8%), and
  P <= 27 wall time 6.12 -> 5.80 s (-5%).
- `lineBreakers`: the per-move extend loop becomes one AND with a
  lazily built "moves from tops of this type" word. Cumulative: 2.575G
  instructions (-13.4%), P <= 27 wall time -8%.
- Incremental line check (`Game::lineMakingMoves`): which moves make a
  line, for all moves at once, proven for all 96 moves under the
  placement and stacking rules. Solver and `ab_match` use it where no line
  stands. Solver P <= 27 wall time against main: 6.25 -> 5.38 s (-14%).
- Line detection alone at P <= 24: 2.973G -> 2.797G instructions (-5.9%), identical
  results and node counts. `getLegalMoves` micro: 53-57 -> 47 ns (-12 to
  -17%). Solver wall time at P <= 27 is within noise.

## Why

The developer asked whether SAT brute forcing could speed up legal-move
generation or line checks, the hot paths of the alpha-beta solver.
Callgrind, solver at P <= 24, no-inline build (5.31G instructions):

| function | share |
|---|---|
| `findLines` | 12.0% |
| `boardHasLine` (incl. its lambda) | 12.8% |
| `basicLegalMoves` | 7.1% |
| `lineBreakers` | 5.5% |
| `computeSpaceInfo` | 2.1% |

So line checking took about a quarter of the solver's instructions.

## The idea: stay in the board's layout

The board word has four bits per space: base, column, capital, frozen.
Both line detectors first gathered each piece type into a 16-bit plane
with `pext`: three for `boardHasLine`, four for `computeSpaceInfo` and
then `findLines`. After that they derived the top planes and ran one
shifted AND per direction and type.

A space's top is its highest piece, which is a bitwise function of its
nibble. So all three top planes fit, interleaved, in one 64-bit word `t`.
Bit k of a space's nibble is set when that space's top is type k. A run
of equal tops in direction d is still `t & t >> s & t >> 2s`, with the
stride now 4 bits per space (s = 4, 16, 20, 12). Every shift is a whole
number of spaces, so bit k of a nibble only ever meets bit k of another
nibble and the types never mix. The run-start masks spread to `7` per
chosen nibble (`kRunStartNibbles`). `findLines` decodes a set bit as
start space `bit >> 2` and type `bit & 3`.

`findLines` no longer needs a `SpaceInfo`. Its lines come out in a
different order (direction-major instead of type-major). Both consumers
are order-independent: `lineBreakers` ANDs and `lineSpaces` ORs.

Entry 11 tried putting the three types in one word with 16-bit lanes and
reverted it, because the per-type loop skips types with no tops. This
version is different: it needs no gather at all, and it still exits early
when no run exists.

## Synthesis: topNibbles

`python3 bench/superopt/synth_tops.py`. The spec, per nibble, is base &
~column & ~capital, column & ~capital, capital. Bit 3 is don't-care
because every caller masks it. Instruction set: and, or, xor, andn, not,
shifts by a constant, and/or/andn with a 64-bit constant (constants are
loop-invariant, so they count as free).

```
n=1: unsat (0.0s)   n=2: unsat (0.2s)   n=3: unsat (1.8s)
n=4: unsat (10.2s)  n=5: found (163.3s, 8 examples)
  c = (b & 0x6666666666666666) >> 1
  t = b & ~c & ~(c >> 1)
```

`c` moves each nibble's column and capital bits down one place, and
`c >> 1` moves the capital onto base. Bits that cross into the next space
land only on bit 3. My first version, `u = b & 0x7777...; u & ~(u >> 1 |
u >> 2 & M1)`, used 6 instructions. I had written a comment claiming it
was minimal; the synthesizer disproved that.

With bit 3 required to be zero as well (`synth_tops.py --clean-bit3`),
n = 5 is UNSAT (375 s) and n = 6 timed out at 600 s. The hand-written
version above is clean and has 6 instructions, so 6 is the minimum
there. Leaving junk in bit 3, which nothing reads, saves exactly one
instruction.

## Proofs

`python3 bench/superopt/prove.py` (each under 0.1 s):

```
PROVED  tops_nibbles == top_plane[0..2]
PROVED  has_line_nibbles == boardHasLine
PROVED  runs_nibbles == findLines run3/run4
```

The models in `kernels.py` transcribe the C++, with `pext` modelled bit
by bit. The proofs are only as good as that transcription, so the usual
gates ran as well.

## Gates

- `bench/verify`: PASS, 200,000 positions identical to the reference.
- `bench/rulecheck`: PASS.
- `golden game`: `d8f9bd2d22ee9731`, unchanged.
- `solver_bench` P <= 24 and P <= 27 against HEAD (`solver_gate.py`): 0
  disagreements, nodes 2,197,120 and 29,771,409 on both sides.

## Measurements

Container: 4-core Intel Xeon @ 2.80 GHz, clang++ 18.1.3 (clang-20 is not
installed here), solver_bench built from `bench/` with the Makefile's
flags. Timings ran next to the background SAT jobs, so wall times are
noisy.

| measure | before | after |
|---|---|---|
| callgrind, solver P <= 24, LTO, `-march=x86-64-v3` | 2,973,011,468 | 2,796,960,633 (-5.9%) |
| `micro_bench` `Game::getLegalMoves` (2 runs, pinned) | 56.8, 53.1 ns | 46.9, 47.0 ns |
| solver P <= 27 wall, 3 interleaved rounds, pinned | 6.17 / 6.39 / 5.81 s | 6.20 / 6.07 / 5.98 s |

Callgrind needs `-march=x86-64-v3`. Valgrind cannot run this machine's
`-march=native` (AVX-512) build; it stops after 12M instructions.

The instruction savings barely show in solver wall time. `pext` runs at
1 per cycle on Intel, so the gathers it removed were cheap in cycles even
though they were not cheap in instructions. On AMD before Zen 3, where
`pext` is microcoded, the gain should be larger, but I have not measured
it. In the network alpha-beta engine (`ab_match`), evaluation dominates a
node's time (entries 22-23), so the effect there is smaller again.

## Sketch query: fewer direction terms?

`python3 bench/superopt/sketch.py 4 3` fixes the shape `OR_k (t & t >> a_k
& t >> b_k & C_k)`, leaves every shift and mask symbolic, and asks
whether K = 3 terms can replace the four directions.

```
K=4: sat after 130 examples, 400.0s
  t & t >> 4  & t >> 8  & 0x0077007700770077   rows
  t & t >> 12 & t >> 24 & 0x0000000077007700   d4-a1 diagonals
  t & t >> 16 & t >> 32 & 0x0000000077777777   columns
  t & t >> 20 & t >> 40 & 0x0000000000770077   a4-d1 diagonals
K=3: unsat after 63 examples, 21.3s
```

Given only "a line exists" as the spec, the K = 4 search rediscovered
exactly the hand-written directions and edge masks (`kRunStartNibbles`).
K = 3 is UNSAT, so no choice of shifts and masks lets three such terms
work: four is minimal for this shape. A shortcut, if one exists, needs a
differently shaped program (shared intermediate terms, say), which is
beyond what exact synthesis can search at this size (~25 instructions).

## Position key and reserve sums

Second round, same method. Every node and every child (the prefetch loop)
calls `Game::key`, which packed the six reserve counts with a six-step
loop. `packRest` repacked them with another loop, and `store` and
`recordCutoff` added them up with a third. `Game::horizon` had the same
sum over the raw bytes.

| routine | was | now |
|---|---|---|
| `Game::key` | 6-step loop | one 6-byte load, one `pext` (portable: three shift-or-mask steps) |
| `packRest` | 6-step loop | one `pext` with mask `0x7777771` |
| reserve sum (`restReserves`) | 6-step loop | add odd and even nibbles pairwise into bytes, one multiply, 7 operations |
| `Game::horizon` reserves | 6-step loop | one load, one multiply by `0x010101010101` |

Each is proven equal to its loop for every reserve count 0-4
(`prove.py`). The reserve sum was also checked exhaustively in Python
over all 31,250 valid `rest` words. The multiply tricks work because the
counts are at most 4: no byte sum exceeds 24, so nothing carries.
`synth_reserves.py` extends the synthesizer with add, sub and
multiply-by-constant and a precondition on the input; lengths 1 and 2 are
UNSAT (67 s for length 2). At length 3 the query timed out after 600 s
with no answer, so lengths 3-6 are open. With symbolic constants, the
64-bit multiplier bit-blasts into a large SAT problem.

Gates: `verify` PASS, `rulecheck` PASS, `golden` game `d8f9bd2d22ee9731`
and engine `2fb48873bf620b61` (both unchanged), and `solver_bench` P <= 24
and P <= 27 with identical results and node counts.

| measure | main | line detection | + key and reserves |
|---|---|---|---|
| callgrind, solver P <= 24 | 2,973M | 2,797M | 2,622M (-11.8% total) |
| solver P <= 27 wall, 3 interleaved rounds | 6.13 / 6.20 / 6.03 s | 6.01 / 5.83 / 6.00 s | 5.66 / 5.77 / 5.97 s |

Mean wall time is 6.12 -> 5.80 s (-5%). A synthesis job was running on
another core, so the noise is still about +-0.15 s.

## Line breaking by extension

`lineBreakers` walked each line's extending moves (up to four) with a
`ctz`, a `kMoveTable` load and a shift test each. A per-shape `pext` /
`pdep` pair cannot replace that walk. For extend square e the sources in
board order are e-4, e-1, e+1, e+4, but their move IDs come in the order
e-1, e-4, e+1, e+4, so a gather-then-deposit would swap the first two.

What works: `movesFrom(top_plane[type])` gives every stack move whose
source is topped by that type, laid out by move ID. It costs two `pext`,
and down and up are plain shifts. Each line then needs one AND with the
shape's extending moves. The word is built lazily, once per type: a four
and its two threes share it. `prove.py` shows that bit `id` of
`movesFrom(tops)` is the tops bit at that move's source for all 48 stack
moves, and that the portable non-BMI2 path (each row shifted down by its
index) equals the `pext` one. The Python move layout was also checked
against a dump of the C++ `kMoveTable`.

Gates as above: identical, including on a `-march=x86-64` build (no
BMI2), which exercises the portable paths of `key` and `movesFrom`.

| measure | main | + key and reserves | + lineBreakers |
|---|---|---|---|
| callgrind, solver P <= 24 | 2,973M | 2,622M | 2,575M (-13.4% total) |
| solver P <= 27 wall, 3 interleaved rounds | 5.92 / 6.30 / 6.44 s | 5.64 / 6.08 / 5.97 s | 5.55 / 5.73 / 5.90 s |

Mean wall time is 6.22 -> 5.73 s (-8%) against main. Only positions with
a line run this code, so its share was always going to be small.

## Incremental line check: every child at once

The solver's classification loop made each child (`doMove`) and asked
`hasLine()` of it, to find the line-making children. `ab_match`'s move
ordering does the same with a hand-built child board word (entry 23).

`Game::lineMakingMoves()` answers for every move of a position in one
branch-free pass, when no line stands. The idea:

- With no line standing, a line after a move must run through the space
  the move lands on. That is the only space whose top changes: a stack
  move's source empties, and an empty space is in no line.
- So for each space and type: would topping this space with this type
  complete a run of three? The space is a run's first, middle or last
  cell, and the other two must already carry that top. That is three
  shifted ANDs per direction, for all types at once in the nibble layout.
- A placement of type k on x makes a line iff bit k of x's nibble is set.
  One `pext` per type drops all 16 placements into their move IDs.
- A stack keeps its top and its source empties. In the move's own
  direction, only the runs through the destination that avoid the source
  count: for a right move, the destination must be a run's first cell;
  for a left move, its last; likewise down and up with columns. The
  arriving top is the tops word shifted from source to destination. A
  hit has at most one bit per nibble, so adding 7 carries it into bit 3,
  and one `pext` per direction gathers the hits into move IDs.

It only applies where no line stands, about 85% of positions. Otherwise
each child is still asked.

### Proof, and what it rests on

`prove.py` checks each of the 96 moves: for every board word with no line
on which the move passes the placement and stacking rules (canPlace /
canMove, reserves aside), bit m equals `hasLine` of the board after
`doMove`. The legality condition is needed. It is what makes the
destination's new top the placed piece or the moving stack's top; an
arbitrary OR of two stacks need not have that top.

The proof is not vacuous. The precondition is satisfiable, and a mutant
that counts every row run for right moves (ignoring the emptied source)
fails on all 12 right moves. The first version of the proof "failed" on
every stack move for a different reason: `top`/`bottom` returned
integers, which takes the query outside pure bit-vector logic, so the
`QF_BV` solver answered "unknown". They are bit-vectors now.

`bench/verify` now also compares `lineMakingMoves` against `doMove` +
`hasLine` for every legal move of every line-free corpus position:
5,129,926 moves, 448,039 of which make a line, no mismatches. That also
checks the model's legality assumption against the real generator. It
passes on the native and the non-BMI2 build.

### Cost: an inlining trap

The first measurement was 2.575G -> 2.865G instructions (+11%). In
isolation the batch is cheap: 6.4 ns per position, against 80-115 ns for
`doMove` + `hasLine` over its ~15 legal moves (a microbenchmark at
P <= 30). The function-level profile showed the cause. The per-child
classification lambda had stopped being inlined into `Solver::search`,
so it became a call per child that reloaded every captured variable.
Keeping `lineMakingMoves` out of line did not bring it back; marking the
lambda `always_inline` did (clang and GCC both accept the attribute).

After that, instructions are flat: 2.567G -> 2.570G. At these solver
nodes the batch costs about what the per-child checks did. Wall time
fell anyway, presumably because the per-child check was a dependent
chain right after each `doMove`:

| solver P <= 27 wall, interleaved, pinned | before | after |
|---|---|---|
| 10 rounds against the previous commit | mean 5.65 s | mean 5.23-5.42 s; faster in 9 of 10 |
| last 6 rounds | 5.78 5.32 5.73 5.72 5.87 5.49 | 5.13 5.07 5.37 5.30 5.36 5.17 |
| 4 rounds against main | 6.50 6.38 6.02 6.10 (mean 6.25) | 5.54 5.33 5.34 5.32 (mean 5.38, -14%) |

These ran on a quiet machine (no synthesis job). Solver results and
node counts are identical at P <= 24 and P <= 27, on both builds.

`ab_match` uses the mask too, and keeps its `AB_CHECK_LINES` comparison
against the real child. It compiles cleanly (against the ONNX Runtime
1.20.1 CPU headers). It could not be run here: there is no small-network
file in the repository, and this Xeon lacks the AVX-VNNI that
`small_net.h` needs. To check it on the i7-12700H, run a match with
`AB_CHECK_LINES=1`; any disagreement aborts.

## Not done

- `basicLegalMoves` (7%) in the nibble layout. Its placement and move
  planes have to come out as contiguous 16-bit move-ID fields, so the
  `pext` calls come back at the end. A count by hand gives about 20+4
  operations against the current 24, which is not worth the risk.

## Next steps

- `synth_reserves.py`: whether the reserve sum fits in 3-6 operations
  (lengths 1-2 UNSAT, length 3 timed out at 600 s). A longer timeout, or
  restricting multiplier constants to sparse patterns, might settle it.

- Run `ab_match` with `AB_CHECK_LINES=1` on the i7-12700H (it could not
  run here), then compare nodes/s with and without the mask.
- Parents with a line still ask each child. Covering them needs the
  breaking rule in the proof, too.
- Re-time the solver on the i7-12700H with clang-20, three interleaved
  rounds, to settle the wall-time question.
