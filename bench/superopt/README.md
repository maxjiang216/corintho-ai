# superopt: SAT-based superoptimization of the bitboard kernels

z3 (`pip install z3-solver`) solves bit-vector problems by bit-blasting
them to SAT. These scripts use it three ways on the engine's line and
legal-move kernels (worklog 2026-09-25-nn-architectures, entry 26):

| script | question | answer |
|---|---|---|
| `synth.py` | the shortest straight-line program for a kernel | CEGIS synthesis. Trying lengths 1, 2, ... in turn, an UNSAT query at length N-1 proves nothing shorter exists for the instruction set |
| `prove.py` | is a rewrite equal to the code it replaces? | a proof over all 2^64 board words: UNSAT for "some board where they differ" |
| `synth_reserves.py` | the shortest reserve sum, with add/sub/multiply and an input precondition | lengths 1-2 UNSAT; length 3 timed out (600 s) |
| `sketch.py` | can a fixed shape of program, with its shifts and masks left free, do the job with fewer terms? | CEGIS over the symbolic parameters; UNSAT proves no parameters work |

`kernels.py` holds z3 models of the C++ (`game.cpp`), current and
rewritten, line for line. A proof is only as good as that transcription,
so the C++ is still gated as usual (`make verify`, `make rulecheck`,
`golden game`, solver results and node counts). The proofs cover every
board word, legal or not.

## Results

`topNibbles` (bit k of each space's nibble set when its top is type k):

```
$ python3 synth_tops.py
  n=1..4: unsat
  n=5: found (163 s, 8 examples)
  v1 = b & 0x6666666666666666
  v2 = v1 >> 1
  v3 = ~v2 & b
  v4 = v2 >> 1
  v5 = ~v4 & v3
```

Five instructions is the minimum; the hand-written version had six.

```
$ python3 prove.py
PROVED  tops_nibbles == top_plane[0..2]
PROVED  has_line_nibbles == boardHasLine
PROVED  runs_nibbles == findLines run3/run4
PROVED  key rest: pext == loop
PROVED  key rest: portable == loop
PROVED  packRest: pext == loop
PROVED  reserve sum: multiply == loop
PROVED  horizon reserves: multiply == loop
PROVED  moves_from == per-move source test
PROVED  moves_from: portable == pext
PROVED  lineMakingMoves == hasLine(child), 96 moves
PROVED  lineMakingMoves: no bits beyond move 95
```

```
$ python3 sketch.py 4 3
K=4: sat after 130 examples, 400.0s     (the four hand-written directions)
K=3: unsat after 63 examples, 21.3s     (three terms cannot work)
```
