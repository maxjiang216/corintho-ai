# 2026-09-20 — Fixing line breaking, and what the detour found

Branch: `perf/engine-optimizations`, off `perf/training-overhaul`.

Started as Stage 1 of the bitboard rewrite. Turned into a correctness
investigation when a question about multi-line positions surfaced two real
defects in the engine's legal move generation, both present since 2023 and both
active through all 95 training generations.

## TL;DR

- **Stage 1 landed** (`8a7af44`): per-space caching. `getLegalMoves` 929 → 646 ns,
  branch mispredicts −15.7%, digests held.
- **Line breaking was wrong in both directions** and is now fixed (`5b2afb6`).
  On 6,139,414 reachable positions the old engine allowed **9,686 forbidden
  moves** (0.152% of positions) and forbade **5,673 legal ones** (0.092%).
- **The fix is a 12–16% slowdown**, accepted. Stage 2 is expected to recover it.
- **`line_breakers` could not be patched.** 13 of its 9,792 cells are outright
  transposition errors, but **816 are top-dependent** and inexpressible in any
  static table.
- **The early return was not a bug.** It rests on a real structural invariant;
  it only misbehaves once other defects create illegal positions.
- **Four of my own measurements were wrong before being right**, every one
  caught by Max. See *Corrections*.

## What was wrong

Two defects, catalogued in `worklog/RULES-CHECKLIST.md` item 4.

**The table is liberal about extend-to-four.** A three is legitimately unmade by
becoming a four, but whether a *move-move* does that depends on the moved
stack's top type — dynamic, and unencodable in a static table. The original
author hit exactly this and wrote a fix-up, but only in `applyRowColLines` and
only for capital lines (`top1 == 2`). Column lines there, and everything on the
long diagonals, kept the over-permissiveness. Short diagonals are immune,
being maximal at three with no fourth cell.

**Thirteen cells are transposed between the two long diagonals.** Lines 72–79
(NW) carry `c4D`, which lands on `(1,2)` on the **NE** diagonal; lines 81–88 (NE)
carry `b4D`, which lands on `(1,1)` on the **NW**. Swapped. Plus `d1L` in line 7,
where `d1U` belongs. Types B and C only — for capitals the same move sometimes
genuinely breaks, so it falls into the top-dependent bucket instead.

## The rule, and why it replaced the table

Settled by Max:

> A move is legal iff, for **every** shape S and type t such that all of S's
> cells had top type t before the move, afterwards either **(a)** S's cells no
> longer all have top type t, or **(b)** S is a 3-shape and its containing
> 4-shape now has all cells of type t.

**No subsumption.** A four and both its threes are checked independently. This
needs no special case: if the four held before it must not hold after, which
makes clause (b) unsatisfiable for its threes, so both must be destroyed
outright. Derived consequence — since a move touches at most two cells and the
ends of a four are not adjacent — **every legal move in a position containing a
four must change one of the two middle cells.** Verified over 224,677 positions
and 155,753 legal moves, zero violations.

The rule's inputs were proven complete: `(shape, type, move, source-top,
extend-cell-top)` determines the outcome, with **0 ambiguous cells over 30,672
observed combinations**. Nothing about stack interiors, reserves, frozen state
elsewhere, or history matters.

That completeness is why the implementation needs no board copy. A move rewrites
at most two tops — a place changes its target; a move-move empties its source and
retops its destination — so the post-move state of any line follows directly from
the move and `SpaceInfo`.

## The early return was sound

`applyLines` returning after the first line in each detector category looked like
a bug. It is not. The structural argument:

1. **Every line after a legal move passes through the move's destination.** The
   source is emptied and cannot be in a line; everything else is untouched; and
   every pre-existing line had to be broken or the move was illegal. Verified:
   661,942 maximal lines, zero exceptions.
2. **A cell belongs to at most one shape per category.** Rows and columns
   trivially. The two long diagonals `{0,5,10,15}` and `{3,6,9,12}` are
   **disjoint** — a 4×4 board has no centre. The four short diagonals are
   disjoint too, covering 12 cells; the corners lie on none.

One destination × one shape per category = at most one line per category, which
is exactly the granularity each detector returns at. Confirmed empirically:
**0 positions out of 6,139,414 with two maximal lines in the same category.**

The invariant depends on *"every pre-existing line was broken"* — precisely what
the other defects violated. So the early return is a latent hazard that only
bites inside an already-illegal position. It is gone anyway, since the new code
checks every line.

## Measurements

Machine on AC, quiet. Records in `bench/results/`.

Correctness, four independent routes:

| check | before | after |
|---|---|---|
| `rulecheck`, exhaustive over the complete index | 1,440 mismatches | **0** |
| `verify`, differential vs re-frozen reference, 200k positions | — | **PASS** |
| independent oracle using real `doMove`, 6,139,414 positions | 9,307 / 5,673 | **0 / 0** |
| `digest_game` | `0f82fc99147482e4` | `d8f9bd2d22ee9731` |
| `digest_engine` | `db113d11c95bcedb` | `ad85cefa8bf9abf5` |

Performance, `after-stage1` → `after-linefix`:

```
ns_game_getlegalmoves          646.4 -> 1006.2   +55.7%
ns_node_ctor_dtor_alloc_heavy  863.5 -> 1082.8   +25.4%
st_engine_seconds                3.7 ->    4.3   +15.8%
big_engine_seconds               2.1 ->    2.4   +12.3%
exact_instructions            917.1M -> 1837.9M +100.4%
exact_branch_mispredicts       9.94M ->  20.49M +106.2%
```

The old code was one AND per line covering all 96 moves at once; the new one
tests each move against each line. Stage 2 reverses that by deriving a breaker
**mask** per line from the board.

⚠️ Counter deltas are confounded: corrected rules change play, so these are not
the same games. Allocations rose 23.9%; per allocation it is about +62%
instructions.

## Corrections

Four wrong measurements, each caught by Max, each inflating the apparent problem.
The pattern is worth more than the conclusions: **every error came from an oracle
that was subtly wrong, and none was caught by the harness**, because
`digest_game` and `getLegalMovesReference` compare the engine against itself.

1. **"44% of single-line positions admit a non-breaking move."** Did not count
   extending to four as a legitimate break. → 1.43%.
2. **"Over-restrictive is 9× larger than over-permissive."** The oracle subsumed
   3-subsets under their containing 4-line, waving through moves that destroyed
   a four while leaving a three. → over-restrictive collapsed 95%.
3. **"1,128 table entries are missing."** The model ignored that 4-line entries
   do triple duty, that a move between two cells of one line is never playable,
   and that a base-topped stack can never be moved at all. → **0**.
4. **"Early return causes 281 bad moves."** Measured by walking the engine's own
   buggy move set into positions that cannot legally occur. → **0** under correct
   play.

Also two harness bugs of my own: `rulecheck`'s rule side did not check basic
legality (reported a meaningless 78%), and randomising the unconstrained cells
created **unrelated lines in 72% of fills**, which destroyed attribution.

## Carried forward

From `2026-09-20-03`:

**Done since**

- ~~Move the `Node` accessors into `node.h`~~ — not done, superseded in priority.
- ~~Build the `verify` mode and `getLegalMovesReference`~~ — done (`8cc2dbc`),
  validated by injecting a deliberate bug and confirming it was caught.
- ~~Bitboard stage 1~~ — done (`8a7af44`).

**Still live**

- **The spreadsheet's last column is still unidentified.** Unchanged since the
  first session and still the highest-value unknown in the project.
- **`to_eval_` oversizing** gates running at realistic scale locally.
- **Stub tree shape uncalibrated**, 18.2 turns/game vs 28.4 real.
- **Distinct-position instrumentation** not built.
- **S3** not set up.
- **`baseline.tsv`** was re-recorded on AC; battery-era records are not
  comparable to it.

## Next steps

1. **Bitboard Stages 2 and 3.** Now on a correct foundation, with `rulecheck`
   and `verify` as gates and digests that must hold. The line-breaking cost is
   the first thing to recover: derive a breaker mask per line from the board
   instead of testing each move.
2. **Move the `Node` accessors into `node.h`** — small, and a clean test of
   understanding, since the prediction is that it changes nothing on an LTO build.
3. **Measure the strength impact of the rules fix.** Gen 92 trained for 95
   generations on the buggy rules and now faces a slightly different legal-move
   space. A fixed-weights match, old engine vs new, would say whether the
   correction costs or gains, and whether the old generations stay valid as
   anchors.
4. **Fix `web/engine.js` and `web/line_breakers.js`.** They carry every one of
   these defects. `line_breakers` remains in `util.h`, unused and marked, only
   because the web file is generated from it.
5. **Fix the rules overlay in `web/corintho.js`** (RULES-CHECKLIST item 3) —
   two sentences, unrelated to any of this.
