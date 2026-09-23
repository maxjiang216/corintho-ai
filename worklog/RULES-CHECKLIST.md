# Rules checklist

One file at the top of `worklog/`, not inside an epic, because it outlives them.
It collects every place where **the engine's behaviour and the written rules do not
obviously agree**, or where the rules are silent and the engine had to pick something.

## Why this file exists

The benchmark harness uses the current implementation as its oracle: `digest_game`
and the planned `getLegalMovesReference` both ask "does the new code match the old
code". That is the right question for a refactor, and it is the only question that
can be automated — the rules are prose, and the subtlest rule ("stable") is encoded
as a 102-entry precomputed table that no parser will recover from a sentence.

But a reference implementation is **descriptive, not normative**. It says what the
engine does, not what it should do. So the two layers work together:

1. **Differential testing** against the reference catches *that* behaviour changed.
2. **The written rules** decide *which side is wrong* when it does.

Usually the new code is wrong. Occasionally the reference was wrong all along — and
that is an engine bug, which matters far more than whatever optimization surfaced it.

**Add to this file whenever a behaviour rests on an inference rather than a
quotation.** Say what to watch for, not only what was assumed. Tick items off when
they are settled, and record who or what settled them.

## Sources

- The rules overlay in the web app: `web/corintho.js`, `getRulesPages()`, 4 pages.
- [jpneto, Games of Towers — Corintho](https://jpneto.github.io/world_abstract_games/corintho.htm),
  which the overlay cites as its source.
- Copyright © 2005 Paolo Scattini, Family Games, Inc.

---

## SETTLED — 1. Diagonal lines DO count. The engine is correct.

**Engine behaviour.** `Game::applyLines` (`corintho_ai/cpp/src/game.cpp:397-403`)
calls four detectors: `applyRowColLines` for rows, again for columns,
`applyLongDiagLines`, and `applyShortDiagLines`. The short diagonals are triples
such as (1,1), (0,2), (2,0) — see `game.cpp:362-395`.

The `line_breakers` table (`corintho_ai/cpp/include/util.h:85`) has **102 entries =
34 shapes × 3 piece types**: 12 rows (4 × three shapes), 12 columns, 6 long
diagonals (2 × three shapes), 4 short diagonals. The diagonals are not incidental;
a third of the table is devoted to them.

`web/engine.js:198,221,249,250` does the same, so the web app and the training
engine agree with each other.

**What the rules say.** All three sources describe orthogonal lines only:

- The web app's own overlay: *"a stable line of three stacks in a row (orthogonal)"*.
- jpneto: *"3 in-a-row of stacks of the same type"*, with rows and columns; diagonals
  are not mentioned.
- The developer's summary: *"Form a stable 3-in-a-row of stacks with the same top-piece type."*

**The sharpest form of the discrepancy:** the rules text in `web/corintho.js`
contradicts the engine in `web/engine.js`, both written for the same app.

**Why it matters.** If diagonals should not count, the engine is playing a different
game, and 95 generations were trained on it. Engine and web app agree with each
other, so the AI is self-consistent and internally valid for *its* game — but that
game may not be Corintho.

**Resolved 2026-09-20 by the developer: diagonals do count. The engine is correct and the
written summaries are incomplete.**

The engine, `web/engine.js`, and the `line_breakers` table all stand as-is. No code
change. The 95 generations of training were on the right game.

---

## SETTLED — 2. Any piece MAY be placed on an empty cell. The engine is correct.

**Engine behaviour.** `Game::canPlace` (`corintho_ai/cpp/src/game.cpp:181-205`)
checks piece availability, then returns `true` immediately if the target space is
empty — **before** it ever examines the piece type. The type checks that follow
(`kBase` may not stack, columns need no column or capital present, and so on) are
only reached for non-empty spaces.

Verified directly on the empty starting board:

```
base     at (0,0): LEGAL
column   at (0,0): LEGAL
capital  at (0,0): LEGAL

Legal placements on empty board: base=16 column=16 capital=16
```

So the opening position has 48 legal placements, and a bare capital may be dropped
onto bare ground.

**What the rules say.**

- The developer's summary: *"A base can only go on an empty cell. A column must go on top of
  an existing base or column. A capital must go on top of an existing column."*
- The web app's overlay: *"A column or capital may sit only on a base or another
  column. A base cannot be placed on top of anything."*
- jpneto: *"A column and a capital can be only be on top of a base or a column. A
  base cannot be on top of any other piece."*

All three agree that a column or capital requires something beneath it.

**Why it matters.** This changes the branching factor of the opening substantially —
48 legal first moves instead of 16 — and therefore the shape of every search tree
the engine has ever built.

**Note for the bitboard rewrite:** `PLAN.md` §13.2 derives
`placeable_base = empty`, `placeable_col = ~f & ~c & ~a`, `placeable_cap = ~f & ~a &
(~b | c)`. Those reproduce the **current engine**, including this behaviour. If the
rule is wrong, fix the rule first and re-derive, or the rewrite will faithfully
preserve a bug and the digests will make it look correct.

**Resolved 2026-09-20 by the developer: any piece may be placed on an empty cell. The engine
is correct and the written summaries are incomplete.**

`canPlace` stands as-is, and the opening genuinely has 48 legal placements. The
bitboard derivation in `PLAN.md` §13.2 — `placeable_base = empty`,
`placeable_col = ~f & ~c & ~a`, `placeable_cap = ~f & ~a & (~b | c)` — is therefore
**correct as written**, and Stage 2 is unblocked.

---

## OPEN — 4. The engine's line handling is wrong in both directions

**Two distinct defects, both letting a player leave a line standing when the
rules require breaking it.**

### 4a. `applyLines` stops after the first line in each category

`applyRowColLines` returns after the first row (or column) containing a line;
`applyLongDiagLines` returns after the first of the two long diagonals;
`applyShortDiagLines` after the first of four. The comment at `game.h:105` states
the assumption: *"there can only be up to 1 of each type, so we can return
early."* It is false.

Measured over 8,190,155 reachable positions: **139 positions (0.0017%) where the
engine offers a move that leaves a second line standing. All 139 were
non-terminal.**

Worked example, found by random play:

```
ply 14  (P1 to move)                ply 15  (P2 to move)
 C  |    |    |B                     C  |    |    |B
  A |    |  A |                       A |    |  A |
 C #| C  | C  |  A                   C  | C  | C  |  A
 CA | CA | C  |  A                   CA | CA | CA#|

lines: row2-left3(C)                lines: row2-left3(C), row3-left3(A)
played: d1L                         3 legal moves, all leave a line standing
```

P1 moves the capital d1 onto c1, creating a second line without breaking the
first. `applyRowColLines` finds row 2, returns, and never applies row 3.

Note the two lines are **disjoint**, so the intuition that co-present lines must
intersect (because the move creating the second would freeze the intersection)
does not hold. Measured: 0.51% of multi-line positions contain a disjoint pair.

### 4b. `line_breakers` is liberal about extend-to-four, and only partly corrected

A 3-in-a-row is legitimately unmade by extending it to a 4-in-a-row. Whether a
*move* move does that depends on the **moved stack's top type**, which a static
table cannot encode, so `line_breakers` includes those moves unconditionally.

The author knew. `game.cpp:252-278`:

> *"A capital must be used to extend line when moving. The applyLine function is
> liberal in this case. So we need to remove the illegal moves."*

That correction exists **only in `applyRowColLines`, and only when
`top1 == kCapital`.**

Measured: **1.43% of single-line positions admit a legal move that leaves the
line fully intact** (after correctly allowing extend-to-four as a break).
Broken down, every cell is explained by the single root cause:

| line kind | base | column | capital |
|---|---|---|---|
| row | 0.000% | **1.806%** | 0.000% |
| column | 0.000% | **1.287%** | 0.000% |
| long diagonal | **5.232%** | **5.946%** | **5.616%** |
| short diagonal | 0.000% | 0.000% | 0.000% |

- Row/column capital lines: clean, the fix-up works.
- Row/column base lines: clean for a different reason — a base-topped stack is
  `{B}`, and `canMove` needs `bottom(from) - top(to) == 1`, i.e. `0 - top(to) == 1`,
  so the destination would have to be empty, which `canMove` forbids. **No move
  can extend a base line to four**, so there is no liberality to correct.
- Row/column column-lines: the fix-up covers only capitals.
- Long diagonals: `applyLongDiagLines` has no fix-up at all.
- Short diagonals: maximal 3-cell shapes with no fourth cell, so no
  extend-to-four entries exist.

Worked example. Line is the a4–d1 diagonal's lower three, `b3 c2 d1`, all
capitals:

```
BEFORE                              AFTER  (move a3U)
      a       b       c       d           a       b       c       d
 4 |<B.. >| ...  | BCA  | ...  |     4 |<BC.*>| ...  | BCA  | ...  |
 3 |{.C. }|[BCA*]| ...  | .C.  |     3 | ...  |[BCA ]| ...  | .C.  |
 2 | .C.  | ...  |[BCA ]| ...  |     2 | .C.  | ...  |[BCA ]| ...  |
 1 | B..  | ...  | .CA  |[.CA ]|     1 | B..  | ...  | .CA  |[.CA ]|
```

`a3U` moves a lone column from a3 onto the base at a4. The line `b3 c2 d1` is
all capitals before and after; a4 becomes a **column**, so the diagonal reads
C,A,A,A and is not a four. The move does nothing to the line, and the engine
calls it legal.

### The correct rule (settled 2026-09-20 by the developer)

> A move is legal iff, for **every** shape S and type t such that all of S's
> cells had top type t before the move, afterwards either
> **(a)** S's cells no longer all have top type t, or
> **(b)** S is a 3-shape and its containing 4-shape now has all cells of type t.
>
> Short diagonals have no containing 4-shape, so (b) never applies to them.

**No subsumption.** A 4-line contains two 3-lines and all three are checked
independently. This is not a special case: if `{a,b,c,d}` holds before, it must
not hold after, which makes clause (b) unsatisfiable for `{a,b,c}` and
`{b,c,d}`, so both must be destroyed outright. The developer: *"the line of 4 and both
lines of 3s have to be broken."*

Derived consequence, since a move touches at most two cells and the two ends of
a 4-line are not adjacent: **every legal move in a position containing a 4-line
must change the top type of one of the two middle cells.**

Verified by an independent route — the consequence was derived by hand and the
rule implemented separately. Over 224,677 positions containing a 4-line and
155,753 moves legal under the rule, **zero** failed to change a middle cell.

### Measured divergence from the rule

2,460,319 positions. Basic legality validated first: on 2,090,206 **line-free**
positions, where the engine's answer *is* basic legality, an independently
written structural test matched it with **zero** mismatches.

| | positions | moves |
|---|---|---|
| engine **allows** a move the rule forbids | 3,790 (0.154%) | 3,972 |
| engine **forbids** a move the rule allows | 2,270 (0.092%) | 2,270 |
| **total divergence** | **5,927 (0.241%)** | |

### Catalogue of `line_breakers` defects

Attributed exactly, using only positions containing a single line:

| class | cells | meaning |
|---|---|---|
| **PURE ERROR** — table includes, move *never* breaks | **13** | encoding mistakes; fixable |
| LIBERAL — table includes, only sometimes breaks | 63 | depends on moved stack's top type |
| CONSERVATIVE — table excludes, sometimes breaks | 181 | **a static table cannot express this** |
| consistent | 7,935 | |

The 13 pure errors are a **transposition between the two long diagonals**:

```
move 13 = b4D  (0,1) -> (1,1)     (1,1) is on the NW diagonal
move 14 = c4D  (0,2) -> (1,2)     (1,2) is on the NE diagonal

NW diagonal: (0,0) (1,1) (2,2) (3,3)     NE: (0,3) (1,2) (2,1) (3,0)
```

Lines 72–79 (NW) contain `c4D` and omit `b4D`. Lines 81–88 (NE) contain `b4D`
and omit `c4D`. They are swapped. Types B and C only — for capitals the same
move sometimes genuinely breaks, so it lands in LIBERAL instead.

The thirteenth is line 7 (row-2 left-3, columns), which contains
`d1L` = (3,3)→(3,2), entirely in row 3. The move that belongs there is
`d1U` = (3,3)→(2,3), landing on the extend-to-four cell d2.

**The other 244 cells are not patchable.** Whether a move-move extends a line to
four depends on the moved stack's top type, which is dynamic. This is why the
author needed the capital-line fix-up, and why it was never finished.

### Category A is UNREACHABLE in correct play

The developer: if b3 is frozen the previous move ended there, so the other line already
existed and was not broken — meaning that earlier move was itself illegal.

Tested by replaying 6,139,414 positions choosing moves by the **rule** rather
than by the engine:

```
maximal lines present:  0 -> 85.005%   1 -> 13.850%   2 -> 1.132%   3 -> 0.014%
positions with >=2 maximal lines            : 70,322 (1.1454%)
positions with 2+ in the SAME detector cat. :      0 (0.000000%)
```

**Zero.** The comment at `game.h:105` is empirically correct for reachable
positions: each of the four detectors really does find at most one line. The
early return is a *latent* hazard, not an active bug — it only bites once some
other defect lets play into an illegal position.

**Methodological note.** Every earlier frequency here was measured by walking
the engine's own buggy legal-move set, which wanders into positions that cannot
legally occur. On rule-reachable positions only:

```
engine ALLOWS a forbidden move : 9,307 positions (0.152%), 9,686 moves
engine FORBIDS a legal move    : 5,673 positions (0.092%), 5,673 moves
```

### Analytic enumeration of the table: only 13 real errors

Classifying all 102 x 96 = 9,792 cells by hand rather than by sampling, after
excluding moves that can never be basically legal:

| class | count | |
|---|---|---|
| **E** — in table, NEVER breaks | **13** | genuine encoding errors |
| **H** — missing, ALWAYS breaks | **0** | none |
| **DEPENDS** on moved stack's top | **816** | 467 present, 349 missing |
| correct | 8,963 | |

Three facts had to be built into the model before the count settled, each of
which removed a large block of false positives:

- **4-line entries do triple duty.** The engine subsumes, applying only the
  4-line's entry, so that entry must encode "breaks the 4-line *and* both
  3-subsets". Confirmed: `line_breakers[24]` (row0 all-four, bases) permits
  exactly `b3U, c3U, Cb4, Cc4` — precisely the four moves that change a middle
  cell, matching the derived consequence. Omitting the rest is correct.
- **A move between two cells of the same line is never playable.** Both tops are
  `ty` and `canMove` needs `bottom(from) - top(to) == 1`, but `bottom <= top`.
- **A base-topped stack can never be moved at all.** It is exactly `{B}`, so
  `bottom == 0` and `canMove` would need an empty destination, which it forbids.
  Likewise many placements onto a line cell are never legal.

So the table is **mostly right**. Its only outright errors are the 13
transpositions. Everything else wrong with it is the top-dependence it cannot
express — and that is unfixable by any static table, which is the whole argument
for replacing it with the explicit rule.

The earlier empirical claim that "H accounted for 97.3% of over-restrictive
cases" was mislabelled: with H analytically zero, every one of those instances
is a DEPENDS-missing case, i.e. conservative top-dependence, not an encoding bug.

### Impact

The search visits on the order of 10^9 positions per generation, so this is
roughly 1.5 million affected nodes per generation. All 95 generations trained
with it.

### How these were found, and a strong caution

Neither is detectable by the benchmark harness. `digest_game` and
`getLegalMovesReference` compare the engine against **itself**, so a defect in
the reference is invisible by construction. They were found by building a line
detector independently from the 34 shapes and diffing.

**Three successive measurements of 4b were wrong before this one**, each caught
by the developer's domain knowledge, and each time the error *inflated* the apparent
problem:

1. **44%** — did not count extending to four as a legitimate break.
2. **1.43%, "over-restrictive is 9x larger"** — the oracle subsumed 3-subsets
   under their containing 4-line, waving through moves that destroyed a 4 while
   leaving a 3 standing.
3. **0.241%** — current, after removing subsumption.

Treat the current figure as provisional. The derived-consequence test above is
the first independent confirmation the rule implementation has passed.

### To resolve

Fix before the bitboard rewrite bakes this in, since `digest_game` would certify
the frozen behaviour as correct. The rewrite is the natural place: the explicit
rule replaces `line_breakers` entirely, and is cheap with bitboards because a
move changes at most two cells' top types, so each present line only needs those
two cells tested. Lines exist in 15% of positions, so the common path skips it.

Sequence: fix first, with digests deliberately moving and the reference
re-frozen; then optimize, with digests held.

## OPEN — 3. The web app's rules overlay is inaccurate

Not an engine bug; a documentation bug, and the reason items 1 and 2 looked like
discrepancies in the first place.

`getRulesPages()` in `web/corintho.js` tells players:

- *"a stable line of three stacks in a row (orthogonal)"* — but diagonals count
  (item 1), and `web/engine.js` in the same app detects them.
- *"A column or capital may sit only on a base or another base or column"* — but any
  piece may be dropped on an empty cell (item 2).

So a player reading the in-app rules will be surprised by legal moves the engine
allows and by losses to lines they did not know existed. The overlay already hedges
with *"when in doubt, trust the game over message"*, which is a symptom rather than
a fix.

**Low priority, user-facing, entirely separate from the optimization work.** Fixing
it is two sentences in `getRulesPages()`.

## How these were found

Worth recording as method, since it generalized. The discrepancies surfaced while
asking a different question — whether the written rules could replace the reference
implementation as a test oracle for the bitboard refactor. They could not, but
reading them closely enough to decide that turned up two places where engine and
prose disagreed.

Both turned out to favour the engine. The value was not in finding bugs; it was in
converting two silent assumptions into confirmed facts, before a rewrite baked them
in where `digest_game` would have certified them as correct.
