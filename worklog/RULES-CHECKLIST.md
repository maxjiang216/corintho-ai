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

## OPEN — 1. The engine counts diagonal lines; the rules describe rows and columns

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
- Max's summary: *"Form a stable 3-in-a-row of stacks with the same top-piece type."*

**The sharpest form of the discrepancy:** the rules text in `web/corintho.js`
contradicts the engine in `web/engine.js`, both written for the same app.

**Why it matters.** If diagonals should not count, the engine is playing a different
game, and 95 generations were trained on it. Engine and web app agree with each
other, so the AI is self-consistent and internally valid for *its* game — but that
game may not be Corintho.

**To resolve:** check the physical rulebook or a more authoritative source than a
summary page. Failing that, Max's recollection of playing it.

---

## OPEN — 2. The engine allows any piece on an empty cell; the rules allow only a base

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

- Max's summary: *"A base can only go on an empty cell. A column must go on top of
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

**To resolve:** same as item 1.

---

## Settled

*(nothing yet)*
