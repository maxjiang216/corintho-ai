# The spreadsheet's last column, and a 74% second-player win rate

*2026-09-21 — the question open since entry 01, answered from the recorded run*

## Summary

The user suggested the unlabelled last column was "the score of the current
generation against the current best generation". **It is not** — that quantity is
exactly recoverable and has the opposite shape. The column is a *second-player*
score.

More importantly, pulling the recorded data to check produced the finding that
has been hypothesised since entry 01 and never tested:

> **In self-play training games, the player who moves second wins ~74%.**

Raw data: `data/colour-imbalance.tsv`, 93 generations.

## Where the data is

`generations/gen_*/` was in the repo the whole time. Each holds `rating.txt`,
`metadata.txt`, `training_logs/score_verbose.txt` and
`testing_logs/score_verbose.txt`. Nobody had looked.

`metadata.txt` also records the real training configuration, which no entry so
far had right:

```
Number of games: 25000        Number of test games: 1600
Number of iterations per turn: 1600
Number of threads used in self play: 48
```

## Ruling out the hypothesis

`main.pyx` computes the gating score as `score = tester.score()` after
`play_games(tester, ..., best_model, new_model)` — so it *is* new-vs-best. It is
not stored per generation, but it is recoverable, because `update_rating` is
invertible:

```python
new_rating = best_gen_rating - 400 * np.log10(1 / score - 1)
```

so `score = 1 / (1 + 10**((best_rating - new_rating) / 400))`. Replaying the
ratings with the promotion rule gives:

| gen | rating | score vs best | passed |
|---|---|---|---|
| 1 | 579.9 | 0.9406 | yes |
| 4 | 1249.0 | 0.6031 | yes |
| 8 | 1367.6 | 0.5428 | yes |
| 89 | 5505.6 | 0.4816 | no |
| 92 | 5536.5 | 0.5259 | yes |
| 93 | 5537.1 | 0.5009 | yes |

Range 0.45–0.94, **decaying from 0.94 toward 0.50**, non-monotonic. The column
in the spreadsheet runs 0.574 → 0.935 the other way. Not this.

### What it is instead

`testing_logs/score_verbose.txt` gives the second-player score in the *test*
games, range **0.5731 … 0.9550**, which brackets the reported 0.574 and 0.935.
That is the column.

Note it is a **confounded** quantity, not a clean colour measure: test games
alternate parity (`i % 2` in `Trainer::initialize`) *and* alternate which agent
plays which colour, so it mixes "how much stronger is the new agent" with "how
much does colour matter". That is why it swings so widely. It is not a useful
metric and should probably be dropped from the sheet.

## The finding that matters

`training_logs/score_verbose.txt` is the clean measure, because **training games
have no parity** — `Trainer::initialize` comments that parity "does not affect
training games", so every game is the same setup.

`Trainer::writeScores` splits the games in two halves and reports P(first player
wins) over the even-indexed half and P(first player loses) over the odd-indexed
half. They are two independent 12,500-game estimates of the same quantity, and
they agree to well within sampling error, which is a useful internal check:

```
gen 93   First player  wins 0.2556  draws 0.0121  losses 0.7323
         Second player wins 0.7403  draws 0.0093  losses 0.2504
```

Across the run:

| gen | P1 win | draw | P2 win | P2 score |
|---|---|---|---|---|
| 1 | 0.4672 | 0.0020 | 0.5419 | 0.5429 |
| 3 | 0.3955 | 0.0133 | 0.5971 | 0.6038 |
| 5 | 0.2393 | 0.0173 | 0.7442 | 0.7528 |
| 6 | 0.2201 | 0.0211 | 0.7659 | 0.7765 |
| 90 | 0.2382 | 0.0106 | 0.7370 | 0.7422 |
| 94 | 0.2579 | 0.0111 | 0.7354 | 0.7410 |

Peak 0.8211. It starts near even at generation 1 — when the network is
untrained and play is near-random — climbs steeply through generation 6, then
sits at **0.74–0.78 for the remaining ~88 generations**.

That shape is important. A near-even result under random play that diverges as
soon as the network learns anything means this is **not** a symmetric game being
mis-measured; it is something the agent discovers and then exploits immediately.

## Why this plausibly outranks the performance work

1. **The value target is nearly constant.** ~74% of training games have the same
   outcome sign. A value head can score well by learning "second player wins"
   and little else, which is a weak gradient for the thing that actually drives
   MCTS.
2. ~~**The gate was passing noise.** ... 64 of 93 generations were promoted.~~
   **RETRACTED (entry 12).** This was wrong. It assumed a promotion threshold of
   0.5 and *replayed* the promotion chain from the ratings. The real threshold
   was **0.52**, recorded in `generations/gen_79+/metadata.txt`, and those same
   files record `best_generation` and `best_gen_rating` directly so no replay was
   needed. Using the recorded chain, generations 79-93 promoted **3 of 15**. The
   gate was rejecting about 80% of candidates, and 0.52 at 1,600 games is roughly
   a one-sided 94% test. See entry 12.
3. Both effects are invisible to every optimization on this branch. The engine
   being 1.8x faster produces more of the same skewed data.

## What is NOT yet established

Being explicit, because this is exactly the kind of finding that invites
over-reading:

- **Whether 74% is the true game or an artifact of the line-breaking defects.**
  Entry 04 found the engine allowed 9,686 forbidden moves and forbade 5,673
  legal ones per 6.1M positions, and every generation here was trained under
  those defects. If they are asymmetric in whose options they remove, some of
  this imbalance is manufactured. **This is directly testable now**:
  `game_reference.cpp` still holds the frozen pre-fix rules, so random-playout
  first/second win rates can be compared between old and fixed engines with no
  training at all. That is the single highest-value experiment outstanding.
- **Whether a second-player advantage is a problem or just a fact.** Plenty of
  games are decided by parity. It hurts *learning* here because of the near
  constant value target, not because it is unfair.
- **The 0.5731–0.9550 identification of the column.** The bounds match well but
  the per-generation values were not checked against the sheet, because the
  sheet is not in the repo. Worth one confirmation from the user.

## Next steps, in order

1. ~~Random-playout parity test~~ — done in entry 11. The rules fix is parity
   neutral; the defects did not manufacture the imbalance.
2. Re-measure colour balance with the fixed engine under a real (not stub)
   network before drawing conclusions from the old 74%.
3. ~~Revisit the promotion gate.~~ Superseded by entry 12: the gate was not
   admitting 0.51, and was roughly correctly calibrated for 1,600 games.
4. If the imbalance survives the rules fix, consider it directly — the usual
   remedies are a komi-like value adjustment, or training with colour-balanced
   position pairs.

## Correction to earlier entries

Entries 01 and 05-09 each carried "the spreadsheet's last column is
unidentified" as the top open item, with the note that *if* it were a
second-player score then colour imbalance would outrank the performance work.
The conditional has now resolved: it is a second-player score, and the colour
imbalance is real and large. The performance work stands, but it was never
going to move strength on its own.
