# The promotion gate was not passing noise. I misread it.

*2026-09-21 — retraction, and what was changed anyway. Commit `1fe1ac8`.*

## The claim being retracted

Entries 10 and 11 both stated, and the index repeated:

> The gate was passing noise. Recovered new-vs-best scores sit at 0.48–0.53 from
> about generation 40 on, yet **64 of 93 generations were promoted** on 1,600
> test games.

**This is false.** It was the stated motivation for changing the gate.

## What was actually true

Generations from 79 onward write `metadata.txt` as JSON, and it records exactly
what I had reconstructed:

```json
"test_threshold": 0.52,
"best_generation": 78,
"best_gen_rating": 5484.984857163416,
"num_test_games": 1600,
```

Two errors, compounding:

1. **The threshold was 0.52, not 0.5.** I never checked; 0.5 was the argparse
   default in `wrapper.py`, and the recorded run overrode it.
2. **I replayed the promotion chain instead of reading it.** `best_gen_rating` is
   just `rating.txt` of the best generation, so with the recorded
   `best_generation` the scores are exact. I instead re-derived the chain by
   simulating the promotion rule from generation 0 — and a replay diverges
   permanently after its first wrong decision.

Using the recorded chain (`best_gen_rating == R[best_gen]` on all 15 rows, so it
is self-consistent):

| gen | score | pass @ 0.52 |
|---|---|---|
| 79 | 0.4925 | no |
| 81 | 0.5144 | no |
| 83 | 0.5016 | no |
| 84 | 0.5213 | **yes** |
| 87 | 0.5253 | **yes** |
| 89 | 0.4831 | no |
| 92 | 0.5275 | **yes** |
| 93 | 0.5009 | no |

**3 of 15 promoted.** The gate was rejecting about 80% of candidates.

And it was reasonably calibrated: at 1,600 games with a ~1% draw rate, 0.52 sits
about 1.59 standard errors above 0.5, i.e. roughly a **one-sided 94% test**.

## How the error survived as long as it did

Worth recording, because the shape recurs:

- The reconstruction was *internally* consistent. Inverting `update_rating` is
  correct, the arithmetic was right, and the output looked plausible — scores
  hovering near 0.5 late in a plateaued run is exactly what one expects.
- Nothing cross-checked it against recorded ground truth until the gate was
  being rewritten and the counts were needed. The check that caught it
  (`best_gen_rating` vs `R[best_generation]`) took one line and could have been
  run at the start.
- **A replay cannot be spot-checked at the end.** One wrong decision poisons
  every later row, so agreement on the last generation would have proved
  nothing. This is different from a derived quantity, where checking a sample is
  meaningful.

**Rule: read what was recorded before reconstructing it.** The threshold and the
full chain were on disk the entire time.

## What was changed anyway, and on what grounds

The user asked for the gate to be a 95% confidence interval on decisive games
(draws discarded). That was implemented in `1fe1ac8`. The original justification
is gone, so here is the honest one:

**Kept because:** a fixed threshold does not track sample size. 0.52 is roughly a
one-sided 94% test at 1,600 games, a much stronger demand at 6,400, and much
weaker at 400. The plan is to vary the number of test games, so a fixed number
would silently change how strict the gate is. A confidence level will not.
Discarding draws is also more correct in kind: a draw carries no information
about which side is stronger, and averaging it in at 0.5 only shrinks the
apparent distance from 0.5.

**Not kept because it fixes a plateau.** It does not. Retroactive effect over
generations 79-93 is **one changed decision**: generation 84 (score 0.5213, CI
low 0.4968) becomes a reject. 87 and 92 still promote, at CI lows of 0.5009 and
0.5032. Break-even at 1,600 games moves 0.520 → 0.525.

This is a tidiness-and-robustness change, not a fix. It should not be credited
with anything if the next run improves.

## What still stands from entry 10

The colour finding is untouched by this and rests on different data
(`training_logs/score_verbose.txt`, not the ratings): **the second player wins
~74% of training self-play games**, saturating by generation 6 and flat for ~88
generations while the rating climbs 4,000 Elo. The value target being nearly
constant remains the best available explanation for the plateau.

Entry 10's two-item list is now a one-item list. That one item is the real one.

## Verification status of the change

- C++ (`Trainer::numWins/numDraws/numGames`): builds clean, digests unchanged.
- `promotion.py`: 9 tests in `corintho_ai/python/test_promotion.py`, all passing.
  Deliberately plain Python rather than cython so it is testable without the
  extension. Writing the tests caught a real flaw — Wilson returned `-2.8e-17`
  for zero wins; now clamped to [0, 1].
- **`main.pyx` and `wrapper.py` are NOT verified.** Neither Cython nor
  TensorFlow is installed here, so the cython binding of the three new methods
  and the `from promotion import ...` line have never been compiled. They must
  be built before the next training run.
