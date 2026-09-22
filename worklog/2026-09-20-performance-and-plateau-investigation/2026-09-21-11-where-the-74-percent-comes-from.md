# The rules fix did not cause the second-player advantage

*2026-09-21 — branch `perf/engine-optimizations`*

## Question

Entry 10 found the second player wins ~74% of training self-play games, across
all 95 recorded generations — every one of which trained under the
line-breaking defects of entry 04. Was the imbalance manufactured by those
defects?

**No.** And nothing else tested reproduces 74% either. This entry records what
was ruled out.

## Correction to entry 10

Entry 10 proposed comparing rule sets via `getLegalMovesReference`, on the
grounds that `game_reference.cpp` "still holds the frozen pre-fix rules".

**It does not.** `git log -- corintho_ai/cpp/src/game_reference.cpp` shows two
commits: the one that created it, and **`5b2afb6`, the fix itself**, which
re-froze the reference to the corrected rules so `bench/verify` would keep
passing as a differential test.

The first run of this experiment compared the fixed rules with themselves. It
returned a difference of `+0.0000` with *identical* win counts and identical
mean game length to five decimals. That impossible-looking agreement is the
only reason the mistake was caught: a real comparison over 200,000 games
diverges by sampling noise even when the underlying rates match.

**Lesson: a differential reference that is kept passing is, by construction, not
a record of the old behaviour.** To compare against an old rule set, build
against an old tree:

```
git worktree add --detach /tmp/prefix 5b2afb6^
```

(gsl is not in the worktree, so point `-I` at the main tree's `gsl/include`.)

## Result 1 — the rules fix is parity neutral

Uniform random play, 200,000 games per arm, same seed, separate binaries built
from the two trees:

| rules | P1 | draw | P2 | P2 score | mean plies |
|---|---|---|---|---|---|
| pre-fix (`5b2afb6^`) | 0.4776 | 0.0193 | 0.5031 | **0.5127 ± 0.0011** | 19.47 |
| fixed (HEAD) | 0.4764 | 0.0192 | 0.5043 | **0.5139 ± 0.0011** | 19.46 |

Difference +0.0012 against a standard error on the difference of ~0.0016. Not
significant. The defects affected 0.152% / 0.092% of move decisions, and
whatever they did, they did it roughly symmetrically.

So entry 10's caveat is discharged: **the 74% was not manufactured by the
line-breaking bug.**

## Result 2 — but strength alone does not produce it either

`bench/parity.cpp` plays plain UCT with uniform random rollouts — no network,
no training. Second-player score against the simulation budget:

| search | P2 score |
|---|---|
| random play | 0.5134 ± 0.0035 |
| 25 sims | 0.4600 ± 0.0129 |
| 100 sims | 0.5163 ± 0.0129 |
| 400 sims | 0.5387 ± 0.0128 |
| **trained network, 1600 searches** | **~0.74** |

There is at best a weak upward hint (0.513 → 0.539) and it is not monotonic.
UCT with random rollouts does not find the advantage.

## Result 3 — the decisive observation, from the recorded data

This is the strongest evidence and it needed no new experiment, only reading
`data/colour-imbalance.tsv` against the ratings:

| gen | P2 score | rating |
|---|---|---|
| 1 | 0.5429 | 580 |
| 3 | 0.6038 | 1176 |
| 6 | 0.7765 | 1294 |
| 90 | 0.7422 | 5510 |
| 94 | 0.7410 | ~5537 |

**The imbalance saturates by generation 6 and then does not move for ~88 more
generations, while the rating climbs by over 4,000 Elo.**

If the second-player advantage were a function of playing strength, it would
have kept growing. It did not. It is a structural property of the game that even
a barely-trained network finds almost immediately, and further strength does not
deepen it.

That also explains Result 2: random rollouts give essentially no positional
signal in a game this tactical, so UCT-with-random-rollouts is not weak-but-
purposeful play, it is closer to noise. A one-generation network already beats
it at finding the relevant structure.

## Also checked and clean

`SelfPlayer::writeSamples` assigns value targets by starting at +1 for the last
recorded position and alternating backwards, with 0 throughout for a draw. That
is correct — the player to move in the last sample is the one who made the
winning move. A sign error here would have produced exactly this symptom, so it
was worth ruling out.

Note also that the board half of the network input is **not** flipped by side to
move, and does not need to be: Corintho pieces are neutral, so the only
player-specific state is the piece supply, which `writeGameState` does rotate by
`to_play_`. The encoding is sound.

## What this means for training

The cause is narrowed but the consequence is unchanged and is the point:

- **The value target is close to constant.** ~74% of training games end the same
  way. A value head scores well by learning "second player wins", which is a
  weak gradient for the signal that actually drives MCTS.
- **The gate was admitting noise.** Recovered new-vs-best scores sit at
  0.48–0.53 from about generation 40 on, yet 64 of 93 generations were promoted
  on 1,600 test games. 0.51 over 1,600 games is roughly one standard error from
  0.5.

Both are untouched by any of the engine work on this branch, and both are better
explanations of the plateau than anything in entries 02–09.

## Next

1. **Confirm the advantage with a real network.** Load `model_93.tflite` (or
   gen 92) and play it against itself under the *fixed* rules, measuring colour
   balance. This is the one measurement that would settle whether 0.74 survives
   the rules fix in the regime that actually matters. Needs a TF or tflite
   runtime, which is not available locally yet.
2. **Fix the promotion gate** regardless of the above: either raise the
   threshold, raise `num_test_games` well above 1,600, or use a sequential test.
   Promoting on 0.51 is promoting on noise.
3. If the advantage is confirmed, treat it directly — a komi-like value offset,
   or colour-balanced position pairs in the training set.

## Tooling added

`bench/parity.cpp` — random play (`sims 0`) or UCT with random rollouts. No
network, no Python. Its header documents the worktree method, so the next
session does not repeat the `game_reference.cpp` mistake.

Note `sims 0` is handled as an explicit branch: with no simulations there are no
visit counts, so the most-visited-child rule would pick move 0 every time and
replay one deterministic game. The first version did exactly that and reported
`P2 1.0000`, which is how it was caught. `sims 0` at 200,000 games now
reproduces the standalone random figure (0.5139) exactly, which is the
cross-check that the two code paths agree.
