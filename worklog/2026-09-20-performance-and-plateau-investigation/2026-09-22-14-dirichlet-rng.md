# 14 — Dirichlet noise RNG: splitmix64, −3.7% engine time per request

## What was there

`TrainMC::generateDirichlet` draws one bucket index into the 1024-entry
`gamma_samples` table (`util.h`, Max's 2022 approximation of Gamma sampling) for
every legal move, every time a node receives its evaluation:

```cpp
dirichlet[i] = gamma_samples[(*generator_)() % kNumGammaBuckets];
```

`std::mt19937` was 5.9% of instructions (entry 13's profile): 2.18M calls in
3 games, ~26 instructions each, out of line even under LTO. Each call returns
32 bits; `% 1024` kept the low 10 and discarded 22. (`% 1024` is exact, since
1024 divides 2^32, so there was no bias.)

## What changed

A second generator, splitmix64, used **only** for this noise. One 64-bit
draw supplies six 10-bit indices. `noise_state_` is one `uint64_t` per
`TrainMC`, seeded from two `generator_` outputs in the constructor.
`std::mt19937` still does everything else: seeding games and opening-move
sampling. `generateDirichlet` lost its `const`, because it now advances state.

## Not bit-identical, still reproducible

Max first read "not bit-identical" as "non-deterministic". It is not: the same
seed gives the same games on every run (engine digest `983c2f3d4ef7f5e5`,
repeated). It is a *different* stream from before:

- the noise values differ from the second move onward;
- the generator is shared, so consuming fewer mt outputs (two per `TrainMC` at
  construction instead of one per move) shifts every later mt consumer too.

So the check is statistical, not a digest match:

- **Game digest unchanged** (`d8f9bd2d22ee9731`): move generation untouched.
- **Index uniformity** (`data/dirichlet-rng/chisq_splitmix.cpp`): 10M draws,
  chi-square on each of the 6 slots and on the 5 adjacent-slot pairs (top 5
  bits each), under 4 seeds = 44 tests. 2 exceeded |z| = 2.3, in different slots
  each time, against ~1 expected by chance. There is no repeating slot, so no
  systematic defect.
- **Self-play aggregates**, 40 seeds × 20 games per arm: turns/game 18.49 →
  18.34 (t = −0.45); requests/turn 1274.3 → 1270.4 (t = −0.85).

## Choosing the generator: measured, not assumed

Candidates were compared in isolation (`data/dirichlet-rng/noise_micro.cpp`,
2M calls, 10–40 moves each, min of 7):

| variant | ns / call |
|---|---|
| old: one mt call per move | 40.7–42.6 |
| mt, three indices per call | 32.9–33.3 |
| splitmix64, six per call | 25.4–26.0 |
| xoshiro256** | 25.8 |
| xoshiro256++ | 26.1 |

The last three sit on a floor set by `sum += dirichlet[j]`: a serial chain of
float adds, ~4 cycles each × ~25 moves. At that point the generator no longer
matters. splitmix won on state size (8 bytes against xoshiro's 32) and
simplicity. Plain xoshiro256+ was excluded: its low bits are weak, and the low
bits are exactly what `& 1023` uses.

End to end, three arms interleaved over 24 seeds (`three-arms-24-seeds.txt`),
time per NN request:

| | change | 95% CI |
|---|---|---|
| mt, three indices per call | −1.8% | −2.6 .. −0.9 |
| splitmix64 | −5.8% | −6.5 .. −5.0 |

The final code against HEAD, 40 fresh seeds (`final-40-seeds.txt`): **−3.7%**
(95% CI −4.2 .. −3.2, median −3.5%).

**The two splitmix runs disagree** (−5.8% vs −3.7%, non-overlapping CIs). Same
code apart from comments and a defined seeding order, which gives the same
digest. The CI only covers seed-to-seed variation, not drift in machine state
between sessions. Quote −3.7%, and read the true figure as somewhere in 3.5–6%.

## Why splitmix beat its microbenchmark

In isolation, splitmix saves ~16 ns per evaluation, about 1.7% of the ~870 ns
spent per request. End to end it saved 3.7–5.8%.

**Hypothesis, not verified:** `sizeof(std::mt19937)` is **5000 bytes** here
(624 × 8-byte `uint_fast32_t`). Every 624th call regenerates all of it,
evicting tree nodes from the 48 KB L1. A warm microbenchmark cannot show that.
Cachegrind could not confirm it: arms play different games, so their totals
are not comparable (next section).

## Measurement trap, hit again

The first counter comparison said −34% instructions. It was meaningless: a new
noise stream means the 3 profiled games are *different games* with different
amounts of search. Entry 05 recorded this failure for allocations; it applies
to any change that alters the random stream. What worked:

- normalize to **per NN request**, over many seeds, interleaved;
- isolate the function in a microbenchmark to see the local cost.

Callgrind Ir per request over 8 seeds (`callgrind-ir-per-request.txt`) was too
noisy to use: the per-seed sd (~150 of ~10,000) was as large as the effect.

## Instructions vs time: both directions now seen

- Entry 13 (`lround`): −6.9% instructions gave −3.8% time. The removed code was
  cheap, predictable and high-IPC.
- This entry: 5.9% of instructions removed, and time fell by 3.7–5.8%. The
  removed code had a cost that instructions do not count (plausibly cache
  pollution).

An instruction share is not a time share in either direction.
