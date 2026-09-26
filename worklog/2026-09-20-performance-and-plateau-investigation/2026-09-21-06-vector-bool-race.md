# The `vector<bool>` data race: real, latent, and never observed to fire

*2026-09-21 — branch `perf/engine-optimizations`*

## Summary

`Trainer::is_done_` was a `std::vector<bool>`, written from inside
`#pragma omp parallel for`. That is a genuine data race. This entry establishes
four things:

1. The race is **real** — the same code shape loses updates readily in isolation.
2. It **never fired** in this engine across every configuration tested,
   including one built specifically to maximise its chance.
3. **Why** it never fired: the write compiles to a single non-atomic `orq`, whose
   vulnerable window is a few cycles, and the closest two same-word writes ever
   came was **3,870 cycles**.
4. If it *had* fired, the result would have been a **use-after-free write**, i.e.
   a crash — not silently corrupted training data. This matters for trusting the
   95 generations already trained.

The fix (`std::vector<uint8_t>`) is committed anyway. A race that is merely
improbable is still a race, and this one costs nothing to remove.

## The defect

`corintho_ai/cpp/include/trainer.h` declared:

```cpp
std::vector<bool> is_done_{};
```

and `Trainer::doIteration` wrote it from two parallel regions
(`trainer.cpp`, the training branch and the testing branch):

```cpp
#pragma omp parallel for
for (size_t i = 0; i < games_.size(); ++i) {
  ...
  bool done = games_[i].doIteration(...);
  if (done) {
    is_done_[i] = true;     // <-- race
  }
}
```

`std::vector<bool>` is the standard's bit-packed specialisation: on LP64
libstdc++ it stores 64 flags per `uint64_t`. `is_done_[i] = true` is therefore
not a byte store but a **read-modify-write of a word shared with 63 other
games**. Two threads finishing games in the same word can each read the word,
set their own bit, and write back — and the second write-back erases the first
thread's bit.

Note the asymmetry: a lost update can only ever **clear** a bit that should be
set. It can never spuriously set one, because every writer ORs in its own bit.
So the single failure mode is *a finished game being marked unfinished*.

## Step 1 — is the race real at all?

`data/vector-bool-race/standalone_race.cpp` runs the identical shape (parallel
for, each iteration setting its own index) and counts flags that should be set
but are not. Build with `g++ -std=c++17 -O3 -fopenmp`. 3000 indices, 14 threads,
500 trials:

| schedule | `vector<bool>` lost | `vector<uint8_t>` lost |
|---|---|---|
| `static` (the default, what the engine used) | 3,305 (0.220%) | 0 |
| `static, 1` (interleaved) | 101,294 (6.753%) | 0 |
| `dynamic, 1` | 377,145 (25.143%) | 0 |

So the race is not theoretical. Note that the default `static` schedule is by far
the safest of the three: it gives each thread a *contiguous* chunk, so threads
share a word only at chunk boundaries. The engine got that scheduling by
accident, not by design.

Important caveat about this table: these iterations do no work. Every thread
races through its chunk, so all the boundary writes happen at nearly the same
instant. That is the *opposite* of the real workload, where each iteration is a
full MCTS batch.

## Step 2 — does it fire in the real engine?

Patched `Trainer` with a detector: a `std::vector<std::atomic<uint8_t>>`
shadow written next to every `is_done_[i] = true`, plus a check after each
parallel region for any index where the shadow is set and `is_done_` is not.

Run through `bench/selfplay_bench`, which drives `Trainer` exactly as
`main.pyx` does.

**Production configuration** — 3000 games, 1600 searches, 16 per eval, 14 threads,
three seeds:

```
LOST UPDATES  0        (9,000 games total)
```

**Adversarial configuration** — `searches_per_eval == max_searches`, so there is
exactly one parallel region per turn and every game advances in lockstep. All
games of equal length then finish *in the same region*, which is the necessary
condition for the race:

```
parallel regions                 76
games finished                 3000
same-word pairs in one region  2911     <-- necessary condition met 2911 times
LOST UPDATES                      0
```

Also ran with `schedule(static, 1)` forced (the interleaving that produced a
6.75% loss rate in isolation): still 0.

## Step 3 — validating the detector

A negative result from an unproven detector is worth nothing. Positive control:
deliberately skip the real write for exactly two game indices (100 and 1500) and
confirm the detector notices.

```
same-word pairs in one region  2911
LOST UPDATES                      2     <-- exactly the two dropped writes
```

The detector works. The zeros above are real zeros.

## Step 4 — why it never fires

`g++ -O3 -march=native -S` on `trainer.cpp` shows the flag write compiles to:

```asm
orq  %r10, 0(%rbp)
```

A **single non-atomic** read-modify-store. The vulnerable window is the gap
between that instruction's load and store micro-ops — a few cycles, not the
tens-of-nanoseconds a multi-instruction sequence would give.

So the question becomes: how close in time do two same-word writes actually
land? Instrumented each flag write with `__rdtsc()` and histogrammed the gap for
every same-word pair within a region.

**Adversarial config** (2911 pairs):

| gap (cycles) | pairs |
|---|---|
| 1e3 – 1e4 | 70 |
| 1e4 – 1e5 | 468 |
| 1e5 – 1e6 | 1,794 |
| 1e6 – 1e7 | 579 |

Minimum gap observed, over the whole run: **3,870 cycles**.

**Production config** (3000 games, 1600 searches) — 48 pairs, minimum gap
**172,104 cycles (~43 µs)**.

That is the answer. Even in the configuration built to force collisions, the
two writes are separated by at least ~1,000× the width of the window that could
lose one. In the production configuration the exposure is a further ~1,000×
lower: 48 collision opportunities per 3000-game run instead of 2911, at gaps
two orders of magnitude wider.

Treating each pair as an independent chance of `W / gap` for window `W`, the
observed zero over 2911 adversarial pairs bounds `W` at roughly **≲ 66 cycles**
(95%), consistent with a single cache-missing `orq`. Extrapolating to the real
training history — ~25,000 games/generation × 95 generations — gives an expected
count somewhere in the range 0.1 to a few events, depending on where in that
bound `W` actually sits. So: probably never happened, but not provably never.

Which leads to the question that actually matters.

## Step 5 — what would have happened if it had fired?

Traced the consequence of a wrongly-cleared bit:

1. The trailing `if (!is_done_[i]) return false;` scan makes `doIteration`
   report the round unfinished, so the loop continues.
2. The next region sees `!is_done_[i]` and calls `games_[i].doIteration(...)`
   on an **already-finished** game.
3. `SelfPlayer::endGame()` (`selfplayer.cpp`) has already run, and it does:

   ```cpp
   players_[0].null_root();
   players_[1].null_root();
   to_eval_.reset();        // frees the buffer
   log_file_.reset();
   ```

4. `TrainMC::to_eval_` is a raw `float*` **into** that freed buffer
   (`trainmc.h:180`). With the roots nulled, `TrainMC::doIteration` takes the
   `uninitialized()` branch, allocates a fresh root, and executes:

   ```cpp
   cur_->writeGameState(to_eval_);   // write through a dangling pointer
   ```

   The guarding `assert(to_eval_ != nullptr)` is compiled out under `NDEBUG`,
   which is how both the benchmarks and `setup.py` build.

So the failure mode is a **heap use-after-free write** of `kGameStateSize`
floats: a crash or an allocator abort, not a quietly wrong training sample.

**This is the reassuring part.** The worry with a race in a training pipeline is
silent data corruption poisoning a run you then trust. That is not the shape of
this bug. If it had fired during the 95 generations, the generation would have
died loudly. It did not. The existing training data is not suspect on account of
this race.

(It is still suspect on account of the line-breaking defects — see entry 04.
Those are unrelated and were real.)

## The fix

```cpp
std::vector<uint8_t> is_done_{};
```

Byte-granular, so each write is a plain store with no read-modify-write and no
shared word. Applied to `Trainer` and to `Tourney`, which had the identical
declaration (`tourney.h:42`). `Tourney`'s writes are single-threaded today, but
the declaration is the trap, so it goes too.

This does not make the code formally race-free under the C++ memory model —
concurrent unsynchronised writes to distinct bytes of the same object are still
technically UB, and the bytes still share cache lines (false sharing). It does
remove the mechanism by which a write could actually be lost on any real
hardware. A fully clean fix would be `std::vector<std::atomic<uint8_t>>` or
`std::atomic_ref`; that was not done because the measurements above show no
contention worth paying for, and the serial scans that read this array would
then need care.

### Cost

Interleaved, pinned single-thread A/B (arm A = `vector<bool>` at HEAD,
arm B = working tree), 5 reps each, alternating:

| games | `vector<bool>` median | `vector<uint8_t>` median | delta | arm spreads |
|---|---|---|---|---|
| 50 | 1.334 s | 1.354 s | +1.5% | 4.8% / 3.0% |
| 500 | 21.771 s | 21.735 s | −0.17% | 1.9% / 2.5% |

Both deltas sit inside the arms' own spreads, so neither is a real difference.
The 500-game figure is the one to trust: the array is scanned twice per
parallel region with length equal to the game count, so if the 8× size increase
were going to cost anything it would show up more at 500 games than at 50, and
it does not. **Perf-neutral.**

Raw runs: `data/vector-bool-race/ab-50games-st.txt`,
`data/vector-bool-race/ab-500games-st.txt`.

### Gates

Clean rebuild, no warnings:

```
digest_game    d8f9bd2d22ee9731   (unchanged)
digest_engine  ad85cefa8bf9abf5   (unchanged)
verify         PASS, 200000 positions identical
rulecheck      PASS, 59405 comparisons, 0 mismatches
```

Digests unchanged is the expected result: they are recorded single-threaded, and
the race could not affect them.

## Method notes for the next session

The detector was scratch and is not committed; it was three edits to
`trainer.h`/`trainer.cpp` under `#ifdef CORINTHO_RACE_DETECT`:

- a `std::vector<std::atomic<uint8_t>> shadow_done_`, stored to immediately
  before each `is_done_[i] = true`;
- a `checkRace()` called at the end of each parallel region, counting indices
  where the shadow is set and the flag is not (and repairing them, so one lost
  bit is reported once rather than every subsequent round);
- `__rdtsc()` at each write, to histogram same-word gaps.

Two lessons worth keeping:

- **Always build a positive control before believing a null result.** The first
  three runs reported zero lost updates and it would have been easy to stop
  there. The zero only became evidence once dropping two writes on purpose
  produced exactly `LOST UPDATES 2`.
- **A negative needs a mechanism.** "We ran it and it didn't happen" is weak.
  "The window is a few cycles and the closest writes were 3,870 cycles apart" is
  a reason, and it also predicts the conditions under which the answer would
  change — more threads, shorter games, or a `dynamic` schedule would all narrow
  the gaps.

## Status of the outstanding list

- [x] `vector<bool>` race — fixed, and shown not to have affected past training.
- [ ] `to_eval_` oversizing (`selfplayer.cpp:24`, `match.cpp:18`): 448 KB/game vs
      4.5 KB needed. Now the top remaining item, and more pressing since the
      arena raised peak heap 16.9%.
- [ ] `scripts/bash/build.sh` still never build-verified with `-flto`.
- [ ] Strength impact of the line-breaking fix — needs a fixed-weights match.
- [ ] `web/engine.js`, `web/line_breakers.js` still carry every line-breaking defect.
- [ ] **The spreadsheet's last column is still unidentified** (0.574 → 0.935 over
      78 generations). Open since entry 01. If it is the second-player self-play
      score, colour imbalance outranks every performance item on this list.
