# Corintho AI — Performance & Playing-Strength Plan

Branch: `perf/training-overhaul`
Status: plan only, nothing implemented.

**See also:** `worklog/` for what actually happened and in what order, including
measurements, corrections and rejected approaches. `bench/README.md` for the
benchmark harness, the measured baseline and the commit convention for
optimizations. When this document and the worklog disagree, the worklog records
what was true at the time and this document records what we believe now.

Ultimate goal: **a stronger trained network**, not a faster program. Speed matters
only as a means — more generations per hour, or more searches per move at equal
cost. Every change below is judged by that standard.

---

## 1. What the last run actually tells us

Evidence pulled from `generations/gen_*/` (95 generations, run ending 2023-07-22)
and the per-generation spreadsheet.

### 1.1 The rating curve plateaued at generation ~61

| Gens | Rating | Behaviour |
|---|---|---|
| 0–13 | 100 → 1388 | Fast climb |
| 14–61 | 1414 → 5497 | Steady climb |
| 62–93 | 5477 – 5537 | **Flat. 32 generations, no net progress.** |

### 1.2 But the training loss plateaued at generation ~13

Loss from the spreadsheet: 1.75 → 1.00 by gen 13, then flat at 1.04–1.13 for the
next 55 generations. It never improves again.

This is the single most important fact in the dataset. **The network stopped
fitting its own data 48 generations before the rating stopped moving.** Two
possible readings, and they have very different remedies:

- (a) The data kept getting better but the network could not represent it —
  a capacity or trainability ceiling.
- (b) The network was already at the data's noise floor, and the rating climb
  from gen 14–61 was substantially an artifact of the rating chain (see 1.3).

Resolving (a) vs (b) is the first experiment, not a later one. See Phase 0.

### 1.3 The rating number is self-referential and cannot be trusted

`update_rating` (`main.pyx:263`) computes
`new_rating = best_gen_rating - 400*log10(1/score - 1)` — a cumulative chain
against the previous best, with no fixed anchor. A run of noisy 0.52 promotions
inflates it monotonically. "5537 Elo" is not an absolute quantity.

Corroborating: test scores for gens 79–93 are 0.483, 0.493, 0.514, 0.509, 0.502,
0.521, 0.508, 0.511, 0.525, 0.502, 0.483, 0.489, 0.493, 0.528, 0.501. At 1600
test games (σ ≈ 0.0125) and a promotion threshold of 0.52, promotion fires at
roughly 1.6σ. **Late-run promotions are consistent with pure noise.**

`corintho_ai/rating/` already contains round-robin tournament tooling
(`tourney.pyx`, `round.py`, `rating_gd.cpp`). That is the correct instrument and
it is not being used in the training loop.

### 1.4 Self-play games are almost entirely decided by who moves first

Generation 93 test results (`generations/gen_93/testing_logs/score_verbose.txt`):

```
First player  wins 198/800 (24.75%)  draws 10/800 (1.25%)  losses 592/800 (74%)
Second player wins 596/800 (74.5%)   draws  5/800 (0.63%)  losses 199/800 (24.9%)
```

The spreadsheet's final unlabeled column runs 0.574 → 0.935 over 78 generations,
monotonically. **If that column is the second-player score in training self-play,
it is very likely the root cause of the plateau** — by gen 78, ~93% of training
games are won by the second player. The value head is then predicting a near
constant, and nearly every position the first player sees is objectively lost, so
the policy target carries little discriminating signal. Learning stalls not
because the optimizer failed but because self-play stopped producing informative
games.

> **Open question for the developer — please confirm what that last column is.** The whole
> priority ordering below changes depending on the answer. If it is the
> second-player self-play score, §5.4 becomes the highest-value work in this
> document, ahead of all performance work.

### 1.5 Where the wall-clock goes

Generation 93, measured (`training_logs/play_time.txt`, `fit_time.txt`):

| Phase | Time | Share |
|---|---|---|
| Self-play total | 40m21s | — |
| — of which NN prediction | 24m58s | **62%** |
| — of which C++ self-play | 14m47s | 37% |
| Network fitting | 9m23s | — |
| Testing total | 6m54s | — |
| — of which NN prediction | 6m20s | **92%** |
| **Generation total** | **~48 min** | |

78 generations ≈ 62 hours.

The headline: **the C++ MCTS engine is not the bottleneck — the Keras prediction
loop is.** 4539 predict calls at 0.533s each in training; 3541 calls at 0.117s
each in testing, where the batch is small enough that per-call framework overhead
dominates almost completely. This reorders the optimization list from my initial
audit: Python-side inference work outranks the C++ node allocator.

---

## 2. Success metric (decide before writing any code)

Replace the chained rating with:

1. **Anchored Elo.** Freeze a set of anchor opponents — say gens 0, 20, 40, 61,
   93 — and rate every new model in a round-robin against them using the existing
   `corintho_ai/rating/` tooling. Anchors never change, so ratings stay comparable
   across the whole project.
2. **Elo per wall-clock hour**, not Elo per generation. A 2× faster generation
   that gains slightly less per generation is still a win.
3. **Elo at matched node count and at matched wall-clock.** A smaller network that
   is weaker per-node but 3× faster may be stronger per second. Both numbers
   must be reported or architecture comparisons are meaningless.
4. Sample size: 1600 games gives σ ≈ 1.25%, i.e. ±~9 Elo at even strength.
   For detecting a 20-Elo change, budget **≥5000 games** per comparison and
   report a confidence interval. The current 0.52 threshold on 1600 games is not
   a decision rule, it is a coin flip.

---

## 3. Phase 0 — Benchmarking harness (do this first, change nothing else)

No optimization lands before this exists. Everything after depends on being able
to detect a regression.

1. **Microbenchmarks (C++, Google Benchmark or a simple timing harness).**
   Separate `Release` CMake target — the current `CMAKE_CXX_FLAGS` is
   `--coverage -g -pg -O3` (`CMakeLists.txt:14`), so any timing taken through the
   test build is meaningless.
   - `Game::getLegalMoves` on a corpus of ~10k positions sampled from real game logs.
   - `Game::doMove`, `Game::writeGameState`.
   - `Node` construction + destruction.
   - Full `TrainMC::search()` for a fixed tree and fixed RNG seed.
2. **End-to-end throughput benchmark.** One fixed-seed generation at reduced
   scale (e.g. 500 games instead of 25000), reporting the same split as
   `play_time.txt`: self-play vs predict vs fit. Target runtime ~2 minutes so it
   can run on every change.
3. **Determinism / correctness harness.** Fixed seed → fixed sequence of moves and
   node counts. Any optimization that changes the search tree must be flagged
   deliberately, not discovered later. Existing `tests/cpp/` is the starting point.
4. **Strength harness.** Scripted round-robin against the frozen anchors, emitting
   Elo + CI. Wraps `corintho_ai/rating/`.
5. **Record the baseline.** Commit the numbers for current `main` so every later
   claim is relative to a stored measurement, not a memory.
6. **Profile before optimizing.** `perf record` on the self-play phase to confirm
   the C++ hot spots below are actually hot. My audit predicts `getLegalMoves`
   and the allocator dominate; the profile decides.

---

## 4. Phase 1 — Correctness bugs (cheap, and they cost playing strength)

These are not performance items. They are wasted compute and degraded search, and
several may be contributing directly to the plateau. Do them first — they are
small, and they change the baseline that everything else is measured against.

| # | Location | Bug | Consequence |
|---|---|---|---|
| 1.1 | `trainmc.cpp:299` | `int32_t max_prob = 0` compared against a float probability in (0,1]; assignment truncates to 0 so the condition is always true | `chooseHighProbMove` returns the **last** legal move, never the highest-probability one. Used as the fallback at `:368` and `:436` |
| 1.2 | `trainmc.cpp:518` | `if (cur->drawn())` should be `cur_child->drawn()` | Draw deduction is wrong → `kDeducedDraw` vs `kDeducedLoss` misclassified, corrupting terminal propagation |
| 1.3 | `main.pyx:198-201` | `np.concatenate(...)` return value never assigned | **The replay buffer does nothing.** Old-generation samples are loaded, concatenated, and thrown away. Training uses only the current generation |
| 1.4 | `trainer.cpp:192,226` | `is_done_[i] = true` on a `std::vector<bool>` inside `#pragma omp parallel for` | Bit-packed concurrent writes = data race, lost updates. Change to `std::vector<uint8_t>` |
| 1.5 | `wrapper.py` (`ModelCheckpoint`) | `save_frequency=1` is not a Keras argument (it is `save_freq`) | Silently ignored |
| 1.6 | `play_setup.py:19` | No `-O` flag; stale source paths (`../cpp/node.cpp`, should be `../cpp/src/`) | Builds unoptimized, probably does not build at all |

**1.3 deserves emphasis.** With `num_old_gens: 2` in the config and the
concatenation discarded, the effective replay window is a *single generation*.
AlphaZero-style training relies on a sliding window of many generations to
stabilize targets. This is a strong plateau suspect independent of everything
else, and it is a three-line fix.

---

## 5. Phase 2 — Playing strength

Ordered by expected effect on strength. This section matters more than Phase 3.

### 5.1 Dirichlet noise is applied at every node, not just the root

`TrainMC::receiveEval` (`trainmc.cpp:268-283`) calls `generateDirichlet` +
`setProbs` for **every** evaluated node. AlphaZero applies exploration noise
*only at the root* of the search. Injecting ε=0.25 of Dirichlet noise at every
interior node persistently corrupts the priors the search relies on, degrading
every playout rather than just diversifying the root.

This is probably the largest single strength bug in the engine. Fix: apply noise
only when `cur_ == root_`. Expect a measurable Elo jump at fixed node count.

### 5.2 Fix the replay window (Phase 1.3) and then tune it

Once concatenation actually works, sweep the window size. `num_old_gens: 2` is
small by AlphaZero standards; typical is 10–20 generations of sliding window.

### 5.3 Replace the "+1.0 default evaluation" hack with virtual loss

`TrainMC::search` (`trainmc.cpp:534-545`) gives every visited node a default
evaluation of +1.0 to stop the batched search from re-visiting the same node
`searches_per_eval` times in a row. The comment documents this as an empirical
fix for exactly the collision problem that **virtual loss** solves properly.
Virtual loss is the standard mechanism, is better understood, and does not bias
the evaluation of every node in the tree. Worth an A/B.

### 5.4 The second-player-advantage problem

If §1.4's reading is right, this is the main event. Options, roughly in order of
invasiveness:

- **Measure it properly first.** Plot first-player score in training self-play
  across generations; confirm it is the spreadsheet column.
- **Opening randomization.** `kNumOpeningMoves = 6` (`trainmc.h:150`) with
  visit-proportional sampling is the only diversity source. If games are 93%
  decided, widen it: more opening plies at temperature 1, or seed games from a
  book of balanced/random legal openings.
- **Balanced game pairs.** Play each sampled opening twice with colours swapped,
  so the value target is not dominated by parity.
- **Check whether the game is simply a second-player win.** If Corintho is solved
  in the second player's favour at this depth, the ceiling is structural and the
  goal should be restated as "play the first side as well as possible", with
  evaluation weighted accordingly.

### 5.5 Regularization is very likely too strong

`wrapper.py` uses `regularizers.L1L2()` with **default arguments**, i.e.
`l1=0.01, l2=0.01`, on every one of the 12 Dense kernels. For a ~133k-parameter
network that is heavy L1 — enough to drive a large fraction of weights to zero
and to put a floor under the loss. This is a strong candidate explanation for the
loss plateauing at ~1.05 from gen 13 onward and never moving again.

Experiment: re-fit gen 93's data with `l1=0, l2=1e-4` and see whether the loss
floor drops. This is a one-line change and a cheap, decisive test of §1.2's
reading (a) vs (b).

### 5.6 Training-loop hygiene

- `validation_split=0.3` holds out 30% of samples from training. Keras splits
  *before* shuffling, so the validation set is the last 30% of games — correlated
  with each other, and containing near-duplicate positions from the same games as
  the training set. `val_loss` is therefore optimistic, which makes
  `ReduceLROnPlateau` and `save_best_only` unreliable. Reduce to ~5% and split by
  game, not by row.
- `loss_weights=[1.0, 0.25]` down-weights the policy head 4×. AlphaZero weights
  value and policy roughly equally. Worth an A/B.
- Learning rate annealed to `5e-06` by gen 87 — effectively frozen. Combined with
  a plateau-triggered schedule driven by an unreliable `val_loss`, the LR schedule
  may have shut down learning prematurely. Consider a fixed schedule or warm
  restarts.

---

## 6. Phase 3 — Architecture

Current model (`wrapper.py:255-270`): input 70 → **12 ×
[Dense(100) → ReLU → BatchNorm]** → `tanh` value head (1) + `softmax` policy head (96).

Parameter count (arithmetic from the layer definitions, to be confirmed against
the saved model):

| Component | Params |
|---|---|
| Input Dense 70→100 | 7,100 |
| 11 × Dense 100→100 | 111,100 |
| 12 × BatchNorm(100) | 4,800 |
| Value head 100→1 | 101 |
| Policy head 100→96 | 9,696 |
| **Total** | **~132,800** |

The developer's instinct is that this is too large. My read is more specific: **it is not
too large so much as badly shaped.**

1. **12 plain layers with no residual connections.** This is the real problem.
   At depth 12 without skip connections, gradients degrade and the network is
   hard to train — which matches a loss that stops improving at generation 13.
   Residual blocks are the standard fix and would likely let it train deeper
   *and* better.
2. **`Dense → ReLU → BatchNorm` is the wrong order.** Standard is
   `Dense → BatchNorm → ReLU`. Normalizing after the activation discards the
   centring that BN is supposed to provide going into the non-linearity.
3. **No spatial inductive bias.** The input is a flat 70-vector, but the board is
   4×4 with 4 bits per square, and the win condition is *lines* — an inherently
   spatial, translation-related pattern. A small convolutional tower over a
   4×4×4 input would encode this directly instead of forcing a dense net to
   rediscover the geometry from scratch. This is likely worth more than any
   change to width or depth.
4. **Symmetry is handled by 8× data augmentation** (`selfplayer.cpp:78-103`),
   which multiplies fitting cost 8×. With a convolutional or symmetry-aware
   architecture some of that could be structural instead of sampled.
5. **Size vs speed.** Since prediction is 62% of generation wall-clock, shrinking
   the network buys throughput directly. But it must be judged on
   **Elo per second**, not Elo per node — see §2.3.

Proposed sweep, all measured against the same anchors:

| Variant | Shape | Rationale |
|---|---|---|
| A (baseline) | current 12×100 | Control |
| B | 12×100, BN-before-ReLU, l1=0 | Isolates the two suspected training bugs |
| C | 6 residual blocks × 128 | Tests depth-with-skips |
| D | 4×256 dense | Tests "shallower and wider" |
| E | small conv tower on 4×4×4 | Tests spatial inductive bias |
| F | deliberately tiny (~30k params) | Tests whether Elo/second favours small |

Run each for a fixed, small number of generations (5–10) from a common starting
point, and compare anchored Elo and Elo/hour. Do **not** run a full 78-generation
race per variant — at 48 min/generation that is 62 hours each.

---

## 7. Phase 4 — Performance

Now the original audit list, re-ordered by the measured profile from §1.5.

### 7.1 Python / inference — 62% of training, 92% of testing

- **Replace `model.predict()` with a `tf.function` on a fixed input signature**
  (`main.pyx:74-81`). `predict()` carries large per-call overhead (tf.data
  wrapping, callback machinery, retrace checks). The testing numbers — 0.117s per
  call on a small batch — are nearly pure overhead. Use `model(x, training=False)`
  under `@tf.function`, or `jit_compile=True` for XLA.
- **`verbose=1` prints a progress bar on every one of ~4500 calls.** Set to 0.
- **Slice the input.** The full `num_games × searches_per_eval` array is passed
  every call regardless of how many rows are valid; pass `game_states[:num_requests]`.
- **Pipeline predict against self-play** (`main.pyx:135-160`). The loop is
  strictly serial: play → predict → play. Split the games into two halves and
  double-buffer so the C++ engine plays batch A while the model predicts batch B.
  **Value depends entirely on the hardware:** on a GPU box this is close to a free
  1.5×, because the two phases use different silicon. On the CPU-only cloud box
  both phases contend for the same cores and the gain is much smaller. (Corrects
  an earlier claim of ~1.6× that assumed separate hardware.)
- Consider TFLite / ONNX / a plain NumPy forward pass for inference — at ~133k
  params with no convolutions, the Keras graph may cost more in overhead than the
  arithmetic.

### 7.2 C++ engine — 37% of training

- **Arena allocator for `Node`** (`node.cpp:273`, `trainmc.cpp:566`). Two mallocs
  per node (64-byte over-aligned `Node`, plus `Edge[]`), from many OpenMP threads,
  contending on the allocator. Per-`SelfPlayer` slab with the edge array inline;
  tree teardown becomes a pointer reset instead of the recursive destructor at
  `node.cpp:19-23`.
- **`to_eval_` is oversized ~100×** — **now a hard blocker for local work, see §10.2**
  (`selfplayer.cpp:24`, `match.cpp:18`):
  `kGameStateSize * max_searches` = 448 KB per game, but only `searches_per_eval`
  (16) slots are ever used = 4.5 KB. At 25000 games this is ~11 GB of buffer for
  ~110 MB of live data. Large cache and memory win.
- **`Game::getLegalMoves`** (`game.cpp:28-41`) recomputes per-square state ~150×
  per call: 96 `Move` constructions from ID plus repeated `top()`/`bottom()` from
  both the move loop and the line scanners. Cache a 16-entry
  `{top, bottom, empty, frozen}` table once per position (or extract `board_` to a
  `uint64_t` and use nibble-parallel ops), and drive the move loop from a
  `static constexpr` move table. Called from every `Node` constructor.
- **`getFilteredProbs` scans all 96 moves** (`trainmc.cpp:205-226`) when the edges
  already hold the legal move IDs — replace with a direct
  `filtered[j] = probs[move_id(j)]` loop over `num_legal_moves`.
- **`doMove` clears 16 frozen bits one at a time** (`game.cpp:66-71`) — single
  mask AND instead.
- **`writeGameState` writes 64 floats bit by bit** (`game.cpp:44-57`) — nibble LUT
  + `memcpy`.
- **OpenMP scheduling** (`trainer.cpp:176,217`): default static schedule over
  games with highly variable per-game work → stragglers. Use `schedule(dynamic,1)`.
  Also consider hoisting the fork-join out of the per-iteration loop.
- **VLAs zero-filled every iteration** (`trainer.cpp:170,207`, `tourney.cpp:57`):
  `int32_t offsets[games_.size()] = {0}` stack-allocates and memsets `num_games`
  ints on every `doIteration`. Hoist to a member vector.
- **`line_breakers` is built from 102 string literals at static init**
  (`util.h:85`) and is non-`const`. Make it `constexpr`.

### 7.3 Build flags

- `python/setup.py:26-31` is `-O3 -std=c++17 -fopenmp -DNDEBUG`. Add
  `-march=native -mtune=native`, `-flto`, `-fno-semantic-interposition`.
  Evaluate PGO — one stable hot loop makes it a good fit. Test `-ffast-math`
  separately and verify the search does not drift.
- Add a non-instrumented `Release` CMake config (see Phase 0.1).
- Cython directives are missing: `# cython: language_level=3str` in `setup.py` is
  an inert comment. Pass
  `compiler_directives={"language_level":"3","boundscheck":False,"wraparound":False}`.

### 7.4 Deferred / larger bets

- **Transposition table.** No position hashing anywhere; each `Node` stores a full
  `Game` and transpositions are re-searched. Corintho transposes heavily because
  move-moves commute. Biggest algorithmic win available, biggest cost.
- **Batched tree descent** with virtual loss, to raise NN batch size without
  raising game count.
- `Trainer::num_requests` and `writeRequests` both re-walk all games computing the
  same counts (`trainer.cpp:39,80`) — fuse.

---

## 8. Sequencing

1. **Phase 0** — harness + baseline. Nothing else starts first.
2. **Phase 1** — correctness bugs. Small, and they move the baseline.
3. **§5.5 regularization test** — one line, decisively resolves "capacity ceiling
   vs noise floor" and steers everything after it.
4. **§5.1 root-only Dirichlet + §5.2 replay window** — the two most likely
   strength wins.
5. **§7.1 inference** — biggest wall-clock win, makes every later experiment cheaper.
6. **Phase 3 architecture sweep** — now affordable, because of step 5.
7. **§7.2 C++ engine** — real, but only 37% of the time; do it when it stops
   being the cheap win.

Steps 3–4 are cheap enough to run concurrently with 5 if there is machine capacity.

---

## 9. Open questions

1. **What is the last column of the spreadsheet?** (§1.4). Changes the priority order.
2. Is there a reason `epsilon` noise was applied at every node rather than the
   root — a deliberate experiment, or an oversight?
3. Was the `num_old_gens: 2` window chosen deliberately, or picked while the
   replay code was silently broken?
4. What hardware is available for the architecture sweep? At 48 min/generation the
   sweep in §6 is the dominant cost of this whole plan.
5. Is `train_94.zip` (2 GB) worth unpacking for the full loss history, or is the
   spreadsheet the authoritative record?

---

## 10. Local development on the RTX 3060 laptop

### 10.1 The hardware

| | |
|---|---|
| GPU | RTX 3060 Laptop, **6 GB VRAM** |
| CPU | i7-12700H, 14 cores (6P + 8E), 20 threads |
| RAM | 15 GB total, **~9 GB available** |
| Disk | 627 GB free |

**RAM is the binding constraint, not the GPU.** The GPU is more than adequate for
a 133k-parameter model; system memory is what caps the number of concurrent
self-play games.

### 10.2 The `to_eval_` bug is now a blocker, not an optimization

`SelfPlayer` allocates `kGameStateSize * max_searches` floats per game
(`selfplayer.cpp:24`) = **448 KB per game**, of which only `searches_per_eval`
(16) slots are ever used = 4.5 KB.

At the cloud config of 25,000 games that is **~11 GB of buffer alone** — more
than this laptop has available. The old run only survived it because the cloud
box had more memory. **A 25,000-game generation cannot run locally until this is
fixed.** After the fix the same config needs ~112 MB.

Add to that the MCTS node arena (25,000 games × up to 1600 nodes × ~64 B, plus a
separate `Edge[]` allocation per node and per-allocation malloc overhead — easily
5–8 GB at peak) and the 8×-expanded sample arrays (~3.8 GB, §7.4). Local runs
must be scaled down regardless.

**Move this to Phase 1. It gates everything local.**

### 10.3 Realistic local scale

Target **~4,000 games per generation** rather than 25,000. Rough projection from
the gen-93 cloud timings, adjusted for 14 cores and a GPU:

| Phase | Cloud (25k games, 32 vCPU, no GPU) | Local (4k games, 14 cores + 3060) |
|---|---|---|
| Self-play | 14m47s | ~4–5 min |
| NN predict | 24m58s | ~1–2 min |
| Fit | 9m23s | ~1–2 min |
| **Total** | **~48 min** | **~8–10 min** |

Projections, to be replaced by Phase 0 measurements. If they hold, a 10-generation
experiment is ~1.5 hours — fast enough to iterate properly.

### 10.4 The GPU flips the bottleneck back to C++

On the cloud box NN prediction was 62% of wall clock. With a 3060 that collapses,
and the 14-core self-play engine becomes dominant. Consequences:

- The **C++ optimizations (§7.2) get promoted** for local iteration speed —
  arena allocator and `getLegalMoves` now directly control how fast experiments run.
- **Pipelining (§7.1) becomes genuinely valuable again**, because GPU and CPU are
  separate silicon.
- **TFLite/quantization (§7.1) is irrelevant locally** but still matters if the
  final run goes back to a CPU cloud instance.
- ⚠️ **Timing conclusions do not transfer between the laptop and a CPU cloud box.**
  Strength conclusions do. Keep the two kinds of result clearly separated, and
  consider a GPU cloud instance for the final run so the profile matches.

### 10.5 The old training samples are gone

`train_94.zip` (2 GB) is **truncated — not a valid zip archive**. No `.npz` sample
files survive anywhere in the repo except a tiny toy run under
`corintho_ai/python/logs/`. Models and logs survived; training data did not.

So the cheapest experiments need a dataset regenerated first: **one self-play pass
using gen 92, dumped and frozen**. That is a single generation of cost and then
serves every offline experiment for free. Do it early.

### 10.6 The anchor is gen 92, not gen 93

`generations/gen_94/metadata.txt` records `best_generation: 92`. Gen 93 scored
0.5009 and was not promoted. **Gen 92 is the reigning champion** and is the model
to beat.

This is the right success metric and it dissolves the §1.3 problem: "does the new
run's plateau beat gen 92 head-to-head" is an absolute question with an absolute
answer, unlike the chained rating.

---

## 11. Experiment tiers

The constraint is that a full run is expensive and a plateau takes ~60
generations to reveal itself. So: push as much learning as possible into
experiments that need **no training loop at all**.

### Tier A — frozen dataset, offline re-fits (minutes each, GPU)

Regenerate one dataset with gen 92 (§10.5), freeze it, then re-fit variants on
identical data. No self-play, no search. Pure supervised comparison.

Answers:
- **Is the loss floor of ~1.05 a regularization artifact?** (§5.5 — `l1=0.01`.)
  This is the single highest-information experiment in the plan and costs minutes.
- Does `Dense→BN→ReLU` beat `Dense→ReLU→BN`? (§6.2)
- Do residual blocks fit the same data better than 12 plain layers? (§6.1)
- Do conv layers over a 4×4×4 input beat dense? (§6.3)
- What do `loss_weights` and `validation_split` do to the fit? (§5.6)

**Caveat:** a better fit on fixed data is necessary but not sufficient for more
Elo — the data itself is a product of the old plateaued policy. Tier A prunes bad
ideas cheaply; it does not confirm good ones.

### Tier B — frozen weights, search A/Bs (minutes each, no training)

Load gen 92, change only the search, play it against unmodified gen 92. Measures
Elo per change at fixed node count with zero training cost.

- **Root-only Dirichlet noise** (§5.1) — expected to be the biggest single win.
- `chooseHighProbMove` fix (§4, bug 1.1).
- Terminal-propagation fix (§4, bug 1.2).
- `c_puct` sweep — currently 3.0, never re-tuned.
- `max_searches` sweep — Elo per second at 400 / 800 / 1600.
- Virtual loss vs the `+1.0` default-eval hack (§5.3).

Tier B is the best value in the whole plan: real Elo numbers, no training loop,
and it validates the Phase 1 bug fixes immediately.

### Tier C — short training runs (~1.5 h each at 4k games × 10 gens)

Two distinct questions, needing two different setups:

- **Warm start from gen 92** → "does this change *unstick* the plateau?" Cheap and
  directly on-target. Note gen 92 sits at LR 5e-6 in a deep basin, so the LR
  schedule must be reset or the answer is always "no".
- **From scratch** → "does this change produce a *better* plateau?" Architecture
  changes cannot warm start, so they must go this route.

⚠️ At 4,000 games the gradient noise and effective replay window differ from
25,000. Findings about architecture and search should transfer; findings about
replay-window size, batch size and LR schedule may not. Re-verify those at scale.

### Tier D — the final full run

One from-scratch run with everything that survived Tiers A–C. Judged by whether
its plateau beats gen 92 head-to-head.

Protocol for the final comparison:
- ≥5,000 games, colours alternated (σ ≈ 0.7%, ~±5 Elo).
- Report Elo ± CI at **both** equal node count and equal wall-clock, since the new
  network may be a different size.
- Decide the venue first: a GPU cloud instance matches the laptop's profile;
  a CPU instance matches the old run's. They favour different architectures.

### Suggested order

1. Phase 0 harness + Phase 1 bug fixes (`to_eval_` first — it gates local runs).
2. Regenerate and freeze the gen-92 dataset.
3. **Tier B** search A/Bs — fast, high-signal, validates the bug fixes.
4. **Tier A** regularization test, then the architecture sweep.
5. **Tier C** on the two or three survivors.
6. **Tier D** once.

---

## 12. Search and game-count scheduling

The developer's proposal: ramp `max_searches` and `num_games` upward over the course of a
run rather than holding them fixed at 1600 / 25,000.

### 12.1 Why this is principled, not just a speed trick

AlphaZero-style learning works because **the MCTS visit distribution is a better
policy than the raw network prior**. The network is trained to imitate its own
search. The learning signal is the *gap* between the two.

When that gap shrinks to noise, training stops — regardless of how much data you
generate. Increasing the search budget widens the gap again. So ramping searches
is a direct attack on the plateau mechanism, not merely a compute optimization.

The run's own history is consistent with this: the loss fell fast through gen 13,
then sat at ~1.05 for 55 generations. A flat loss with a flat rating is what
"search no longer teaches the network anything new" looks like.

Cost asymmetry also favours ramping. Early generations, with a near-random
network, gain little from deep search — a bad prior searched deeply is still bad.
Running gens 1–20 at 200 searches instead of 1600 is roughly 8× cheaper in
self-play and prediction, which at ~48 min/generation saves on the order of
**10+ hours** over the early run, and the saved time buys more generations later
where depth actually matters.

### 12.2 The diagnostic that should come first

Before designing a schedule, **measure the policy-improvement gap** — the KL
divergence between the network prior and the root visit distribution, as a
function of generation and search count.

This is a Tier B experiment: frozen gen 92 weights, no training, minutes to run.
It discriminates between the two competing plateau explanations:

- **KL is small at 1600 searches** → search has stopped improving on the prior.
  Ramping searches is exactly the right lever.
- **KL is large** → search is still producing better targets and the network is
  failing to fit them. The problem is capacity / regularization (§5.5), and more
  searches will not help.

Sweep search count (200 / 400 / 800 / 1600 / 3200) against gens 13, 40, 61 and 92
to see how the gap has evolved. This is probably the single most informative
cheap experiment available, alongside the `l1` test.

⚠️ Note this competes with the second-player-dominance explanation (§1.4/§5.4).
If ~93% of games are decided by parity, the *value* target is near-constant no
matter how deep the search goes. Ramping searches addresses the policy signal,
not the value signal. Both diagnostics should be run before committing.

### 12.3 The trap: evaluation search count must be frozen

**The training search count and the evaluation search count must be separated.**

Currently both players in the promotion match use `max_searches` from the same
params (`wrapper.py`, passed to both `Trainer` instances). If generation N trains
at 400 searches and generation N+1 at 800, and the match is played at the new
value, the result measures the *search budget*, not the network.

Worse, it would make ratings non-comparable across the run and would produce
exactly the kind of spurious improvement that §1.3 already warns about.

**Requirement:** add a separate `eval_searches` parameter, fixed for the entire
run (and for the final gen-92 comparison), independent of the training schedule.
This is a prerequisite for any scheduling work.

### 12.4 Playout cap randomization — probably better than a plain ramp

KataGo's approach, and worth preferring to a simple global ramp: give **most**
positions a small search budget, and a small fraction (~25%) the full budget,
**training the policy head only on the full-budget positions** while using all
positions for the value head.

This decouples the two needs:
- Value head wants **many games** — more independent outcomes, lower variance.
- Policy head wants **deep search** — better targets, but only on some positions.

A global ramp forces one tradeoff for both. Cap randomization gets both, and it is
one of the larger reported efficiency wins in KataGo's training. It requires
tagging samples with their search budget so the loss can mask accordingly —
a moderate change to `Sample` and `writeSamples`.

### 12.5 Practical schedule design

Prefer **adaptive over calendar-based**: step the search budget up when the
policy-improvement gap (§12.2) or the promotion rate falls below a threshold,
rather than on a fixed generation number. This targets the actual stall and
avoids paying for depth before it is useful.

A reasonable starting shape, to be tuned:

| Phase | `max_searches` | `num_games` | Rationale |
|---|---|---|---|
| Early (gens 1–~15) | 200–400 | high | Weak prior; breadth beats depth; very cheap |
| Middle | 800 | high | Ramp as the gap narrows |
| Late | 1600 → 3200+ | high | Depth is the remaining lever |

Keep `num_games` high throughout rather than trading it away for depth — cutting
games raises gradient variance, and §12.4 is the better way to buy depth.

### 12.6 Implementation notes and constraints

- **Where the schedule lives.** `main.py:44-56` builds one identical command per
  generation and `os.system`s them, so all generations get the same flags. The
  schedule must be computed inside `wrapper.py` from `current_generation`, which
  `setup_existing_run` already reads. Straightforward.
- **`visits_` is `int16_t`** (`node.h:164`), capping a node at 32,767 visits. Fine
  for a ramp to 3200, but it is a hard ceiling — and `set_visits` uses
  `narrow_cast`, so overflow would be silent. Add an assert if the ramp goes high.
- **`searches_per_eval` must stay below `max_searches`** (asserted in the
  `Trainer` constructor, and clamped in `wrapper.py:155`). A low early
  `max_searches` shrinks the NN batch, which on GPU means worse utilization —
  early generations may want a *larger* `searches_per_eval` to compensate.
- **Memory scales with `num_games`**, so on the laptop the ramp interacts with
  §10.2. Fix `to_eval_` first.
- **Opening diversity.** `kNumOpeningMoves = 6` (`trainmc.h:150`) is the only
  source of game variety. At 25,000 games with 6 temperature-1 plies, duplicate
  games are already plausible; raising `num_games` further amplifies it. Worth
  measuring the count of *distinct* games in the existing logs before scaling up.
- **Record the schedule per generation** in `metadata.txt` so later analysis can
  separate "the network got better" from "the search got deeper".

---

## 13. Deep dive: bitboard move generation and shared computation

Borrowing the approach from `~/projects/big2-ai`, which is the right model for this:

- `HandBits` (`src/core/util.h:59`) — **threshold bit-planes** (`at1/at2/at3/at4`),
  one bit per rank, so a predicate over all ranks is a single mask operation.
- `hand_bits_tables()` (`src/core/util.cpp:60`) — **superset-indexed lookup tables**
  built once, enumerating supersets with the `sup = (sup-1) & complement` trick.
- `compute_legal_moves` **emits** moves via `__builtin_ctz` bit iteration rather
  than testing every candidate.
- `compute_legal_moves_into` — fill-caller-buffer variant, no allocation.
- Auto-generated `constexpr` tables in `.inc` files (`move_to_cards.inc`).

Corintho's `Game::getLegalMoves` does the opposite of all of this: it sets all 96
bits, then **filters** by constructing a `Move` from each ID and testing it, while
recomputing `top()`/`bottom()` from scratch on every query.

### 13.1 Bit-plane state

Replace the interleaved `bitset<64>` (`row*16 + col*4 + piece`) with four
16-bit planes, one bit per space:

```
uint16_t b_;  // base present
uint16_t c_;  // column present
uint16_t a_;  // capital present
uint16_t f_;  // frozen
```

Derived quantities, each **one expression covering all 16 spaces at once**,
replacing the ~150 per-space `top()`/`bottom()` calls per move generation:

```
empty     = ~(b_ | c_ | a_) & 0xFFFF
top_base  = b_ & ~c_ & ~a_      // top() == 0
top_col   = c_ & ~a_            // top() == 1
top_cap   = a_                  // top() == 2
bot_col   = c_ & ~b_            // bottom() == 1
bot_cap   = a_ & ~b_ & ~c_      // bottom() == 2
```

### 13.2 All 48 place moves → 3 mask expressions

From `Game::canPlace` (`game.cpp:181`), noting that an empty space can never be
frozen (only `doMove`'s destination is ever frozen, and it always holds a piece):

```
placeable_base = empty
placeable_col  = ~f_ & ~c_ & ~a_              // empty is a subset of this
placeable_cap  = ~f_ & ~a_ & (~b_ | c_)       // empty is a subset of this
```

Mask each with `pieces_[to_play*3 + P] > 0 ? 0xFFFF : 0`, then iterate set bits
with `__builtin_ctz`. **48 `Move` constructions and 48 `canPlace` calls become
three mask computations.**

### 13.3 All 48 move-moves → 8 shift-AND operations

From `Game::canMove` (`game.cpp:206`): a move requires both spaces non-empty and
unfrozen, and `bottom(from) - top(to) == 1`. Since both are non-empty,
`bottom(from)` and `top(to)` are each in {0,1,2}, so only **two cases** satisfy
the difference:

| Case | `bottom(from)` | `top(to)` | Source mask | Dest mask |
|---|---|---|---|---|
| 1 | 1 (column) | 0 (base) | `bot_col & ~f_` | `top_base & ~f_` |
| 2 | 2 (capital) | 1 (column) | `bot_cap & ~f_` | `top_col & ~f_` |

Then, for each of the four directions, this is the standard bitboard
sliding-piece pattern:

```
legal_src[d] = (src1 & shift_back(dst1, d)) | (src2 & shift_back(dst2, d))
```

with the usual file/rank edge masks to stop column wraparound. **4 directions ×
2 cases = 8 shift-AND pairs cover all 48 move-moves.** Map the resulting source
masks to move IDs through a 16-entry `constexpr` LUT per direction (the ID
encodings in `encodeMove`, `move.cpp:84`, are a fixed permutation per direction).

**Net: the basic legality of all 96 moves goes from ~96 `Move` constructions plus
~150 `top()`/`bottom()` calls down to roughly 20 bitwise operations.** Emitted,
not filtered.

### 13.4 Line detection

The 102 `line_breakers` entries (`util.h:85`) are 34 line shapes × 3 piece types
(4 rows × 3 shapes + 4 cols × 3 shapes + 2 long diagonals × 3 + 4 short
diagonals). A line of type P over space set L exists iff `(top_P & L) == L`.

So `applyRowColLines` / `applyLongDiagLines` / `applyShortDiagLines` — currently
~50 `top()` calls between them — become **three plane computations plus 34 mask
compares each**, preserving the existing early-return structure and the
capital-line special case at `game.cpp:252-278`.

If profiling shows this still matters, go further: a `uint64_t lut[65536]`
indexed by a top-plane mask returning a 34-bit "lines present" set (512 KB,
read-only, shared across threads) reduces detection to three table lookups. Start
with the mask compares; only build the LUT if measurements justify it.

### 13.5 Other direct consequences

- `Game::doMove` (`game.cpp:58`) — clearing frozen becomes `f_ = 0` instead of a
  16-iteration loop; the piece transfer becomes a handful of mask ops.
- `Game::writeGameState` (`game.cpp:44`) — from four 16-bit planes, emit via a
  16-entry × 4-float LUT and `memcpy`.
- `Node` shrinks: four `uint16_t` + 6 piece counts + `to_play` = 15 bytes vs the
  current 16-byte `bitset` plus 7. Combined with the arena (§7.2), this helps the
  64-byte target.
- Everything becomes `constexpr`-friendly, so the tables can be compile-time.

### 13.6 Shared computation *across* nodes — the larger prize

§13.1–13.5 share work across the 96 moves within one call. The bigger opportunity
is sharing across nodes and across games, and Corintho is unusually favourable:
a tiny state space, heavy transposition (move-moves and place moves both commute),
and 8-fold board symmetry.

**The starting position is evaluated 25,000 times per generation.** Every game's
first `doIteration` (`trainmc.cpp:141`) constructs `new Node()` for the initial
position and requests an NN evaluation of it. All 25,000 requests are byte-identical.
That is the most visible instance of a general problem: with `kNumOpeningMoves = 6`,
early-game positions are shared across enormous numbers of games.

A risk ladder, cheapest and safest first:

| Level | Change | Semantics |
|---|---|---|
| 0 | **Deduplicate positions within a predict batch.** Hash the ~250k rows, evaluate distinct ones, scatter results back. | **None.** Identical outputs |
| 1 | **Persistent eval cache for the generation.** The playing model is frozen during self-play, so a cache is exactly correct. | **None.** Identical outputs |
| 2 | **Canonicalize by symmetry** before hashing (min over the 8 transforms). | Changes evals — makes them symmetry-invariant. Expected **strength-positive** |
| 3 | **Full transposition table** sharing tree nodes between transposing lines. | Changes the search. Biggest win, biggest risk |

Level 2 deserves emphasis: the pipeline already applies 8× symmetry augmentation
at training time (`selfplayer.cpp:78`), which is an admission that the network is
not symmetry-equivariant. Canonicalizing at inference enforces exact symmetry
invariance **for free**, and simultaneously cuts distinct positions by up to 8×.
That is a rare case of a speed optimization that should also gain Elo.

Memory note: a full cache entry is ~400 B (hash + eval + 96 policy floats), so
1M entries is ~400 MB — too much alongside everything else on a 15 GB laptop.
Mitigate with float16 policy storage, a fixed-size open-addressing table with
replacement, or — best value — **caching only shallow depths**, where sharing is
overwhelmingly concentrated.

### 13.7 Measure this before building it

The decisive Phase 0 number: **instrument one generation and count distinct
canonical positions versus total evaluation requests**, broken down by depth.

That single measurement sizes the entire §13.6 opportunity, and it is cheap. If
distinct positions are 10% of requests, Level 0–1 alone removes most of the NN
cost. If they are 90%, skip to §13.1–13.5 and the transposition table.

Report alongside it: distinct *games* per generation (opening diversity, §12.6)
and the distribution of `num_legal_moves` (sizes the arena and the edge layout).

### 13.8 Profiling setup, borrowed from big2-ai

- A dedicated `benchmark` build target, separate from the instrumented test build
  (big2's `Makefile:182-193`, `-DBENCHMARK_MAIN`). Corintho's `CMakeLists.txt:14`
  currently forces `--coverage -pg` into every build (§7.3), so there is no clean
  target to profile at all today.
- A scripted profiling driver with perf / callgrind / plain modes, after the
  pattern of `scripts/profile_pimc_selfplay.sh`.
- `-march=native` in the default flags — big2 has it (`Makefile:4`), Corintho does not.
- Validate every rewrite against the old implementation over an exhaustive or
  randomized position corpus: old and new `getLegalMoves` must return identical
  bitsets. Move generation is the one place where a subtle bug silently corrupts
  every downstream result.
