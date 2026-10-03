# corintho-ai — codebase audit

Branch `audit/codebase-review`, cut from `main` at `c6539bc`. **No files were modified.**

Hosted version: https://claude.ai/code/artifact/c677f10d-20c7-4120-bfa3-bf94538c06ae

**Method:** full read of the 5,537 first-party C++/Python/Cython lines; `-Wall -Wextra -Wpedantic` compile of every source; a standalone repro for CPP-2; the existing gtest suite (45/45 pass on the prebuilt binary); three subagents on the web app, the Python layer, and reachability.

Severity is about consequence, not effort. **Critical** = silently produces wrong results or hangs. **High** = wastes a lot of something, or breaks a documented feature.

| Subsystem | Critical | High | Medium | Total |
|---|---:|---:|---:|---:|
| C++ engine core | 4 | 1 | 5 | 10 |
| Training pipeline (Cython/Python) | 3 | 4 | 4 | 11 |
| Web app | 2 | 3 | 5 | 10 |
| Build & CI | 0 | 1 | 3 | 4 |
| **Total** | **9** | **9** | **17** | **35** |

Plus a dead-code inventory: ~7 MB tracked and unreachable, ~2.4 GB untracked local training detritus.

---

## C++ engine core

### CPP-1 — Drawn positions are always deduced as losses — **CRITICAL**

`corintho_ai/cpp/src/trainmc.cpp:518` · *verified against JS port*

Inside `propagateTerminal()`'s loop over a node's children, the draw check tests the **parent**, not the child:

```cpp
while (cur_child != nullptr) {
  if (cur_child->child_id() != cur->move_id(edge_index) || !cur_child->known())
    return;
  if (cur->drawn()) {          // BUG: cur is the parent being deduced
    has_draw = true;
  }
  ...
}
```

At that point `cur->result_` is always `kResultNone` — that is the value the loop is computing. So `has_draw` can never become true, and the `if (has_draw)` branch below is dead code: every fully-resolved non-winning node is marked `kDeducedLoss`, never `kDeducedDraw`.

The browser port at `web/trainmc.js:288` correctly reads `curChild.drawn()`, which is how I confirmed intent rather than guessing it.

**Impact:** a drawn position is scored as a loss for the side to move, and its parent then flips to `kDeducedWin` on the next iteration. This propagates upward, changes which branch of `chooseMove` runs, and feeds wrong evaluation labels into training. Every generation trained since this was introduced has learned from it.

---

### CPP-2 — `chooseHighProbMove` returns the last legal move, not the best one — **CRITICAL**

`corintho_ai/cpp/src/trainmc.cpp:298-308` · *reproduced*

```cpp
int32_t max_prob = 0;   // probability() returns float in (0, 1]
int32_t choice = 0;
for (int32_t i = 0; i < root_->num_legal_moves(); ++i) {
  if (root_->probability(i) > max_prob) {
    max_prob = root_->probability(i);   // truncates to 0
    choice = root_->move_id(i);
  }
}
```

Every assignment truncates a sub-1.0 float to `0`, so the guard `probability(i) > 0` passes on every iteration and `choice` ends up as whichever legal move happens to be last. Confirmed with a standalone repro: given weights `{0.5, 0.1, 0.3, 0.1}` the loop returns index 3, not 0.

**Impact:** fallback path, but not a rare one — it fires in `chooseMoveNormal` and `chooseMoveOpening` whenever `max_visits == 0`, i.e. one-search turns and positions where every child is a losing move. Exactly the solved-endgame case.

**Fix:** `float max_prob = 0.0f;`. The JS port uses the raw integer edge weight and is correct.

---

### CPP-3 — `std::vector<bool>` written concurrently from an OpenMP loop — **CRITICAL**

`corintho_ai/cpp/src/trainer.cpp:192`, `tourney.cpp:70` (declarations: `trainer.h:65`, `tourney.h:42`)

```cpp
#pragma omp parallel for
for (size_t i = 0; i < games_.size(); ++i) {
  ...
  if (done) is_done_[i] = true;   // vector<bool> — bit-packed proxy
}
```

`std::vector<bool>` packs 64 elements into one word. Two threads finishing games `i` and `i+1` perform read-modify-write on the *same* word with no synchronisation — a textbook data race, and one that loses updates rather than crashing.

**Impact:** a completed game can be marked incomplete, so it keeps being iterated after `endGame()` has already reset its root and released its eval buffer. The loop also reads `is_done_[i]` concurrently. Non-deterministic; more likely as thread count rises.

**Fix:** `std::vector<uint8_t>` or `std::deque<bool>`.

---

### CPP-4 — Tournament matches read the wrong network evaluations — **CRITICAL**

`corintho_ai/cpp/src/tourney.cpp:56-62`

```cpp
for (size_t i = 1; i < matches_.size(); ++i) {
  if (!is_done_[i] && matches_[i]->to_play() == id) {   // tests i
    offset += matches_[i - 1]->num_requests();          // adds i-1
  }
  offsets[i] = offset;
}
```

The eligibility guard and the accumulation disagree by one index. `writeRequests` (`tourney.cpp:44-52`) packs the batch as *"sum over j < i of eligible `num_requests(j)`"*, so `offsets[i]` only matches by coincidence. The equivalent loop in `Trainer::doIteration` (`trainer.cpp:208-215`) tests `i-1` in both places and is correct — this is a transcription slip in the copy.

**Impact:** matches read another match's evaluation and policy rows out of the batch. Silent: no crash, no assert, just subtly wrong play. Everything downstream of a tournament — the Elo table, generation-to-generation rating deltas — is built on this.

---

### CPP-5 — Self-play eval buffer over-allocated 100× — **HIGH**

`corintho_ai/cpp/src/selfplayer.cpp:24`, `match.cpp:18-20`

```cpp
to_eval_{std::make_unique<float[]>(kGameStateSize * max_searches)}
```

The buffer is only ever written at `to_eval_ + searched_.size() * kGameStateSize`, and `searched_.size()` is bounded by `searches_per_eval_`, never by `max_searches_`. The correct size is `kGameStateSize * searches_per_eval`. `DockerMC` (`dockermc.cpp:16`) already gets this right, which is good evidence it's a slip rather than a deliberate margin.

**Impact:** at `max_searches=1600, searches_per_eval=16` that's 448 KB per game instead of 4.5 KB. Every `SelfPlayer` is constructed up front in `Trainer::initialize`, so the default `num_games=25000` reserves ~11 GB of eval buffers before a single search runs. At the shipped default `searches_per_eval=1` the over-allocation is 1600×. This is almost certainly the ceiling on how many games you can run at once.

---

### CPP-6 — `TrainMC` has defaulted copy/move but owns a raw pointer it deletes — MEDIUM

`corintho_ai/cpp/include/trainmc.h:21-25`, `src/trainmc.cpp:34-36`

Rule-of-three violation: `~TrainMC() { delete root_; }` alongside `TrainMC(const TrainMC&) = default`. Any real copy double-frees the tree. It is currently harmless only because C++17 guaranteed elision means `SelfPlayer`'s array-of-two initialisation never actually copies, and `root_` is `nullptr` at that moment anyway.

Separately, both assignment operators are declared `= default` but are implicitly *deleted* because of the `const` members — the declaration says something untrue.

**Impact:** latent. But it detonates the first time someone puts a `TrainMC` in a container or refactors `SelfPlayer`. Delete the copy operations explicitly, or hold `root_` in a `unique_ptr`.

---

### CPP-7 — Variable-length arrays sized by `num_games` — MEDIUM

`trainer.cpp:116, 170, 207` · `tourney.cpp:56` · `trainmc.cpp:250, 274, 277`

`-Wpedantic` flags four: `float scores[games_.size()]` and three `int32_t offsets[...]`. These are a GCC/Clang extension, not ISO C++ (MSVC rejects them), and they are stack allocations scaled by a user-supplied flag. At `num_games=25000` that's 100 KB of stack per array; the default 8 MB stack holds, but nothing enforces it. The small VLAs in `trainmc.cpp` (bounded by 48 legal moves) are fine in practice.

**Fix:** swap the three large ones for `std::vector`.

---

### CPP-8 — Move id and new root can disagree when no child matches — MEDIUM

`trainmc.cpp:310-335` (`chooseMoveWon`), `337-361` (`chooseMoveLostDrawn`)

Both initialise `choice = 0, best_prev = nullptr` and only overwrite them if a child passes the filter. If none does, the function *returns move id 0* but calls `moveDown(nullptr)`, which promotes `root_->first_child()` — a different move. The caller then tells the opponent one thing and continues searching another.

`chooseMoveNormal` handles this case explicitly (rebuilds the root from `choice`); these two don't. Reaching it requires a `won()`/`drawn()` root with no correspondingly-resolved child, which the invariants should prevent — but `chooseMoveLostDrawn` on a terminal `kResultDraw` root has no children at all, and `moveDown` would then dereference `nullptr`.

---

### CPP-9 — Inconsistent null guard on `prob_sample` — MEDIUM

`trainmc.cpp:383` vs `396`

`chooseMoveOpening` guards `if (prob_sample != nullptr)` at line 383, then writes `prob_sample[choice] = 1.0` unguarded thirteen lines later. Currently unreachable because `assert(!testing_)` implies a non-null sample, but the two lines disagree about what's true.

---

### CPP-10 — `Node::visits_` is `int16_t` with no saturation guard — MEDIUM

`corintho_ai/cpp/include/node.h:164`

The header documents the ceiling — *"Search should stop for any node that has been visited 32,767 times"* — but nothing enforces it. `increment_visits()` is a bare `++visits_`. Overflow gives a negative visit count, which then divides the evaluation in `chooseNext` and inverts the UCB ranking.

**Impact:** not reachable at the training default (1600) or the web app's 4000. Reachable in deep analysis runs. Either clamp in `increment_visits` or treat ≥32767 as `all_visited`.

---

## Training pipeline (Cython / Python)

### PY-1 — The replay buffer has never been used — **CRITICAL**

`corintho_ai/python/main.pyx:208-219`

```python
for cur_path in old_training_samples:
    old_game_states = np.load(f"{cur_path}/game_states.npz")
    ...
    np.concatenate((game_states, np.reshape(old_game_states["arr_0"], (-1, 70))))
    np.concatenate((eval_labels, old_eval_labels["arr_0"]))
    np.concatenate((prob_labels, np.reshape(old_prob_labels["arr_0"], (-1, 96))))
```

`np.concatenate` is pure — it returns a new array and mutates nothing. All three return values are discarded, and line 219 returns the original three arrays untouched.

**Impact:** `--num_old_gens` (default **20**) is silently inert. Every generation has trained only on its own self-play samples. This is a real change to training dynamics, not a micro-optimisation — sample reuse is normally what keeps AlphaZero-style training stable.

**Fix:** accumulate into lists and do one `np.concatenate` per array after the loop (also avoids the O(n²) repeated copy). Note the `np.load` handles are closed inside the loop, so a naive fix must materialise the arrays first.

---

### PY-2 — Prediction buffers sized from the wrong game count — **CRITICAL**

`main.pyx:132-134`, called at `343` with the tester built at `329`

`play_games` always reads `params["num_games"]` to size `evals`, `probs` and `game_states` — but it is also called with the testing `Trainer`, which was constructed with `params["num_test_games"]`. The C++ side writes up to `games_in_trainer × searches_per_eval` rows.

**Impact:** heap buffer overflow past the end of three NumPy arrays whenever `num_test_games > num_games`. The two flags are clamped independently in `wrapper.py` (`num_games >= 1`, `num_test_games` default **400**), so `--num_games=100` is enough to trigger it. Silent corruption, no exception.

**Fix:** give `play_games` an explicit game-count argument.

---

### PY-3 — Unnamed runs are dead on arrival — **CRITICAL**

`corintho_ai/python/wrapper.py:247-250, 341-350`

```python
if len(name) == 0:
    now = datetime.now()
    name = f"_run_{now.year}{now.month}{now.day}"
    f"{now.hour}{now.minute}{now.second}"   # no-op expression statement
```

Two bugs stacked. The second f-string is a dangling expression, not a continuation, so generated names are date-only and collide for two runs on the same day. And `params["name"]` is never assigned the generated name — `setup_new_run` creates `{cwd}/_run_20260808/` using the local variable, then `start_generation` reads `params["name"] == ""` and builds `{cwd}//generations/gen_1`, so `os.mkdir(train_log_folder)` raises `FileNotFoundError`.

**Impact:** latent only because the shipped TOMLs always set `name`. Anyone running without `--name` gets an immediate crash after the model has been built and saved.

---

### PY-4 — A perfect test score poisons the rating chain permanently — **HIGH**

`main.pyx:274-283`

`new_rating = best_gen_rating - 400 * np.log10(1 / score - 1)`. At `score == 1.0` that is `log10(0)` → `+inf` with only a `RuntimeWarning`. The `inf` is written to `rating.txt`; since `score > test_threshold` the generation is promoted, so `setup_existing_run` reads `inf` as `best_gen_rating` next generation and every subsequent rating is `inf` or `nan`. The `score > 0` guard covers the mirror case but not this one.

**Impact:** reachable in practice — `test.toml` sets `num_test_games = 2`.

**Fix:** clamp the score to `[1/(2n), 1 - 1/(2n)]` before the log.

---

### PY-5 — `predict()` is handed the full 400,000-row array every call — **HIGH**

`main.pyx:70-83`

`get_predictions` passes the entire `game_states` array to `model.predict` and relies on `steps=1` to truncate it. At `num_games=25000, searches_per_eval=16` that's a 400,000×70 float32 array (~112 MB) converted into a `tf.data` pipeline on every evaluation, thousands of times per generation, to compute at most `num_requests` rows. `verbose=1` also prints a Keras progress bar per call.

**Impact:** very likely the dominant term in `predict_time`.

**Fix:** slice the input and use `model(x, training=False)` instead of `predict` for in-loop batches.

---

### PY-6 — Symmetry augmentation leaks across the train/val split — **HIGH**

`main.pyx:249-258`

Each position is written as 8 contiguous symmetry rows. Keras's `validation_split=0.3` takes the last 30% of the array *before* shuffling, so nearly every validation position has 1-7 of its own rotations sitting in the training set.

**Impact:** `val_loss` is optimistically biased — and it drives `ModelCheckpoint(save_best_only=True)`, `ReduceLROnPlateau`, and the `losses.txt` → `write_learning_rate` chain. Every learning-rate and model-acceptance decision keys off a leaked metric.

**Fix:** split by position group, or hold out whole games, and pass explicit `validation_data=`.

---

### PY-7 — `new Trainer` / `del trainer` is not exception-safe — **HIGH**

`main.pyx:299-357`

Both `Trainer` objects are allocated with bare `new` and only freed on the happy path. `play_games` raises explicitly at line 163; `train_neural_network` can raise on Keras OOM or checkpoint I/O; `get_samples` allocates multi-GB arrays. Each leaked `Trainer` owns `num_games` `SelfPlayer`s with their trees. Since `main.py` drives generations via `os.system` in a loop, this leaks per failed generation.

**Fix:** wrap each in `try/finally`.

---

### PY-8 — `cdef void` without `except *` can swallow the explicit raise — MEDIUM

`main.pyx:85, 123, 221`

`log_stats`, `play_games` and `train_neural_network` are all `cdef void` with no exception specification. Under Cython 0.29 semantics an exception inside them — including the deliberate `raise Exception("No requests during training")` at line 163 — is reported via `WriteUnraisable` ("Exception ignored in:") and **execution continues** into `get_samples` with garbage state.

Which behaviour you actually get depends on the installed Cython, and that's undetermined: the `language_level=3str` directive lives in `setup.py` as an inert Python comment rather than in the `.pyx` or in `cythonize(compiler_directives=...)`. The same ambiguity silently changes `main.pyx:116` from floor to true division.

---

### PY-9 — `cdef extern from "../cpp/src/trainer.cpp"` defeats rebuild tracking — MEDIUM

`main.pyx:17` · `setup.py:17-25` · `tourney.pyx:15`

No ODR problem — `trainer.cpp` is deliberately absent from `setup.py`'s sources, so it compiles exactly once inside the generated TU. The cost is dependency tracking: setuptools does no include scanning, so editing `trainer.cpp` or `trainer.h` alone leaves the object stale and `build_ext` reports "up-to-date". `build.sh` accidentally papers over it with `rm main.cpp`; a direct `setup.py build_ext` silently links yesterday's engine.

**Fix:** add `trainer.cpp` to `sources` and extern from `trainer.h`.

---

### PY-10 — `os.system` with no exit-code check — MEDIUM

`corintho_ai/python/main.py:56` · `corintho_ai/rating/tourney.py:42`

Both drivers build a shell command string from unvalidated TOML values and loop `for element in commands: os.system(element)`. If generation 3 of 20 segfaults, generations 4-20 run anyway against a half-written directory. It's also a shell-injection surface via `name`/`cwd`.

**Fix:** `subprocess.run([...], check=True)`.

---

### PY-11 — Rating utilities: sparse-matchup `KeyError`, unreset patience, biased pairing — MEDIUM

`corintho_ai/rating/rating.py:62-71, 129-135` · `round.py:188-191`

Three separate defects:

- `compute_gradient_r` indexes `self.matchups[k]` and `self.matchups[opp][k]` unconditionally while the dict is built only from rows present in `results.txt` — `KeyError` on any pair that never played, which is most of them. The C++ twin avoids this by materialising a dense 97×97 tensor.
- `updates_since_improvement` is incremented but never reset, so the learning rate decays on a fixed schedule regardless of progress. `rating_gd.cpp:125-133` does it correctly.
- `round.py` mutates `n1` before evaluating `m1`'s denominator, adding 0.96 games skewed toward the winner instead of a symmetric 1.0. Corrupts the priority heap that schedules pairings.

---

## Web app (`web/`)

Port fidelity is otherwise excellent: the 102×96-bit `LINE_BREAKERS` table is byte-identical to `util.h`, the reversed `std::bitset` string ordering is handled correctly, and the capital-line special case (`game.cpp:280-309`) is faithfully ported. The two divergences found (CPP-1, CPP-2) both have the JS as the correct side.

### WEB-1 — The search worker can spin forever — **CRITICAL**

`web/trainmc.js:120-133` → `web/worker.js:114-134`

`doIteration`'s `while` loop exits on **four** conditions but its return value only reports two of them:

```js
while (this.searched.length < this.searchesPerEval &&
       this.searchesDone   < this.maxSearches   &&
       !this.root.known() && !this.root.allVisited) { this.search(); }

return (this.searchesDone === this.maxSearches || this.root.known())
       && this.searched.length === 0;
```

When the loop exits because `root.allVisited` went true with `searchesDone < maxSearches`, it returns `false` with zero pending requests. The worker then skips `await runInference(...)` — so there is **no `await` anywhere on that path** — and calls `doIteration` again, which immediately fails the same guard. Forever.

`search()` itself sets `allVisited` on the root and *decrements* `searchesDone` when `chooseNext` returns `"none"`, so once the root is exhausted `searchesDone` can never reach `maxSearches`.

**Impact:** worker pins a core at 100%, the event loop never yields so `onmessage` can never be re-entered, and the UI sits on "CPU is thinking…" permanently with no timeout.

**Fix:** make the return condition total — `... || this.root.allVisited`.

---

### WEB-2 — CDN failure hangs the game, and the UI promises otherwise — **CRITICAL**

`web/corintho.js:656-690` · `web/worker.js:9`

`chooseCPUMove` subscribes only to `message` — no `error`, no `messageerror` handler. `worker.js` does a *top-level* `import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/…"`. If that import fails — ad-blocker, corporate proxy, offline, CSP, jsdelivr outage — the module never evaluates, `self.onmessage` is never installed, the posted message is dropped, and the promise never settles.

The shipped copy at `index.html:422` says *"If the model fails to load, the game falls back to random legal moves."* That's true for `InferenceSession.create` failures, which the worker's `try/catch` does cover — but not for load failures, which happen before the `try` exists.

**Fix:** add `error`/`messageerror` listeners routing to the existing random fallback, plus a wall-clock timeout. Moving to `await import()` inside the try block makes CDN failure catchable.

---

### WEB-3 — Starting a new game mid-search corrupts the board — **HIGH**

`web/corintho.js:288-324, 656-690, 846-871`

Nothing cancels an in-flight search and nothing tags requests. Click New Game while the CPU is thinking and a second message goes to the same worker with a second listener attached; both listeners receive every reply, and the first resolves on whichever arrives first. The old `runCpuResponse` then resumes after its `await`, reads the *module-level* `gameState` (now the new game), and applies a move computed for the old position.

`doCPUMove` performs zero validation: `placePiece` decrements `pieceCounts` below zero, `movePiece` moves pieces out of empty cells.

**Impact:** silent board corruption, plus two searches interleaving at their `await` points and roughly doubling each other's latency.

**Fix:** monotonic request id; re-check `gs === gameState` after the await; disable New Game while thinking, or terminate and recreate the worker.

---

### WEB-4 — NaN cascade when the policy head zeroes every legal move — **HIGH**

`web/trainmc.js:150, 178` · `node.js:102-105`

If the float32 policy output underflows to 0 across all legal moves, `scalar = (1/0) * (1-ε) = Infinity`, then `0 * Infinity = NaN`. In `setProbs`, `NaN > maxProb` is false so `maxProb` stays 0, `denom = 511/0 = Infinity`, and `Math.max(1, Math.round(NaN))` is `NaN` — every edge probability and the denominator go NaN. `node.js` guards only `denominator <= 0`, and `NaN <= 0` is false.

**Impact:** every `u` in `chooseNext` becomes NaN, `maxEval` stays `-Infinity`, `"none"` is returned at the root — the deterministic trigger for WEB-1. The C++ at least has `assert(denominator > 0.0)`.

**Fix:** guard with `if (!(sum > 0))` / `if (!(this.denominator > 0))`, which catches NaN.

---

### WEB-5 — Chosen move id is applied without a legality check — **HIGH**

`web/worker.js:136-138` · `trainmc.js:378-415`

`chooseMoveWon`, `chooseMoveLostDrawn` and `chooseHighProbMove` all default `choice = 0`. `buildResponseAfterCpuMove` takes the result straight to `g.doMove(moveId)` with no validation (C++ at least has `assert(isLegalMove(move_id))` at `game.cpp:64`, even if `NDEBUG` compiles it out). Move id 0 is "move stack a4 → b4", almost always illegal.

**Impact:** defensive-only in the normal flow, but WEB-4 makes it live.

**Fix:** check the id against the root's edges before applying; on failure post an error so the random fallback fires.

---

### WEB-6 — No tree reuse; every CPU move rebuilds 4,000 nodes from scratch — MEDIUM

`web/worker.js:96-107`

`TrainMC::moveDown` and `receiveOpponentMove` have no JS counterpart, and the worker constructs a fresh `TrainMC` per incoming message. The C++ keeps the subtree under the chosen move; the web app discards it. Roughly doubles time-to-move and peak allocation.

Related, same file: `new Float32Array(16 * K_GAME_STATE_SIZE)` at line 93 hardcodes the batch size separately from `searchesPerEval: 16` at line 101. Change one without the other and `writeGameState` writes past the typed array — which JS **silently ignores**, so positions would be evaluated as all-zero game states with no error.

---

### WEB-7 — ~1M short-lived objects per CPU move — MEDIUM

`web/node.js:26, 59-76` · `web/engine.js:137-140, 312-349`

Each `MCTSNode` clones two `Int8Array`s (~200 bytes of object headers for 70 bytes of data, versus C++'s inline 16-byte `Game`), allocates `new Array(96).fill(true)`, and calls `isLegalMove` 96 times — each of which runs `decodeMove`, allocating a move object plus nested `{row, col}` objects. Retained edges are up to 48 `{moveId, prob}` objects, ~2 KB per node against C++'s 2-byte packed bitfields.

**Impact:** at 4,000 searches that's roughly 1M transient objects and ~9 MB retained per move.

**Fix:** precompute the 96 decoded moves into a frozen module-level table (they're constant); reuse a scratch `Uint8Array(96)` for the legal mask.

---

### WEB-8 — No CSP, no SRI, and the Tailwind JIT compiler in production — MEDIUM

`web/index.html:19, 469, 477` · `web/worker.js:9, 14` · `web/vercel.json`

No `Content-Security-Policy`, `X-Content-Type-Options`, `Referrer-Policy` or `Permissions-Policy` anywhere. `index.html:19` loads `cdn.tailwindcss.com` — the in-browser JIT compiler that Tailwind explicitly documents as not for production; it's ~100 KB of render-blocking JS that needs `unsafe-eval`, so it actively blocks adopting a CSP. Zero `integrity` attributes on it or the two Google Fonts stylesheets. The worker pulls both the ORT bundle and its `.wasm` binaries from jsdelivr at runtime. Inline `onclick=` handlers at 469 and 477 would break under any `script-src` policy.

**Impact:** a jsdelivr or Tailwind CDN compromise executes arbitrary JS on the page and inside the worker. Self-hosting `ort.min.mjs` + the wasm files also fixes WEB-2's availability failure.

---

### WEB-9 — `model.onnx` is immutably cached for a year at a stable path — MEDIUM

`web/vercel.json:4-8`

`public, max-age=31536000, immutable` on `/model.onnx`, which is not content-hashed. Ship a retrained model and returning visitors play the old one for up to a year with no cache-bust available. Meanwhile the JS that would reference a new filename is cached for only an hour, so there's no coherent versioning story.

**Fix:** rename to `model.<hash>.onnx` and reference it via the constant already at `worker.js:18`.

---

### WEB-10 — Accessibility: focus destroyed on every redraw, modals don't trap it — MEDIUM

`web/corintho.js:421-433, 271-295` · `web/index.html:190-229, 266-270`

- `drawBoard` does `boardElement.innerHTML = ""` and rebuilds all 16 tile buttons on every `drawGame`, so a keyboard user who commits a move has focus reset to `<body>` immediately, and again when the CPU replies.
- Tiles carry only `aria-label="Cell a4"` — no stack contents, no frozen state, no `aria-pressed`.
- Overlays never move focus into the dialog, never trap Tab, never restore focus on close, and the background isn't `inert`.
- `#turn-counter` changes from "It's your turn!" to "You won!" with no `aria-live`, so the result is never announced.
- Several `role="tab"` buttons sit outside any `tablist`, and no panel has `role="tabpanel"`.

---

## Build & CI

### CI-1 — Nothing in CI builds the Cython extensions or runs a Python test — **HIGH**

`.github/workflows/` — doxygen, C++ gtest, isort, black+flake8

No job runs `setup.py build_ext`, and the repo contains zero Python tests (`tests/` holds only `tests/cpp/`). The entire Python↔C++ boundary is unverified — PY-1, PY-2, PY-8 and any signature drift between `trainer.h` and the `cdef extern` block all land on `main` green. flake8 is also scoped to `corintho_ai/python` only, so `corintho_ai/rating/*.py` is entirely unlinted.

---

### CI-2 — The test build and the shipping build are different configurations — MEDIUM

`CMakeLists.txt:14` vs `setup.py:26-31`

```cmake
set(CMAKE_CXX_FLAGS "--coverage -fopenmp -g -pg -O3")
```

Four problems in one line:

1. `set` *replaces* rather than appends, discarding toolchain flags.
2. Coverage and `-pg` profiling are both unconditional, so there's no way to produce a clean or benchmarkable build.
3. `-O3` without `-fno-inline` makes the lcov line attribution reported in CI unreliable.
4. No `-DNDEBUG` — while `setup.py` and `setup_tourney.py` both *do* pass it. So gtest exercises a build with asserts live, and the Cython artifact that actually trains has them compiled out.

No `-Wall -Wextra` either; adding it surfaces sign-compare warnings on all six `assert(searched_.size() <= searches_per_eval_)` sites.

---

### CI-3 — The vendored gtest isn't the gtest that gets linked — MEDIUM

`CMakeLists.txt:16-17` · `run-cpp-tests.yaml:24-28`

`googletest/` is gitignored and untracked, and the workflow clones only GSL. So in CI, `include_directories(googletest/googletest/include)` and `link_directories(googletest/lib)` both point at nothing, and the build silently succeeds against `libgtest-dev`'s system headers and libs. Locally, where `googletest/` exists, you compile against vendored headers and link system libs — a genuine ODR hazard.

**Fix:** `FetchContent` or `find_package(GTest REQUIRED)` + `GTest::gtest_main`.

---

### CI-4 — Path filters that match nothing, and a docs job with three failure modes — MEDIUM

`run-cpp-tests.yaml:10` · `run-lint-cpp.yaml:8` · `generate-doxygen.yaml:13-25`

- `'CMakelists.txt'` — GitHub path filters are case-sensitive and the file is `CMakeLists.txt`, so build-definition pushes never trigger the C++ tests.
- `'tests/cpp/src/*.cpp'` matches nothing; the tests are flat at `tests/cpp/*.cpp`.
- The doxygen job runs `sudo apt-get install doxygen graphviz` with **no `-y`** (every other workflow has it) — with no TTY, apt-get reads EOF at the prompt and aborts. It also uses the deprecated `actions/checkout@v2` and never declares `permissions: contents: write`, which `peaceiris/actions-gh-pages` needs.

---

## Dead weight — ~7 MB tracked, unreachable

Everything here was proven unreachable by grepping for its basename and its defined symbols across every build file, workflow, script and source. **Nothing was deleted.** Most of it is fallout from commit `c6539bc`, which removed the Cloud Run deployment.

| Path | Size | Why it's dead |
|---|---:|---|
| `external/spdlog/` | 1.6 MB, 169 files | Zero `#include <spdlog/…>` anywhere. Not in CMakeLists or any setup.py. Committed as plain files, not a submodule — there is no `.gitmodules` and `git submodule status` is empty. A local clone got staged by accident. |
| `corintho_ai/model/`<br>`first_run/` | 2.8 MB<br>2.8 MB | Two tracked Keras SavedModels with zero tracked references. The pipeline writes and reads models at `{cwd}/{name}/generations/gen_N/model`. `first_run/`'s only consumer is `tflite.py`, which is itself gitignored. |
| `cpp/src/dockermc.cpp`<br>`cpp/include/dockermc.h` | 3.1 KB | Absent from the CMakeLists source list and from both setup.py extension lists. Repo-wide `grep DockerMC` hits only the two files and a comment in `web/trainmc.js:40`. The Cloud Run wrapper, orphaned. Still gets Doxygen'd. |
| `python/play.pyx`<br>`python/play_setup.py`<br>`python/play_corintho.py`<br>`scripts/bash/play.sh` | 6.6 KB | The whole human-play path is uncompilable. `play.pyx:15` externs from `"../cpp/playmc.cpp"`, which does not exist — `grep -r PlayMC corintho_ai/cpp/` returns nothing. `play_setup.py` also lists `../cpp/node.cpp` etc., which moved to `../cpp/src/` in the 2023 reorg. Superseded by the web build. |
| `flask_script.sh` | 174 B | `cd corintho_ai/docker` — deleted. Added by the very same commit that deleted the directory it targets. |
| `scripts/bash/make.sh`<br>`lint.sh` · `tourney.sh` · `test.sh` | 670 B | All four reference paths that no longer exist (a `cpp/Makefile`, `corintho_ai/docker/`, `python/setup_tourney.py`, `bash/build.sh`). None use `set -euo pipefail`, so `train.sh` proceeds to training even when the extension build failed — silently training with a stale `.so`. |
| `corintho_ai/rating/rating_gd` | 27 KB | A compiled ELF binary tracked in git. `a.out` in the same directory is caught by `.gitignore`'s `*.out`; this one isn't. |
| `assets/images/corintho.png`<br>`web/base.png` · `capital.png` · `column.png` | 254 KB | `grep -rn "\.png" web/` returns nothing — the board renders pieces with Material Symbols glyphs. The README uses `logo.png`, not `corintho.png`. |
| `rating/rating.py` · `rating_gd.cpp`<br>`python/compare_outputs.py`<br>`python/get_ratings.py` | 11.8 KB | Stale one-offs. No importer, no caller. Their inputs (`results.txt`, `gd_ratings.txt`) were deleted in `c6539bc`; the surviving paths point at directories that moved. `rating.py:150-171` also runs 10,000 iterations and never writes the result anywhere. |
| `rating/tflite_models/*.tflite` | 48 MB, 94 files | **Uncertain — decide, don't assume.** No tracked file names this directory. Written by the gitignored `tflite.py`; the one reader, `compare_outputs.py`, has the wrong relative path. Could be a deliberate rating corpus. |

**Untracked local detritus:** ~2.4 GB of root zips (`train_94.zip` alone is 2.0 GB), plus `generations/` at 322 MB and ~90 MB of build output. All correctly gitignored — `git status` is clean — so this is disk hygiene, not repo hygiene.

**Separately:** `corintho-ai-json-key.json` is a GCP service-account key sitting in the repo root. It is gitignored and `git log --all --` confirms it was **never committed**, so this is not a leak. But it's a live credential for a deployment you retired — worth revoking rather than leaving on disk.

---

## Where I'd start

1. **CPP-1 and CPP-2** are one-line fixes with the JS port as a ready-made reference for what correct looks like. They change what the engine believes about positions, so everything else is downstream of them.
2. **CPP-5** is a one-word fix (`max_searches` → `searches_per_eval`) that should reclaim most of the self-play memory footprint. Worth measuring before and after — it may raise the ceiling on `num_games` by two orders of magnitude.
3. **WEB-1 + WEB-2** are the two ways a visitor's game hangs with no recovery. Both are small, and WEB-2's fix makes the fallback the UI already advertises actually true.
4. **CPP-3 and CPP-4** need a moment's thought about the concurrency model, not just an edit. CPP-4 in particular means past tournament results are suspect — worth deciding whether the rating history needs regenerating.
5. **PY-1** is a behaviour change, not a bug fix: turning the replay buffer on will alter training dynamics. Probably wants a controlled comparison rather than a straight merge.
6. **A differential fuzz harness** between `web/engine.js` and `cpp/src/game.cpp` — random playouts through both, comparing the 96-bit legal mask each ply — would have caught CPP-1 and CPP-2 from either direction, and is cheap next to maintaining two hand-written copies of the rules.
