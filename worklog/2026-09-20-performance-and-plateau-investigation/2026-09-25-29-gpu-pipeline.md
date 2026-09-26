# 29 — The GPU pipeline: built, gated, and running full generations in ~7 minutes

The developer asked for the new pipeline (entry 21's design) to be built,
tested and profiled for big remaining gains, unattended while they were at
work, without breaking anything. Branch `feat/gpu-pipeline`, cut from
`perf/engine-optimizations`. The Cython pipeline is untouched. How to use it
is in `pipeline/README.md`. Raw data is in `data/gpu-pipeline/`.

## What was built

| commit | piece |
|---|---|
| `86e08d8` | `pipeline/cpp/corintho_play`: the C++ driver (`train`, `test`, `bench`), with CPU `Mlp` and ONNX Runtime CUDA backends; `model.py`; `keras_to_npz.py`; `setup.sh` |
| `0bd4b71` | `fit.py` (PyTorch, equivalent to the Keras fit) and `run.py` (the resumable generation loop); `models/gen_93.npz` |
| `16ed8b8` | **engine fix:** the node pool shares free blocks between threads (below) |
| `4018068` | overlapping search and GPU evaluation (`--groups`), and pinned staging buffers |
| `1562761` | the fitter replays a recorded CUDA graph; loop defaults 2 x 2,000 games |
| `3e58f1c` | fitter: graph-capture flags restored after loading optimizer state |

Environment: `setup.sh` makes a uv venv with torch 2.14 (cu130) and
onnxruntime-gpu 1.30. The wheels bring CUDA 13 and cuDNN 9, so no system
toolkit is needed. It also downloads the ONNX Runtime C++ release, CUDA 13
build, sha256-checked. The driver carries an old-style DT_RPATH to the
venv's CUDA libraries, because the CUDA provider `dlopen`s them.

## Gates (nothing trusted until it matched something known)

1. **Driver vs `selfplay_nn`:** with the CPU backend in one chunk,
   `corintho_play train --digest` reproduces the recorded sample digests on
   seeds 1–4 exactly. That held again after every later driver change (one
   group).
2. **Keras to PyTorch:** gen_93's SavedModel was converted with the
   BatchNorm statistics kept. It matches Keras to 4.9e-5 (value) and 7.3e-6
   (policy). The folded `.mlp` maths and the `.onnx` file match PyTorch to
   ~3e-5.
3. **GPU vs CPU network on real search batches** (`--check`, 36M rows):
   - max difference 2.1e-4 (value) and 7.3e-5 (policy);
   - the top policy move differs on 196 rows, all near-ties.

   This is fp32 on both, with TF32 off: entry 20 found TF32/fp16 off by up
   to 0.18.
4. **Loss definitions:** PyTorch losses on gen_93 self-play (value 0.561,
   policy 2.032) match Keras's recorded gen_92 (0.564, 2.036).
5. **Graph vs eager fitting:** bit-identical weights after 2 epochs on
   5.7M rows.
6. **Engine gates for the pool change:**
   - unit tests 45/45;
   - golden, verify and rulecheck;
   - sample digests (seeds 1–4) and the assert build all identical.
7. **Resume:** a run killed during self-play restarted from that step and
   completed.

## Problems found and fixed on the way

- **BatchNorm on the GPU:** exported with its 12 BatchNormalization nodes,
  the ONNX model ran at ~1,640 ns/row on ORT's CUDA provider, against ~200
  for the plain MLP of entry 20. The export now folds BatchNorm into the
  following layer (`FoldedNet`), the same maths as the `.mlp`.
- **Heap corruption at exit** (`corrupted double-linked list`) in every
  optimized build with the GPU backend. The static `Ort::Env` was destroyed
  during exit, after the CUDA provider's own teardown. The Env is now never
  destroyed. Checked with 9 runs over three builds.
- **The ORT environment must exist before `CreateCUDAProviderOptions`.**
- **Pinned buffers:** bound as `CudaPinned`, ORT tried to copy a buffer
  onto itself. They are bound as CPU tensors instead: CUDA recognises
  page-locked memory by address, so the copies are still DMA.
- **A race:** two groups shared one backend's staging buffers. It showed as
  blank runs and one wrong digest. Each group now has its own backend.
- **The old pipeline never used older generations' samples.** `main.pyx`
  `get_samples()` called `np.concatenate(...)` and discarded the result, so
  `num_old_gens: 2` never took effect. `run.py --old-gens` defaults to 0,
  what was actually done. Using a window is the developer's decision.
- **Optimizer flags:** `load_state_dict` restored `capturable=False` from
  state saved before the graph fitter, which stopped gen 2 of the first
  multi-generation run. Fixed in `3e58f1c`. `run.py` now also logs a failed
  step clearly in `progress.log`.

## The engine's node pool leaked memory across threads (engine fix)

Driving chunk after chunk showed resident memory growing **~290 MB per
1,000 games** at 20 threads and never shrinking:

- 1.66 → 4.83 GB over 12 chunks of 1,000;
- 10.1 GB peak for one 25k-game generation.

With 1 thread it stayed flat (687 → 697 MB).

**Cause:** `Arena`'s header said *"Trees do not migrate between threads"*.
That stopped being true with entry 19's dynamic schedule: a game runs on
different threads from one iteration to the next. So its blocks are freed
into other threads' lists, which pile up while the allocating threads carve
new memory.

**Fix (`16ed8b8`):**
- free blocks move in batches of 256;
- a thread keeps up to 16 full batches per size class and passes older
  surplus batches to a shared pool;
- a thread with no free blocks takes a pool batch before carving new
  memory;
- chunks are owned by the pool.

**Result:** memory is flat (1.48 → 1.53 GB over 12 chunks), and one chunk
peaks lower (~800 against ~950 MB).

**Cost:** engine +2.2% at 20 threads (95% CI +1.5 to +3.0, n=6), and
single-threaded +1.6% (not significant). The path to that:

| version | engine time vs old pool |
|---|---|
| one spare batch per thread | +11% |
| 16 local batches | ~+7% |
| lock-free empty check | ~+4% (single chunk; +2.2% measured paired) |

- **Cross-thread handoff costs locality:** blocks freed on one core were
  handed to another core cold.
- **The empty-pool check took the mutex on every allocation while
  carving,** with 20 threads contending on it. That cost ~5%.
- **Freshly carved memory is also contiguous,** which gives new subtrees
  good locality. That is probably what the rest of the gap is.

A possible refinement: when no trees are alive (between chunks, one group),
reset every pool to fresh contiguous memory. Not done.

## Performance: one 25k-game generation

The first full generation (1 group x 1,000 games, eager fitting):

| phase | time |
|---|---|
| self-play | 407 s (engine 181 s + GPU 223 s, taking turns) |
| fit | 127 s |
| test, 1,600 games | 26 s |
| **total** | **~9.4 min**, against ~57 min on the cloud |

### Overlapping search and the GPU (`--groups`)

- **The GPU was ~41% busy.** A GPU call at 16k rows is ~2.0 ms of
  compute plus ~0.8 ms to copy the policy out. This thin, 100-wide
  network is limited by GPU memory bandwidth, not arithmetic (125 ns/row).
- **With pageable buffers, two groups gave nothing,** because CUDA's
  CPU-side staging copies competed with the engine threads. Pinned staging
  (and one backend per group) made overlap work.
- **8k games:**

  | groups x in-flight | wall |
  |---|---|
  | 1 x 1000 | 138 s |
  | 2 x 1000 | 124 s |
  | **2 x 2000** | **107 s** |
  | 3 x 2000 | ~the same as 2 x 2000 |

- The engine slows ~15% while overlapped: the GPU transfers and the engine
  share memory bandwidth.
- **At 25k games with 2 x 2000: self-play 336 s against 407 s (−17.5%)**,
  at 3.7 GB peak after the pool fix (10.1 GB before).

### Fitting (`fit.py`)

- **The problem:** a step took ~6 ms at any batch size from 2048 to 8192.
  It was ~150 small GPU operations, bound by Python dispatch and kernel
  launches.
- **First try:** removing the per-step `.item()` sync and using fused Adam
  saved only ~10%.
- **The fix:** recording the whole step as a CUDA graph and replaying it.
  Two epochs plus load and export went from 30.1 s to 12.3 s, with
  bit-identical weights.

### Full generations with everything

`runs/full-2`, four 25k-game generations. Gen 1 still used the eager
fitter, gens 2–4 the graph fitter. Rows are copied to
`data/gpu-pipeline/full-2-*`.

| gen | self-play | fit | test | total | result vs best | first-player score |
|---|---|---|---|---|---|---|
| 1 | 345.7 s | 114.5 s | 27.8 s | 8.1 min | 825W 48D 727L, lower bound 0.507: **promoted** | 0.502 |
| 2 | 346.9 s | 38.5 s | 27.3 s | **6.9 min** | 769W 65D 766L, 0.476: kept gen 1 | 0.502 |
| 3 | 345.7 s | 41.2 s | 28.2 s | 6.9 min | 757W 57D 786L, 0.466 | 0.502 |
| 4 | 346.6 s | 40.7 s | 28.3 s | 6.9 min | 803W 41D 756L, 0.490 | 0.503 |

- **~6.9 minutes per generation**, against ~57 on the cloud: ~8x.
- An 8-hour night is ~70 generations.
- Validation loss ~1.080, flat. Learning rate 5e-6 (gen_93's; no anneal
  yet).

Where the time goes now:
- **Self-play is 84%.** Engine ~205 s. The GPU work, ~157 s at the
  standalone rate, only partly overlaps it; ~130 s of waiting remains.
- **Remaining self-play levers:**
  - rolling starts (no chunk tails);
  - smoother overlap;
  - ORT CUDA graphs with padded fixed-size batches.

  Each is probably 5–15%.
- **Fit (~40 s) and test (~28 s) are small now.**

## Findings to flag

- **Colour balance:** the first-player score in self-play is ~0.50 (0.499
  and 0.502 in the first two full generations). Entries 10–11 found a ~74%
  second-player win rate with the old engine and rules. It looks gone with
  the fixed rules (entry 04) and this engine. That bears directly on the
  plateau question, but it is four generations (0.499–0.503), not a study.
- **The gate promotes again.** Generation 1 of `full-2` beat gen_93 (825W
  48D 727L, Wilson lower bound 0.507). The first try, `full-1`, did not
  (0.487).

## Not done / next

- **Starting new games as others finish,** instead of fixed chunks. Chunk
  tails give small batches (248 against ~205 ns/row) and leave threads idle.
  Now worth maybe 5–10% of self-play.
- **A pool reset between chunks,** to recover the 2.2%.
- **An old-generation sample window** (the developer's decision), and
  whether to keep samples.
- **A match harness** for the strength questions (entry 21's open threads).
  Colour balance can now be measured directly from `generations.tsv`.

## Addendum: the stagger, threads, and the developer's questions

**Gen 1 was not trained from scratch.** Gen 0 of a run is gen_93, the last
cloud model, converted. Gen 1 fine-tunes it on 25k fresh games from the fixed
engine. The first attempt (`full-1`) failed the gate (lower bound 0.487); the
second (`full-2`) passed narrowly (0.507), with decisive win rates of 51–53%
in both. If the two models were equally strong, the gate would pass by chance
~2.5% of the time per test. A small real gain is plausible (gen_93 trained on
data from the old rules and the entry-17 draw bug), but it is weak evidence.

**The staggered start spans ~16 turns, not one.** `Trainer` starts
`games / max_searches` games per iteration. But an iteration is 16
searches, so a turn is 100 iterations, and a 2,000-game chunk ramps up over
2,000 iterations. GPU calls averaged 46% full.

- **`Trainer::set_stagger_iterations`** is new (default: the original rule).
  With it, `corintho_play --stagger 100` spreads a chunk's starts over one
  turn: calls are 74% full.
- **The wall time does not change** (152.4 against 153.1 s, 12k games). With
  more games alive at once, the memory-bound engine slows ~13%, cancelling
  the GPU gain, and memory rises 25%. The long ramp acts as throttling that
  suits this engine.
- The driver keeps the original rule.
- Start times do not change the games: the digests are identical.

**Engine threads and OpenMP spinning:** `KMP_BLOCKTIME=0` and 19 engine
threads show nothing beyond thermal drift. The same configuration drifted
+15% over the 10-minute series. At this point self-play tuning is below the
laptop's thermal noise, so it stopped here. Raw data:
`data/gpu-pipeline/stagger-and-threads.txt`.

**The developer asked whether fit and test could overlap self-play.**

- **Pipelining generations** means starting gen g+1's self-play with the
  current best while gen g fits and tests, and switching models at a chunk
  boundary if gen g is promoted.
  - At most ~16% (fit + test is ~68 s of ~415 s), realistically ~5–10%:
    fit competes for the GPU and test for the CPU.
  - After a promotion, the first 1–2 chunks would be played by the previous
    best (AlphaZero-style staleness). That is a behaviour change, so it is
    the developer's decision.
- **Fitting chunk by chunk** is not equivalent to the current 10-epoch fit
  with the last 30% held out, and fit is only ~33 s. Not worth it except as
  a deliberate move to continuous training.

**Thermal mode.** Under sustained load the CPU ran at ~2.6 GHz and 91–94 C,
against a ~4.1 GHz all-core turbo, and throughput drifted +15% within 10
minutes. The developer switched Dell's `AWThermalManagement` BIOS attribute
(exposed by `dell-wmi-sysman`) from Balanced to Performance. The same 5-run
series then went 101.7 → 107.3 s (+5.5% drift): 88 C mean, ~2.7 GHz, fans
~3,800–4,300 rpm. That is ~10% faster once hot, approximate because the
comparison is not interleaved. Keep Performance mode for long runs.
