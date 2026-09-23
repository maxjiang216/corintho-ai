# 18 — The network in-process: a C++ forward pass, parity-checked on real search states

First step of the inference plan agreed with the developer on 2026-09-23. Raw outputs are
in `data/in-process-mlp/runs.txt`.

## The plan this belongs to

The developer's decisions, 2026-09-23:

- **Training moves to this laptop**, overnight runs. It has an RTX 3060 Laptop
  GPU (6 GB) and an i7-12700H (6 P-cores + 8 E-cores, 20 threads). There is no
  CUDA toolkit, TensorFlow or Cython here, only the driver (CUDA 13.1).
- **Fitting moves to PyTorch.** It gives room for architecture experiments
  later: supervised training on the best current model to compare
  architectures, then self-play continues with the winner. For now the
  architecture stays as it is.
- **The network runs inside the C++ process**, not through Cython. That opens
  up the pipeline: waves of games, alternating game groups so the GPU and CPU
  overlap, pinned transfer buffers.
- Efficiency is judged by where the GPU saturates, not at 25k games per batch.

The three steps:

1. **(this entry)** A hand-written CPU forward pass, with parity against the
   tflite model and a CPU baseline.
2. Measure the engine's real cost per generation locally.
3. PyTorch in a venv, GPU throughput by batch size, then an ONNX Runtime GPU
   backend and alternating game groups.

### Where a generation's time went (gen_93, cloud, 2023)

From `generations/gen_93/{training,testing}_logs/play_time.txt` and
`fit_time.txt`, on a 32-vCPU CPU-only box using Keras `.predict`:

| Phase | Time | Share |
|---|---|---|
| Self-play: network inference | 24m58s | 44% |
| Self-play: C++ engine | 14m47s | 26% |
| Fitting | 9m23s | 17% |
| Testing: network inference | 6m20s | 11% |
| Testing: C++ engine | 31s | 1% |
| **Total** | **~57 min** | |

The "~48 min" in entry 01 left testing out. Testing inference cost ~4× more
per row than self-play (~9 µs against ~2 µs), because its batches are small
(~13k rows against ~250k) and Keras's fixed cost per call dominates.

## The model at inference time

The tflite converter had already folded every BatchNormalization into the
Dense layer after it. So `model_93.tflite` holds a plain MLP:

- 12 × (FC + ReLU), 70 → 100 → … → 100;
- value head: FC to 1, then tanh;
- policy head: FC to 96, then softmax.

That is 133k multiply-adds, or ~266 kFLOP per row. Weights are stored
`[out][in]`.

`bench/export_mlp.py` reads the weights from the tflite interpreter and writes a
flat `.mlp` file (format documented in `mlp.h`). It checks the op sequence
before writing. It does **not** check the fused activation of each FC layer: it
assumes ReLU on the hidden layers from the tensor names. The parity test below
is what confirms that.

The Keras SavedModel (`generations/gen_93/model`) cannot be read without
TensorFlow, so tflite is the source of truth here. Selfplay_nn has always used
the tflite files as "the real network". Whether the tflite export matches Keras
exactly has not been checked; tflite.py produced them in 2023.

## `Mlp` (`corintho_ai/cpp/{include/mlp.h,src/mlp.cpp}`)

- Weights are transposed to `[in][out_pad]`, with outputs padded to a multiple
  of 8 using zeros, so there is no remainder loop.
- Rows go in blocks of 16, **one layer at a time**, so each layer's ~41 KB of
  weights stays in L1 across the block.
- The kernel is an AVX2/FMA microkernel: a 4 rows × 3 vectors register tile,
  i.e. 12 accumulators plus 3 weight registers. Each step does 7 loads for 12
  FMAs. 100 outputs = 13 vectors = 4 × 3 + 1.
- A portable fallback is used when AVX2/FMA are not compiled in.
- `evaluate` runs on the calling thread, which is what the engine will want
  when each thread evaluates its own games. `evaluateParallel` uses OpenMP.

### Two findings on the way

1. **The first, plain-C++ version reached 15 GFLOP/s on one core.** There were
   two reasons:
   - **`-std=c++17` makes GCC default to `-ffp-contract=off`**, so it emits no
     FMA at all;
   - the output row went through memory on every input.

   The intrinsics kernel reached 82 GFLOP/s on a pinned P-core (~57% of
   ~147 peak at 4.6 GHz). This matters for any future hand-written float code
   in this repo: FMAs need `-ffp-contract=fast` or intrinsics. Do not add
   `-ffp-contract=fast` globally, because it would change engine arithmetic and
   the golden digest.
2. **`schedule(static)` is wrong on a hybrid CPU.** An E-core does ~24 GFLOP/s
   against a P-core's ~82, so a static split waits on the E-cores: 6 P-cores
   alone gave 366 GFLOP/s, and adding 8 E-cores dropped that to 272.
   `schedule(dynamic, 4)` gives ~450 on 14–20 threads. **The engine's own
   `#pragma omp parallel for` in `Trainer::doIteration` uses the default
   (static) schedule too**, and may be losing the same way. That is not checked
   yet, and it is a candidate for step 2.

Single-thread numbers must be pinned with `taskset`. Unpinned, a single thread
can land on an E-core, which is how the 24 GFLOP/s "single-thread" number first
appeared.

### Throughput, i7-12700H, 65536 rows

| | 1 thread | 6 | 14 | 20 |
|---|---|---|---|---|
| Mlp (committed) | 82 on a pinned P-core / 24 on an E-core | 308 | 459 | 443 GFLOP/s |
| LiteRT / XNNPACK | 39 (unpinned) | 353 | 488 | 545 GFLOP/s |

XNNPACK is ~20% ahead at full width, probably from a larger register tile and
better packing. That is not worth chasing, because the CPU path is the reference
and fallback, and the GPU is the target. Throughput is flat from 4k to 262k
rows (421–438 GFLOP/s at 20 threads), so the CPU path does not need big
batches.

## Parity on real search states

`selfplay_nn` gained two modes:

- `-` in place of the python runs `Mlp` in-process;
- `MLP_CHECK=<file.mlp>` evaluates every batch both ways and still plays from
  the tflite outputs.

100 games, 1600 searches, seed 4242, 3,507,507 rows:

| | max abs diff |
|---|---|
| value | 2.4e-7 |
| probability | 7.7e-7 |
| rows where the top policy move differs | **0** |

These are float-rounding differences (different accumulation order), so ReLU on
the hidden layers is confirmed.

## What the in-process network buys

Same seed, 100 games, 1600 searches:

| | engine | network | turns/game |
|---|---|---|---|
| tflite over the pipe, 8 threads | 1.56 s | 23.3 s | 27.59 |
| Mlp in-process, 8 threads | 1.11 s | 3.12 s | 27.67 |
| Mlp in-process, 20 threads | 0.72 s | 2.61 s | 27.67 |

- 8 and 20 threads play identical games (same request count), so the backend is
  deterministic regardless of thread count.
- The in-process games differ slightly from the tflite games, because outputs
  differ in the 7th decimal and that occasionally changes a visit count.
- Network share with in-process CPU inference: **78%**.

Scaled to a 25k-game generation (~0.88B rows, 1600 searches) on this laptop's
CPU: about 11 minutes of network plus 3 of engine for self-play, against 40
minutes on the cloud box in 2023. That is only an estimate: the requests per
turn and the engine time come from 100-game runs.

## Not changed

- The golden digest is unchanged (`d7581196f9d30591`). `mlp.o` is linked but
  nothing in the engine calls it.
- `corintho_ai/python/setup.py` and the top-level `CMakeLists.txt` do not build
  `mlp.cpp`. The Cython path is on its way out (see the plan above), so neither
  was touched.
