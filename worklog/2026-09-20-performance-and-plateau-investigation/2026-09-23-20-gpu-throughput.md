# 20 — GPU throughput: the GPU outruns the engine at fp32, and the network does not tolerate TF32 or fp16

This is step 3a of the inference plan (entry 18): empirical GPU numbers before
building a C++ GPU backend. Raw outputs are in
`data/gpu-throughput/runs.txt`.

Setup:

- Environments are uv venvs, as Max asked: `uv venv` plus `uv pip install`.
  - `torch` 2.14.0+cu130 and `onnxruntime-gpu` 1.30.0. The pip wheels bring
    their own CUDA libraries, so no system CUDA toolkit is needed.
  - A separate `tensorflow` 2.21 venv, for the Keras check below.
- The uv cache holds 38 GB after this. Not all of that is from this project.
- `bench/gpu_bench.py` loads a `.mlp` file into an identical PyTorch model.

## Throughput by batch size (fp32, TF32 off)

| rows | compute ns/row | end to end ns/row | ONNX Runtime CUDA, end to end |
|---|---|---|---|
| 512 | 1094 | 1183 | 416 |
| 2,048 | 275 | 327 | 256 |
| 8,192 | 148 | 212 | 234 |
| 16,384 | 127 | 192 | 213 |
| 32,768 | 118 | 180 | 197 |
| 65,536 | 114 | 174 | 191 |
| 262,144 | 111 | 171 | 186 |

"End to end" is pinned host memory → GPU → forward → pinned host memory,
synchronized. That is what the engine pays per batch.

- **It saturates at ~16k–32k rows**, within 5–12% of the plateau. That is
  1,000–2,000 games per batch at 16 rows each.
- **Compute is memory-bound, not FLOP-bound.** 266 kFLOP in 111 ns is
  ~2.4 TFLOP/s. Each of the 14 layers is its own kernel, and at width 100
  every layer writes its activations to GPU memory and reads them back. It is
  the same problem Keras had on CPU (entry 01). A fused kernel, with the whole
  MLP per tile in shared memory, could go several times faster, but there is
  no need yet (see below).
- **Transfer is ~60 ns/row**: 280 bytes in and 388 out per row, at about
  11 GB/s. Packing inputs to bytes and returning fewer bytes could cut it.
  Not needed yet either.
- **Small batches are dominated by launch overhead.** PyTorch at 512 rows
  spends ~0.6 ms per call on ~30 kernel launches. ONNX Runtime has much less
  overhead per call: 416 against 1,183 ns/row at 512 rows. At large batches
  they are equal.

### Against the engine

- A 25k-game generation is ~888M rows (entry 19). At ~175 ns/row that is
  **~2.6 minutes of GPU time**.
- With the dynamic schedule on 20 threads, the engine is ~3.4 minutes.
- **The engine is the bottleneck once inference is on the GPU.** So the
  alternating-groups overlap matters more than a faster GPU kernel.
- For comparison, the CPU `Mlp` on all 20 threads is 530–700 ns/row, and it
  takes the cores the engine needs.

### TF32 and fp16: not usable with this network

Largest difference from fp32 on random 0/1 states:

| rows | TF32 value / prob | fp16 value / prob | fp16 end to end ns/row |
|---|---|---|---|
| 8,192 | 0.038 / 0.091 | 0.055 / 0.093 | 124 |
| 65,536 | 0.069 / 0.15 | 0.052 / 0.17 | 118 |
| 262,144 | 0.078 / 0.16 | 0.085 / 0.18 | 115 |

Both formats have a 10-bit mantissa, and both put probabilities off by up to
0.18. That is not rounding noise; this 12-layer network amplifies small errors.
fp16 would be ~31% faster end to end, which is not worth it while the GPU is
not the bottleneck.

Caveats:

- Random states are not real positions. Real positions may behave better.
- A PyTorch-trained network could be made tolerant of reduced precision, for
  example by training with mixed precision.
- Revisit either one if the GPU ever becomes the bottleneck.

## The tflite files are faithful to the Keras models

Entry 18 left open whether `model_93.tflite` is the same function as the Keras
SavedModel of gen 93. In the TensorFlow venv, `tf.saved_model.load` with the
`serving_default` signature, on 4,096 random 0/1 states:

- value: max |Keras − tflite| **3.1e-5**;
- policy: max |Keras − tflite| **6.8e-5**.

That is the rounding of folding BatchNorm into the weights: the larger of the
two differences is ~100× `Mlp`-vs-tflite, and still far below anything that
changes play. Script: `data/gpu-throughput/keras_vs_tflite.py`.

**So `.mlp` / tflite weights can initialize a PyTorch model of gen 93**, for
the supervised experiments Max plans.

## P-cores only vs all cores (engine, dynamic schedule)

Max asked whether to guarantee P-cores. `taskset` or `sched_setaffinity` only
restricts one process and changes nothing system-wide. But on 1,000 games:

| cores | engine s |
|---|---|
| all 20 threads | 7.98 / 8.15 |
| P-cores only, 12 threads | 10.21 / 10.34 |
| 6 P-cores, no hyperthreads | 13.74 / 13.84 |

The E-cores add ~22% throughput once the schedule is dynamic, so use all
cores. Pin only for single-thread measurement, and perhaps for the thread that
feeds the GPU. That second one is untested.

## Next

- **The C++ GPU backend.** The two ways to get a C++ runtime here:
  - ONNX Runtime from its GitHub release tarball (headers plus
    `libonnxruntime.so`), with the CUDA/cuDNN libraries taken from the pip
    `nvidia-*` wheels;
  - libtorch, which the `torch` wheel already contains: headers, `libtorch.so`
    and CMake config.
- **Rolling game starts**, to replace the staggered-start trick, with games in
  flight around 1k–2k per group.
- **Alternating two game groups**, so the engine searches one group while the
  GPU evaluates the other. Measure against a single group.
