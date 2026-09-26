# GPU training pipeline

Self-play training on one machine with an NVIDIA GPU, replacing the Cython +
Keras pipeline in `corintho_ai/python/` (which is untouched and still builds).
The design and the measurements behind it are in the worklog:
`worklog/2026-09-20-performance-and-plateau-investigation/`, entries 21
(design) and 29 (this implementation).

## Setup

Needs the NVIDIA driver, `uv`, `clang-20` and `libomp-20-dev`. No system CUDA
toolkit: the pip wheels bring CUDA 13 and cuDNN 9.

    ./setup.sh

It creates `.venv/` (PyTorch, ONNX Runtime, numpy, TensorRT 10), downloads the ONNX Runtime
C++ release into `third_party/` (sha256-checked), builds
`build/corintho_play`, and converts the starting model `models/gen_93.npz`
into `.pt`, `.onnx` and `.mlp`.

## Running

    .venv/bin/python run.py --name overnight --generations 60

Each generation: self-play (the best model, 25k games), fit (from the latest
generation), test (1,600 games against the best), promotion and learning-rate
update. Output goes to `runs/<name>/`:

- `progress.log`: one line per step, with the CPU temperature;
- `generations.tsv`: one row per generation (result, rating, validation loss,
  learning rate, first-player score, phase times);
- `state.json`: best generation, learning rate, ratings;
- `gen_N/`:
  - `model.{pt,onnx,mlp}`, `model_fit.json`, `model_loss.csv`;
  - `samples/selfplay.json` and the first games' logs;
  - `test/test.json`, `test/score.txt`;
  - `timing.json`.

A stopped run resumes where it stopped: rerun the same command. A run's
settings are fixed when it starts (`config.json`). Samples (~3.8 GB per
generation) are deleted once no later fit needs them, unless
`--keep-samples` is given.

Defaults are gen_93's settings: 1,600 searches, 16 per evaluation, c_puct 3,
epsilon 0.25, learning rate 5e-6, 10 epochs, batch 2048, 1,600 test games,
95% Wilson gate. Also:

- `--games`, `--test-games`;
- `--in-flight` and `--groups` (2 x 2,000);
- `--old-gens` (0; see below);
- `--backend` (`trt`, the default: ONNX Runtime's TensorRT provider; `cuda`:
  its CUDA provider). Runs started before this option keep `cuda`.

Models are exported "compact": uint8 states in, fp16 policy out (fp32 value),
which with TensorRT cuts self-play ~22% against CUDA with fp32 I/O
(`worklog/2026-09-25-nn-architectures/`). TensorRT's fp32 allows TF32; a
same-network match against the CUDA provider was even (2330W 153D 2317L).
The driver takes `trt:` / `trt16:` before a `.onnx` path; TensorRT engines
are built on first use (~3 s) and cached in `trt_cache/` beside the model.

## Pieces

| file | what |
|---|---|
| `cpp/corintho_play.cpp` | the driver: `train` (self-play to `.npy`), `test` (two models), `bench` (network throughput by batch size) |
| `cpp/backend.{h,cpp}` | network backends: ONNX Runtime CUDA or TensorRT (`.onnx`, fp32 or compact I/O) and the CPU `Mlp` (`.mlp`) |
| `model.py` | the network in PyTorch (the Keras architecture) and its exports |
| `fit.py` | training, equivalent to the Keras `fit` in `main.pyx` |
| `run.py` | the generation loop |
| `keras_to_npz.py` | one-time conversion of a Keras generation (needs TensorFlow) |
| `arch/` | architecture experiments: `cost.py` (candidate inference costs), `gpu_limits.py` (a call's parts), `trt_test.py` (providers compared) |

## How it compares with the Cython pipeline

- **The same:**
  - the engine, the search settings and the sample format (8 symmetries);
  - the loss and optimizer: Adam, carried between generations;
  - the validation split: the last 30%;
  - the learning-rate schedules;
  - the promotion gate and the rating formula.
- **Different, on purpose:**
  - games are played in chunks of 2,000 per group, two groups overlapping,
    not all 25k at once, which would not fit in memory;
  - the network runs in fp32 on the GPU, matching the CPU network to
    ~2e-4;
  - the ONNX model has BatchNorm folded in.
- **`--old-gens` defaults to 0**, because that is what the Cython pipeline
  did in practice. Its `get_samples()` discarded the `np.concatenate`
  results, so older generations' samples were never used, despite
  `num_old_gens: 2` in gen_93's metadata.

## Checks

- `corintho_play train --model X.mlp --games 300 --in-flight 300 --seed S --digest`
  reproduces `bench/selfplay_nn`'s sample digests (the CPU backend, one
  chunk).
- `--check Y.mlp` evaluates every batch on a second backend and reports the
  largest differences.
- `fit.py --no-graph` trains without the CUDA graph. The weights are
  bit-identical.
