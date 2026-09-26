# 01 — Self-play is GPU-bound; TensorRT and compact I/O

2026-09-25. Commits `3571532` (costs), `f684a79` (breakdown), `a3fe0ae`
(TensorRT), `0bbdc91` (compact I/O), `a27f214` (run.py default). Data:
`data/inference-cost.txt`, `data/gpu-limits.txt`.

## Why this came first

Starting on architectures, I measured what candidates would cost at
inference (`pipeline/arch/cost.py`, random weights, `corintho_play bench`,
16k rows). Under the CUDA provider: current 12x100 MLP 254 ns/row; residual
MLP 256x4 437 (fp16 285); 4x4 conv ResNet 32x4 1202 (fp16 646); a small
transformer (17 tokens, d 64, 4 layers) 8746 (fp16 3113). Conv nets and
attention do badly on ONNX Runtime at this size.

The developer was surprised that the GPU mattered: worklog entry 20 of the
previous investigation found the engine was the bottleneck. Since then the
engine got >2x faster (entries 22–28, clang), so the balance moved.

## Evidence that self-play is GPU-bound

4,000 games, in-flight 2000, 2 groups, gen_93:
- engine 30 s, **waiting on the GPU ~20 s**, wall 51 s;
- 18 or 16 engine threads instead of 20: no change (it is not CPU contention
  starving the thread that drives the GPU);
- fp16 export of the same network (call 4.0 -> 3.4 ms): wall -7.5%. A
  shorter call shortens the run, so the call is the limit.

Per 16k-row call (4.0 ms), measured in parts (`arch/gpu_limits.py`):
compute 1.98 ms (the network is tiny, ~2 TFLOPS: many separate kernels each
writing activations to memory), copies 0.4 + 0.6 ms (fp32 in, fp32 out),
host memcpy 0.4 ms, the rest unattributed.

## TensorRT

`tensorrt-cu13==10.16.1.11` (ORT 1.30's provider needs TensorRT 10; 4.3 GB of
wheels, installed by `setup.sh`). The driver takes `trt:` / `trt16:` before an
`.onnx` path. Engines build in ~3.5 s on first use and are cached in
`trt_cache/` beside the model.

- Compute 1.98 -> 0.86 ms (fp32) / 0.77 ms (fp16); driver call 3.97 -> 2.77.
- **TensorRT's fp32 is not exact:** against CUDA fp32 on 16k real states,
  value max 2.5e-2 / mean 1.1e-3, top move differs on 0.64%. This matches the
  TF32 differences seen in entry 20 of the previous worklog; TensorRT enables
  TF32 by default and ORT exposes no switch for it.
- fp16 is further off (top move 5.0%) and no faster in self-play: not used.
- Gotcha: ORT's TensorRT options want `True`/`False`, not `1`/`0`; a bad
  value makes the Python API silently fall back to CUDA.

Self-play wall 51-53 -> 44.7 s.

## Compact I/O

The ONNX model takes `uint8` states (4x the engine's floats; all inputs are
multiples of 0.25, so exact) and returns the policy as fp16; the value stays
fp32. 262 bytes a row cross the bus instead of 664. The backend detects the
format from the model's input type, so older fp32 models still work.
`model.export_onnx(compact=True)`, and `model.save` now always exports
compact. Precision: the search quantizes priors to 1/511 of the largest
(`Node::kMaxProbability`) after adding noise, so fp16's 2^-11 relative error
moves a prior by at most one unit, rarely.

- Call 2.77 -> 1.94 ms (TensorRT). Against fp32 I/O on 14.4M rows of real
  search (`--check`): value 1e-5, policy 2.5e-4, top move differs 0.07%.
- **Self-play wall 40.5 s (from 51-53 s, -22%)**, wait 9.3 s, GPU busy ~40%.
  With a free GPU the floor is the engine's 30 s.

## Strength check

Same network, TensorRT + compact (new) vs CUDA fp32 (best), 1,600 games per
seed: 810W 62D 728L (seed 1; Wilson lower bound 0.502 — chance, see below),
768W 42D 790L (2), 752W 49D 799L (3). Total 2330W 153D 2317L: no measurable
difference. Seed 1 alone would have passed the promotion gate: a reminder
that a single 1,600-game match at 95% passes a null about 1 time in 40.

## Adopted

`run.py --backend trt` is the default; a run whose `config.json` predates the
option keeps `cuda` on resume. Smoke run (2000 games, fit, 200 test games)
completed on TensorRT with compact models.

## Consequences for the architecture work

The GPU is now ~40% busy in self-play, so a larger network has room before
it becomes the limit again. The candidate costs above were measured under
CUDA with fp32 I/O and should be re-measured under TensorRT.
