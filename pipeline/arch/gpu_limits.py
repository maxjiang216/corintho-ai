"""Where the time of one network call goes, measured piece by piece.

    python arch/gpu_limits.py models/gen_93.onnx [--rows 16000]

  h2d / d2h   pinned host <-> GPU copies of the call's input and outputs, as
              fp32 (today) and as uint8 input / fp16 outputs (candidates)
  compute     ONNX Runtime CUDA with input and outputs already on the GPU
  host copy   the driver's two memcpys (engine buffer -> pinned -> engine)
  driver      ONNX Runtime bound to host memory, as backend.cpp does it

Medians of repeated synchronized calls after warm-up.
"""
import argparse
import statistics
import time

import numpy as np
import onnxruntime as ort
import torch

S, M = 70, 96


def timed(fn, reps=50):
    for _ in range(5):
        fn()
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--rows", type=int, default=16000)
    a = ap.parse_args()
    n = a.rows
    report = []

    def row(name, s, nbytes=None):
        bw = f"  {nbytes / s / 1e9:6.2f} GB/s" if nbytes else ""
        report.append(
            f"{name:34s} {s * 1e3:7.3f} ms  {s * 1e9 / n:7.1f} ns/row{bw}"
        )

    dev = torch.device("cuda")
    for name, shape, dt in [
        ("h2d states fp32", (n, S), torch.float32),
        ("h2d states uint8", (n, S), torch.uint8),
        ("h2d states bits (12 B)", (n, 12), torch.uint8),
    ]:
        h = torch.zeros(shape, dtype=dt).pin_memory()
        d = torch.empty(shape, dtype=dt, device=dev)
        row(
            name,
            timed(lambda: d.copy_(h, non_blocking=True)),
            h.numel() * h.element_size(),
        )
    for name, dt in [
        ("d2h value+policy fp32", torch.float32),
        ("d2h value+policy fp16", torch.float16),
    ]:
        d = torch.zeros((n, M + 1), dtype=dt, device=dev)
        h = torch.empty((n, M + 1), dtype=dt).pin_memory()
        row(
            name,
            timed(lambda: h.copy_(d, non_blocking=True)),
            h.numel() * h.element_size(),
        )

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        a.model,
        opts,
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
    )
    x = (np.random.default_rng(0).random((n, S)) < 0.3).astype(np.float32)
    xg = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
    vg = ort.OrtValue.ortvalue_from_shape_and_type(
        [n, 1], np.float32, "cuda", 0
    )
    pg = ort.OrtValue.ortvalue_from_shape_and_type(
        [n, M], np.float32, "cuda", 0
    )
    b = sess.io_binding()
    b.bind_ortvalue_input("states", xg)
    b.bind_ortvalue_output("value", vg)
    b.bind_ortvalue_output("policy", pg)
    row("compute (all on GPU)", timed(lambda: sess.run_with_iobinding(b)))

    bh = sess.io_binding()
    v = np.empty((n, 1), np.float32)
    p = np.empty((n, M), np.float32)
    bh.bind_cpu_input("states", x)
    bh.bind_output("value", "cpu", 0, np.float32, [n, 1], v.ctypes.data)
    bh.bind_output("policy", "cpu", 0, np.float32, [n, M], p.ctypes.data)
    row(
        "run bound to host (pageable)",
        timed(lambda: sess.run_with_iobinding(bh)),
    )

    src, dst = np.ones((n, S + M + 1), np.float32), np.empty(
        (n, S + M + 1), np.float32
    )
    row(
        "host memcpy in+out",
        timed(lambda: np.copyto(dst, src)),
        src.nbytes * 2,
    )
    print(f"rows {n}, model {a.model}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
