"""ONNX Runtime's TensorRT provider against its CUDA provider: speed and accuracy.

    LD_LIBRARY_PATH=<tensorrt_libs> python arch/trt_test.py MODEL.onnx STATES.npy

Times one call with data already on the GPU (compute) and bound to host memory
(as the driver does), and compares outputs on real self-play states with the
CUDA provider in fp32. Engines are cached in build/trt_cache.
"""
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from gpu_limits import timed  # noqa: E402

S, M, MAX = 70, 96, 32768


def session(model, kind):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cuda = ("CUDAExecutionProvider", {"use_tf32": "0"})
    if kind == "cuda":
        return ort.InferenceSession(model, opts, providers=[cuda])
    trt = {
        "trt_fp16_enable": "True" if kind == "trt16" else "False",
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": "build/trt_cache",
        "trt_timing_cache_enable": "True",
        "trt_profile_min_shapes": "states:1x70",
        "trt_profile_opt_shapes": "states:16000x70",
        "trt_profile_max_shapes": f"states:{MAX}x70",
        "trt_builder_optimization_level": "5",
    }
    return ort.InferenceSession(
        model, opts, providers=[("TensorrtExecutionProvider", trt), cuda]
    )


def main():
    model, states = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(0)
    all_x = np.load(states, mmap_mode="r")
    pick = np.sort(rng.choice(all_x.shape[0], 16000, replace=False))
    x = np.ascontiguousarray(all_x[pick], dtype=np.float32)
    n = x.shape[0]
    ref = None
    for kind in ("cuda", "cuda", "trt32", "trt16"):
        sess = session(model, kind)
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
        comp = timed(lambda: sess.run_with_iobinding(b))
        v = np.empty((n, 1), np.float32)
        p = np.empty((n, M), np.float32)
        bh = sess.io_binding()
        bh.bind_cpu_input("states", x)
        bh.bind_output("value", "cpu", 0, np.float32, [n, 1], v.ctypes.data)
        bh.bind_output("policy", "cpu", 0, np.float32, [n, M], p.ctypes.data)
        host = timed(lambda: sess.run_with_iobinding(bh))
        sess.run_with_iobinding(bh)
        line = (
            f"{kind:6s} compute {comp * 1e3:6.3f} ms ({comp * 1e9 / n:5.1f} ns/row)"
            f"  host-bound {host * 1e3:6.3f} ms ({host * 1e9 / n:5.1f} ns/row)"
        )
        if ref is None:
            ref = (v.copy(), p.copy())
        else:
            dv = np.abs(v - ref[0])
            dp = np.abs(p - ref[1])
            top = (p.argmax(1) != ref[1].argmax(1)).mean()
            line += (
                f"  |dv| max {dv.max():.1e} mean {dv.mean():.1e}"
                f"  |dp| max {dp.max():.1e}  top-move differs {top:.2%}"
            )
        print(line, flush=True)


if __name__ == "__main__":
    main()
