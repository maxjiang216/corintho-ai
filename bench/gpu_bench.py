"""GPU throughput of the policy/value network by batch size.

  python gpu_bench.py model.mlp [--onnx out.onnx]

Loads a .mlp file (see corintho_ai/cpp/include/mlp.h) into an equivalent
PyTorch model and times, for each batch size:

  compute  the forward pass alone, input already on the GPU
  e2e      what the engine would pay: pinned host input -> GPU -> forward ->
           pinned host output (values and probabilities), synchronized

in fp32 (TF32 off), TF32 and fp16. TF32 and fp16 also report their largest
differences from fp32. With --onnx, the model is exported to ONNX and ONNX Runtime's CUDA
provider is timed end to end as well, as a preview of a C++ ORT backend.

Inputs are random 0/1, like game states; the work does not depend on values.
Timings are medians over repeated calls after warm-up.
"""

import argparse
import statistics
import struct
import time

import numpy as np
import torch
from torch import nn

BATCHES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


class Net(nn.Module):
    def __init__(self, input_size, width, depth, moves):
        super().__init__()
        sizes = [input_size] + [width] * depth
        self.hidden = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(depth)
        )
        self.value = nn.Linear(width, 1)
        self.policy = nn.Linear(width, moves)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return torch.tanh(self.value(x)).squeeze(-1), torch.softmax(
            self.policy(x), dim=-1
        )


def load_mlp(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"CMLP"
    version, input_size, width, depth, moves = struct.unpack_from(
        "<5I", data, 4
    )
    assert version == 1
    floats = np.frombuffer(data, dtype="<f4", offset=24)
    net = Net(input_size, width, depth, moves)
    pos = 0

    def take(layer):
        nonlocal pos
        w = layer.weight
        n = w.numel()
        w.data.copy_(torch.from_numpy(floats[pos : pos + n].reshape(w.shape)))
        pos += n
        b = layer.bias
        b.data.copy_(torch.from_numpy(floats[pos : pos + b.numel()].copy()))
        pos += b.numel()

    for layer in net.hidden:
        take(layer)
    take(net.value)
    take(net.policy)
    assert pos == floats.size
    return net.eval(), input_size, moves


def median_time(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def reps_for(batch):
    return max(5, min(200, 2_000_000 // batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--onnx", default=None)
    args = parser.parse_args()

    dev = torch.device("cuda")
    net, input_size, moves = load_mlp(args.model)
    net = net.to(dev)
    net16 = Net(input_size, net.value.in_features, len(net.hidden), moves)
    net16.load_state_dict(net.state_dict())
    net16 = net16.to(dev).half().eval()
    print(
        f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
        f"CUDA {torch.version.cuda}"
    )

    rng = np.random.default_rng(1)
    modes = ["fp32", "tf32", "fp16"]
    print(
        f"{'rows':>7} {'mode':>5} {'compute ns/row':>15} {'e2e ns/row':>11} "
        f"{'e2e Mrows/s':>12} {'max diff v/p vs fp32':>21}"
    )
    for batch in BATCHES:
        host_in = torch.from_numpy(
            rng.integers(0, 2, (batch, input_size)).astype(np.float32)
        ).pin_memory()
        host_v = torch.empty(batch, dtype=torch.float32).pin_memory()
        host_p = torch.empty(batch, moves, dtype=torch.float32).pin_memory()
        dev_in = host_in.to(dev)
        reps = reps_for(batch)
        ref_v = ref_p = None
        for mode in modes:
            torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
            model = net16 if mode == "fp16" else net

            @torch.inference_mode()
            def compute():
                x = dev_in.half() if mode == "fp16" else dev_in
                return model(x)

            @torch.inference_mode()
            def e2e():
                x = host_in.to(dev, non_blocking=True)
                if mode == "fp16":
                    x = x.half()
                v, p = model(x)
                host_v.copy_(v.float(), non_blocking=True)
                host_p.copy_(p.float(), non_blocking=True)

            t_compute = median_time(compute, reps)
            t_e2e = median_time(e2e, reps)
            diff = ""
            with torch.inference_mode():
                v, p = compute()
                if mode == "fp32":
                    ref_v, ref_p = v.float(), p.float()
                else:
                    dv = (v.float() - ref_v).abs().max().item()
                    dp = (p.float() - ref_p).abs().max().item()
                    diff = f"{dv:.1e} / {dp:.1e}"
            print(
                f"{batch:>7} {mode:>5} {t_compute / batch * 1e9:>15.2f} "
                f"{t_e2e / batch * 1e9:>11.2f} "
                f"{batch / t_e2e / 1e6:>12.1f} {diff:>21}"
            )

    if args.onnx:
        import onnxruntime as ort

        torch.backends.cuda.matmul.allow_tf32 = False
        dummy = torch.zeros(1, input_size, device=dev)
        torch.onnx.export(
            net,
            dummy,
            args.onnx,
            input_names=["state"],
            output_names=["value", "policy"],
            dynamic_axes={
                "state": {0: "n"},
                "value": {0: "n"},
                "policy": {0: "n"},
            },
            opset_version=17,
            dynamo=False,
        )
        sess = ort.InferenceSession(
            args.onnx, providers=["CUDAExecutionProvider"]
        )
        assert "CUDAExecutionProvider" in sess.get_providers()
        print(
            f"\nONNX Runtime {ort.__version__} CUDA provider, end to end "
            "(numpy in, numpy out)"
        )
        for batch in BATCHES:
            x = rng.integers(0, 2, (batch, input_size)).astype(np.float32)

            def run():
                sess.run(None, {"state": x})

            t = median_time(run, reps_for(batch))
            print(
                f"{batch:>7}  ort  e2e {t / batch * 1e9:>8.2f} ns/row  "
                f"{batch / t / 1e6:>7.1f} Mrows/s"
            )


if __name__ == "__main__":
    main()
