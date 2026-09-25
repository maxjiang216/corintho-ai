"""The policy/value network in PyTorch, and its exports for the self-play driver.

The architecture is the Keras one from corintho_ai/python/wrapper.py, unchanged:

    12 x (Linear 100 -> ReLU -> BatchNorm)  from 70 inputs
    value head:  Linear 1 -> tanh
    policy head: Linear 96 -> softmax

BatchNorm uses Keras's defaults (epsilon 1e-3, momentum 0.99, which is PyTorch
momentum 0.01), so a model converted from Keras behaves the same.

Exports:
  .pt    state dict plus optimizer state, for continuing training
  .onnx  input "states" [N, 70], outputs "value" [N, 1], "policy" [N, 96];
         read by the driver's ONNX Runtime backend (pipeline/cpp/backend.h)
  .mlp   BatchNorm folded into the following layer, for the CPU network
         (corintho_ai/cpp/include/mlp.h), the driver's fallback and reference

    python model.py convert models/gen_93.npz models/gen_93
"""
import struct
import sys

import numpy as np
import torch
from torch import nn

GAME_STATE_SIZE = 70
NUM_MOVES = 96
WIDTH = 100
DEPTH = 12


class CorinthoNet(nn.Module):
    def __init__(self, depth=DEPTH, width=WIDTH, bn_eps=1e-3):
        super().__init__()
        layers = []
        prev = GAME_STATE_SIZE
        for _ in range(depth):
            layers += [nn.Linear(prev, width), nn.ReLU(),
                       nn.BatchNorm1d(width, eps=bn_eps, momentum=0.01)]
            prev = width
        self.body = nn.Sequential(*layers)
        self.value = nn.Linear(width, 1)
        self.policy = nn.Linear(width, NUM_MOVES)

    def forward(self, states):
        """Value in [-1, 1] and policy probabilities, as the driver consumes."""
        h = self.body(states)
        return torch.tanh(self.value(h)), torch.softmax(self.policy(h), dim=1)

    def heads(self, states):
        """Value and policy logits, for training (log-softmax is stabler)."""
        h = self.body(states)
        return torch.tanh(self.value(h)), self.policy(h)

    def linears(self):
        return [m for m in self.body if isinstance(m, nn.Linear)]

    def norms(self):
        return [m for m in self.body if isinstance(m, nn.BatchNorm1d)]


def from_npz(path):
    """A CorinthoNet with weights from keras_to_npz.py output."""
    z = np.load(path)
    net = CorinthoNet(bn_eps=float(z["bn_epsilon"]))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    with torch.no_grad():
        for i, (lin, bn) in enumerate(zip(net.linears(), net.norms())):
            lin.weight.copy_(t(z[f"dense{i}_w"].T))  # Keras stores [in, out]
            lin.bias.copy_(t(z[f"dense{i}_b"]))
            bn.weight.copy_(t(z[f"bn{i}_gamma"]))
            bn.bias.copy_(t(z[f"bn{i}_beta"]))
            bn.running_mean.copy_(t(z[f"bn{i}_mean"]))
            bn.running_var.copy_(t(z[f"bn{i}_var"]))
        for name in ("value", "policy"):
            head = getattr(net, name)
            head.weight.copy_(t(z[f"{name}_w"].T))
            head.bias.copy_(t(z[f"{name}_b"]))
    net.eval()
    if "check_x" in z:
        with torch.no_grad():
            v, p = net(t(z["check_x"]))
        dv = np.abs(v.numpy() - z["check_value"]).max()
        dp = np.abs(p.numpy() - z["check_policy"]).max()
        print(f"{path}: max |value - keras| {dv:.2e}, max |policy - keras| {dp:.2e}")
        assert dv < 1e-4 and dp < 1e-4, "conversion does not match Keras"
    return net


def folded(net):
    """The inference-time MLP: BatchNorm folded into the next linear layer.

    Hidden layer i computes bn_i(relu(W_i x + b_i)), and bn_i(z) = a_i z + c_i
    with a = gamma / sqrt(var + eps), c = beta - mean a. So layer i + 1 (or a
    head) sees W' = W diag(a), b' = b + W c. Returns [(W [out][in], b [out])]
    for the hidden layers, then value, then policy, as mlp.h expects.
    """
    lins, bns = net.linears(), net.norms()
    out = []
    scale = shift = None
    with torch.no_grad():
        for lin, bn in zip(lins, bns):
            w, b = lin.weight.double(), lin.bias.double()
            if scale is not None:
                b = b + w @ shift
                w = w * scale
            out.append((w, b))
            a = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
            scale, shift = a, bn.bias.double() - bn.running_mean.double() * a
        for head in (net.value, net.policy):
            w, b = head.weight.double(), head.bias.double()
            out.append((w * scale, b + w @ shift))
    return [(w.float().numpy(), b.float().numpy()) for w, b in out]


def export_mlp(net, path):
    layers = folded(net)
    with open(path, "wb") as f:
        f.write(b"CMLP")
        f.write(struct.pack("<5I", 1, GAME_STATE_SIZE, WIDTH, len(layers) - 2,
                            NUM_MOVES))
        for w, b in layers:
            f.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(b, dtype="<f4").tobytes())


def mlp_forward(layers, x):
    """numpy forward pass of folded layers, the same maths as Mlp (checks)."""
    h = x.astype(np.float64)
    for w, b in layers[:-2]:
        h = np.maximum(h @ w.T + b, 0.0)
    (wv, bv), (wp, bp) = layers[-2:]
    logits = h @ wp.T + bp
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return np.tanh(h @ wv.T + bv), p / p.sum(axis=1, keepdims=True)


class FoldedNet(nn.Module):
    """Inference-only network with BatchNorm folded in (see folded()).

    Exported to ONNX instead of CorinthoNet: 12 BatchNormalization nodes on
    [N, 100] tensors ran ~9x slower than the matrix products on ONNX Runtime's
    CUDA provider (1,640 against ~190 ns/row, worklog entry 29), and folding
    leaves only Gemm + Relu, the same maths as the .mlp file.
    """

    def __init__(self, layers):
        super().__init__()
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        self.hidden = nn.ModuleList()
        for w, b in layers[:-2]:
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data, lin.bias.data = t(w), t(b)
            self.hidden.append(lin)
        self.value, self.policy = nn.Linear(WIDTH, 1), nn.Linear(WIDTH, NUM_MOVES)
        for head, (w, b) in zip((self.value, self.policy), layers[-2:]):
            head.weight.data, head.bias.data = t(w), t(b)

    def forward(self, states):
        h = states
        for lin in self.hidden:
            h = torch.relu(lin(h))
        return torch.tanh(self.value(h)), torch.softmax(self.policy(h), dim=1)


def export_onnx(net, path):
    folded_net = FoldedNet(folded(net.eval().float().cpu())).eval()
    torch.onnx.export(
        folded_net, (torch.zeros(16, GAME_STATE_SIZE),), path,
        input_names=["states"], output_names=["value", "policy"],
        dynamic_axes={"states": {0: "n"}, "value": {0: "n"}, "policy": {0: "n"}},
        opset_version=17, dynamo=False)


def check_exports(net, prefix):
    """Folded .mlp maths and the ONNX file both match the PyTorch model."""
    import onnxruntime as ort
    rng = np.random.default_rng(1)
    x = (rng.random((512, GAME_STATE_SIZE)) < 0.3).astype(np.float32)
    with torch.no_grad():
        v, p = net.eval()(torch.from_numpy(x))
    v, p = v.numpy(), p.numpy()
    mv, mp = mlp_forward(folded(net), x)
    sess = ort.InferenceSession(f"{prefix}.onnx", providers=["CPUExecutionProvider"])
    ov, op = sess.run(None, {"states": x})
    diffs = {"mlp_value": np.abs(mv - v).max(), "mlp_policy": np.abs(mp - p).max(),
             "onnx_value": np.abs(ov - v).max(), "onnx_policy": np.abs(op - p).max()}
    print("export check vs torch: " + ", ".join(f"{k} {d:.2e}" for k, d in diffs.items()))
    assert max(diffs.values()) < 1e-4, "an export does not match the model"


def save(net, prefix, optimizer=None, extra=None):
    """prefix.pt (weights + optimizer), prefix.onnx, prefix.mlp, all checked."""
    state = {"model": net.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, f"{prefix}.pt")
    export_onnx(net, f"{prefix}.onnx")
    export_mlp(net, f"{prefix}.mlp")
    check_exports(net, prefix)


def load(prefix):
    state = torch.load(f"{prefix}.pt", map_location="cpu", weights_only=False)
    net = CorinthoNet()
    net.load_state_dict(state["model"])
    return net.eval(), state


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "convert":
        save(from_npz(sys.argv[2]), sys.argv[3])
    else:
        sys.exit(__doc__)
