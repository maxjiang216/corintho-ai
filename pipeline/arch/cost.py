"""Inference cost of candidate architectures on the driver's ONNX Runtime backend.

    python arch/cost.py OUTDIR       (then corintho_play bench on each file)

Random weights: cost does not depend on values. Every file has the driver's
interface: states [N, 70] float32 -> value [N, 1], policy [N, 96] float32.
fp16 variants cast inside the graph, so the driver is unchanged.
"""
import sys

import torch
from torch import nn

S, M = 70, 96


class Mlp(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        sizes = [S] + [width] * depth
        self.body = nn.Sequential(
            *[
                m
                for i in range(depth)
                for m in (nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU())
            ]
        )
        self.v, self.p = nn.Linear(width, 1), nn.Linear(width, M)

    def forward(self, x):
        h = self.body(x)
        return torch.tanh(self.v(h)), torch.softmax(self.p(h), 1)


class ResMlp(nn.Module):
    """Pre-activation residual blocks; BatchNorm folds away at export."""

    def __init__(self, width, blocks):
        super().__init__()
        self.inp = nn.Linear(S, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(), nn.Linear(width, width),
                nn.ReLU(), nn.Linear(width, width),
            )
            for _ in range(blocks)
        )
        self.v, self.p = nn.Linear(width, 1), nn.Linear(width, M)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        h = torch.relu(h)
        return torch.tanh(self.v(h)), torch.softmax(self.p(h), 1)


class Conv(nn.Module):
    """ResNet on the 4x4 board: 4 bit planes + 6 broadcast reserve planes."""

    def __init__(self, ch, blocks):
        super().__init__()
        self.inp = nn.Conv2d(10, ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1),
                nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1),
            )
            for _ in range(blocks)
        )
        self.v = nn.Sequential(nn.Linear(ch * 16, 64), nn.ReLU(), nn.Linear(64, 1))
        self.p = nn.Linear(ch * 16, M)

    def forward(self, x):
        n = x.shape[0]
        board = x[:, :64].reshape(n, 4, 4, 4).permute(0, 3, 1, 2)
        res = x[:, 64:].reshape(n, 6, 1, 1).expand(n, 6, 4, 4)
        h = self.inp(torch.cat([board, res], 1))
        for b in self.blocks:
            h = h + b(h)
        h = torch.relu(h).flatten(1)
        return torch.tanh(self.v(h)), torch.softmax(self.p(h), 1)


class Attn(nn.Module):
    """Transformer over 16 space tokens + 1 reserve token."""

    def __init__(self, d, layers, heads=4):
        super().__init__()
        self.space = nn.Linear(4, d)
        self.pos = nn.Parameter(torch.randn(16, d) * 0.02)
        self.res = nn.Linear(6, d)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), o=nn.Linear(d, d),
                n2=nn.LayerNorm(d),
                ff=nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)),
            )))
        self.heads, self.d = heads, d
        self.v = nn.Linear(d, 1)
        self.p = nn.Linear(17 * d, M)

    def forward(self, x):
        n = x.shape[0]
        t = self.space(x[:, :64].reshape(n, 16, 4)) + self.pos
        h = torch.cat([self.res(x[:, 64:]).unsqueeze(1), t], 1)
        H, dh = self.heads, self.d // self.heads
        for L in self.layers:
            q, k, v = L["qkv"](L["n1"](h)).reshape(n, 17, 3, H, dh).permute(2, 0, 3, 1, 4)
            a = torch.softmax(q @ k.transpose(-1, -2) / dh**0.5, -1) @ v
            h = h + L["o"](a.transpose(1, 2).reshape(n, 17, self.d))
            h = h + L["ff"](L["n2"](h))
        return torch.tanh(self.v(h[:, 0])), torch.softmax(self.p(h.flatten(1)), 1)


class Half(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net.half()

    def forward(self, x):
        v, p = self.net(x.half())
        return v.float(), p.float()


CANDIDATES = {
    "mlp100x12": lambda: Mlp(100, 12),
    "mlp256x6": lambda: Mlp(256, 6),
    "res256x4": lambda: ResMlp(256, 4),
    "res512x4": lambda: ResMlp(512, 4),
    "conv32x4": lambda: Conv(32, 4),
    "conv64x6": lambda: Conv(64, 6),
    "attn64x4": lambda: Attn(64, 4),
}


def export(net, path):
    torch.onnx.export(
        net.eval(), (torch.zeros(16, S),), path,
        input_names=["states"], output_names=["value", "policy"],
        dynamic_axes={k: {0: "n"} for k in ("states", "value", "policy")},
        opset_version=17, dynamo=False,
    )


if __name__ == "__main__":
    out = sys.argv[1]
    for name, make in CANDIDATES.items():
        torch.manual_seed(0)
        net = make()
        params = sum(p.numel() for p in net.parameters())
        print(f"{name}\t{params}")
        export(net, f"{out}/{name}_fp32.onnx")
        export(Half(make()), f"{out}/{name}_fp16.onnx")
