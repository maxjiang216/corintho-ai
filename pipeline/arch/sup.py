"""Supervised architecture experiments on a fixed dataset (arch/dataset.py).

    python arch/sup.py --data runs/sup/full2.npz --model mlp --width 100 \
        --depth 12 [--mask] [--lines any|type] [--legal-input] [--epochs 20]
    python arch/sup.py --data runs/sup/full2.npz --reference models/gen_93

Trains from scratch on the training sources (default: all but the last) and
evaluates on the last source (a later generation, so no position's
symmetric copies straddle the split). --reference evaluates an existing
model instead of training.

Every model is scored the way the search uses it: the policy renormalized
over legal moves (TrainMC::setProbs keeps only legal moves). Metrics:
  value_mse        against the game result
  policy_ce        cross-entropy of the search's visit distribution against
                   the legal-renormalized policy
  top1             the policy's best legal move is the most-visited move
  loss             value_mse + 0.25 policy_ce (the training loss weights)

Options:
  --mask         train with softmax over legal moves only (the search's view);
                 without it the softmax covers all 96 moves, as today
  --sym          a random board symmetry per sample and step (default on);
                 --no-sym trains on the stored orientation only
  --lines        extra inputs: "any" = 16 flags, space is in a line; "type" =
                 48, the same by the line's top type
  --legal-input  extra inputs: the 96-move legal mask

Results are appended to runs/sup/results.jsonl.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import symmetries  # noqa: E402

S, M = 70, 96
VALUE_WEIGHT, POLICY_WEIGHT = 1.0, 0.25


class Data:
    """The dataset on the GPU, with symmetric views and input features."""

    def __init__(self, path, device, lines, legal_input):
        z = np.load(path)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.states = t(z["states"])
        self.values = t(z["values"])
        self.policies = t(z["policies"])
        legal = np.unpackbits(z["legal"], 1, bitorder="little")[:, :M]
        self.legal = t(legal.astype(bool))
        if lines is not None:  # [n, 3 types, 16 spaces]; skipped when
            # unused, to leave GPU memory for larger datasets
            bits = (
                z["lines"][:, :, None].astype(np.int32) >> np.arange(16)
            ) & 1
            self.lines = t(bits.astype(np.uint8))
        self.source = t(z["source"])
        self.dirs = [str(d) for d in z["dirs"]]
        st, mv, sp = symmetries()
        self.sym_state = t(st)
        self.sym_move = t(mv)
        self.sym_space = t(sp)
        self.line_mode, self.legal_input = lines, legal_input

    @property
    def inputs(self):
        extra = {None: 0, "any": 16, "type": 48}[self.line_mode]
        return S + extra + (M if self.legal_input else 0)

    def batch(self, idx, sym):
        """Inputs, value, policy and legal mask of rows idx, each row in its
        own symmetry sym (int64 [len(idx)], 0 = as stored)."""
        g = lambda a, table: torch.gather(a, 1, table[sym])
        x = g(self.states[idx], self.sym_state).float() * 0.25
        policy = g(self.policies[idx], self.sym_move).float()
        legal = g(self.legal[idx], self.sym_move)
        parts = [x]
        if self.line_mode is not None:
            lines = self.lines[idx]  # [b, 3, 16]
            space = self.sym_space[sym][:, None, :].expand(-1, 3, -1)
            lines = torch.gather(lines, 2, space).float()
            parts.append(
                lines.amax(1) if self.line_mode == "any" else lines.flatten(1)
            )
        if self.legal_input:
            parts.append(legal.float())
        x = torch.cat(parts, 1) if len(parts) > 1 else x
        return x, self.values[idx], policy, legal


class Mlp(nn.Module):
    """The current architecture: depth x (Linear -> ReLU -> BatchNorm)."""

    def __init__(self, inputs, width, depth):
        super().__init__()
        layers, prev = [], inputs
        for _ in range(depth):
            layers += [
                nn.Linear(prev, width),
                nn.ReLU(),
                nn.BatchNorm1d(width, eps=1e-3, momentum=0.01),
            ]
            prev = width
        self.body = nn.Sequential(*layers)
        self.value = nn.Linear(width, 1)
        self.policy = nn.Linear(width, M)

    def forward(self, x):
        h = self.body(x)
        return torch.tanh(self.value(h)).squeeze(1), self.policy(h)


class ResMlp(nn.Module):
    """Residual blocks of two Linear layers, added to the stream.

    block "pre" (pre-activation): norm -> ReLU -> Linear, twice, then add;
    one norm -> ReLU before the heads.
    block "post" (the original ResNet order): Linear -> norm -> ReLU ->
    Linear -> norm, add, then ReLU; the input layer is Linear -> norm -> ReLU.
    """

    def __init__(self, inputs, width, depth, norm, block="pre"):
        super().__init__()
        make_norm = {
            "bn": lambda: nn.BatchNorm1d(width),
            "ln": lambda: nn.LayerNorm(width),
            "none": lambda: nn.Identity(),
        }[norm]
        self.post = block == "post"
        if self.post:
            self.inp = nn.Sequential(
                nn.Linear(inputs, width), make_norm(), nn.ReLU()
            )
            layers = lambda: [
                nn.Linear(width, width),
                make_norm(),
                nn.ReLU(),
                nn.Linear(width, width),
                make_norm(),
            ]
            self.out = nn.Identity()
        else:
            self.inp = nn.Linear(inputs, width)
            layers = lambda: [
                make_norm(),
                nn.ReLU(),
                nn.Linear(width, width),
                make_norm(),
                nn.ReLU(),
                nn.Linear(width, width),
            ]
            self.out = nn.Sequential(make_norm(), nn.ReLU())
        self.blocks = nn.ModuleList(
            nn.Sequential(*layers()) for _ in range(depth)
        )
        self.value = nn.Linear(width, 1)
        self.policy = nn.Linear(width, M)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
            if self.post:
                h = torch.relu(h)
        h = self.out(h)
        return torch.tanh(self.value(h)).squeeze(1), self.policy(h)


class Reference(nn.Module):
    """An existing model (model.py CorinthoNet), for --reference."""

    def __init__(self, prefix):
        super().__init__()
        import model as M_

        self.net, _ = M_.load(prefix)

    def forward(self, x):
        v, logits = self.net.heads(x)
        return v.squeeze(1), logits


def losses(value, logits, v_target, p_target, legal, mask):
    value_mse = F.mse_loss(value, v_target)
    if mask:
        logits = logits.masked_fill(~legal, -1e9)
    policy_ce = -(p_target * F.log_softmax(logits, 1)).sum(1).mean()
    return value_mse, policy_ce


@torch.no_grad()
def evaluate(net, data, rows, batch=16384):
    net.eval()
    sums = torch.zeros(3, device=rows.device, dtype=torch.float64)
    zero = torch.zeros(batch, dtype=torch.long, device=rows.device)
    for i in range(0, rows.numel(), batch):
        idx = rows[i : i + batch]
        x, v, p, legal = data.batch(idx, zero[: idx.numel()])
        value, logits = net(x)
        logits = logits.masked_fill(~legal, -1e9)  # the search's view
        logp = F.log_softmax(logits, 1)
        sums[0] += ((value - v) ** 2).sum()
        sums[1] += -(p * logp).sum()
        sums[2] += (logits.argmax(1) == p.argmax(1)).sum()
    n = rows.numel()
    value_mse, policy_ce, top1 = (sums / n).tolist()
    return {
        "value_mse": value_mse,
        "policy_ce": policy_ce,
        "top1": top1,
        "loss": VALUE_WEIGHT * value_mse + POLICY_WEIGHT * policy_ce,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=("mlp", "res"), default="mlp")
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--norm", choices=("bn", "ln", "none"), default="bn")
    ap.add_argument("--block", choices=("pre", "post"), default="pre")
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--no-sym", dest="sym", action="store_false")
    ap.add_argument("--lines", choices=("any", "type"))
    ap.add_argument("--legal-input", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-source", type=int, default=-1)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", help="evaluate this model prefix only")
    ap.add_argument("--tag", default="")
    ap.add_argument("--models", default="runs/sup/models")
    ap.add_argument("--out", default="runs/sup/results.jsonl")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    t0 = time.perf_counter()
    data = Data(a.data, dev, a.lines, a.legal_input)
    val_source = a.val_source % len(data.dirs)
    val_rows = torch.nonzero(data.source == val_source).squeeze(1)
    train_rows = torch.nonzero(data.source != val_source).squeeze(1)
    if a.train_fraction < 1.0:
        g = torch.Generator(device=dev).manual_seed(a.seed)
        keep = torch.randperm(train_rows.numel(), device=dev, generator=g)
        train_rows = train_rows[keep[: int(a.train_fraction * keep.numel())]]
    record = {k: v for k, v in vars(a).items() if k not in ("data", "out")}
    record.update(
        train_positions=train_rows.numel(), val_positions=val_rows.numel()
    )

    if a.reference:
        net = Reference(a.reference).to(dev)
        record.update(evaluate(net, data, val_rows))
        record["params"] = sum(p.numel() for p in net.parameters())
        print(json.dumps(record))
        with open(a.out, "a") as f:
            f.write(json.dumps(record) + "\n")
        return

    if a.model == "mlp":
        net = Mlp(data.inputs, a.width, a.depth)
    else:
        net = ResMlp(data.inputs, a.width, a.depth, a.norm, a.block)
    net = net.to(dev)
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(
        net.parameters(), lr=a.lr, weight_decay=a.wd, fused=True
    )
    steps_per_epoch = train_rows.numel() // a.batch
    total = steps_per_epoch * a.epochs
    warm = max(1, min(500, total // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm)
        * 0.5
        * (1 + math.cos(math.pi * min(s, total) / total)),
    )
    print(
        f"{a.model} w{a.width} d{a.depth} norm {a.norm} block {a.block} "
        f"lr {a.lr} wd {a.wd} seed {a.seed} mask {a.mask} "
        f"sym {a.sym} lines {a.lines} legal_input {a.legal_input}: "
        f"{params} params, {data.inputs} inputs, {train_rows.numel()} train, "
        f"{val_rows.numel()} val positions",
        flush=True,
    )
    history = []
    for epoch in range(a.epochs):
        net.train()
        perm = train_rows[torch.randperm(train_rows.numel(), device=dev)]
        run = torch.zeros(2, device=dev, dtype=torch.float64)
        for i in range(steps_per_epoch):
            idx = perm[i * a.batch : (i + 1) * a.batch]
            sym = (
                torch.randint(0, 8, (a.batch,), device=dev)
                if a.sym
                else torch.zeros(a.batch, dtype=torch.long, device=dev)
            )
            x, v, p, legal = data.batch(idx, sym)
            value, logits = net(x)
            vl, pl = losses(value, logits, v, p, legal, a.mask)
            loss = VALUE_WEIGHT * vl + POLICY_WEIGHT * pl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += torch.stack([vl.detach(), pl.detach()]).double()
        train_v, train_p = (run / steps_per_epoch).tolist()
        m = evaluate(net, data, val_rows)
        m.update(epoch=epoch, train_value_mse=train_v, train_policy_ce=train_p)
        history.append(m)
        print(
            f"epoch {epoch:2d}  train v {train_v:.4f} p {train_p:.4f}  "
            f"val v {m['value_mse']:.4f} p {m['policy_ce']:.4f} "
            f"top1 {m['top1']:.4f} loss {m['loss']:.4f}  "
            f"{time.perf_counter() - t0:.0f} s",
            flush=True,
        )
    best = min(history, key=lambda h: h["loss"])
    record.update(
        params=params,
        inputs=data.inputs,
        seconds=time.perf_counter() - t0,
        final=history[-1],
        best=best,
    )
    with open(a.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    os.makedirs(a.models, exist_ok=True)
    name = a.tag or f"{a.model}_w{a.width}_d{a.depth}"
    torch.save(
        {"model": net.state_dict(), "args": vars(a)},
        f"{a.models}/{name}.pt",
    )


if __name__ == "__main__":
    main()
