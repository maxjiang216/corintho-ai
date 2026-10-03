"""Small, fast value networks distilled from an AZ network, for an
alpha-beta engine (worklog 2026-09-25-nn-architectures, entry 21).

    python arch/nnue_distill.py targets OUT.npz TEACHER.pt DATA.npz [...]
        Every position of the DATA files with its target: the exact value
        stored in the data at P <= 27 (runs made with --relabel-p 27), else
        the teacher network's value.
    python arch/nnue_distill.py testset OUT.npz TEACHER.pt DATA.npz \
            EXACT.tsv EXACT_IDX.npy
        A held-out set: the teacher's value of every position, the stored
        game outcomes, and exact values (solver_bench results) for the rows
        EXACT_IDX.
    python arch/nnue_distill.py train TARGETS.npz TEST.npz --hidden 256
            [--rows N] [--min-p P] [--steps S] [--tag T]
        Trains 70 -> hidden -> 32 -> 1 (clipped ReLU, tanh output) on the
        first N rows (a fixed shuffle) with horizon >= P, a random board
        symmetry per sample,
        MSE to the targets. Reports error by horizon P against the teacher,
        the exact values and the outcomes; appends to
        runs/nnue/results.jsonl.

    python arch/nnue_distill.py train ... --tie --mono-m 64 --mono-m2 16
        Weights tied across the board symmetries; monotone in the reserves
        (Small, entry 24).
    python arch/nnue_distill.py export MODEL.pt OUT.bin
        small_net.h's file (ab_match --small).

Inputs are the 70 network inputs (64 board bits, 6 reserve counts / 4).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import export  # noqa: E402
import model as M  # noqa: E402
from features import symmetries  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SOLVE_P = 27


def horizon(states):
    """P = 2 x reserves + occupied spaces, from uint8 states (inputs x 4)"""
    occupied = (states[:, :64].reshape(-1, 16, 4)[:, :, :3].sum(2) > 0).sum(1)
    return 2 * states[:, 64:70].astype(np.int64).sum(1) + occupied


@torch.no_grad()
def teacher_values(path, states, batch=65536):
    net = M.CompactIo(export.Probabilities(export.load(path))).to(DEV).eval()
    out = np.empty(len(states), np.float32)
    for i in range(0, len(states), batch):
        x = torch.from_numpy(states[i : i + batch]).to(DEV)
        out[i : i + batch] = net(x)[0].squeeze(1).float().cpu().numpy()
    return out


def targets(a):
    states, values = [], []
    for p in a.data:
        z = np.load(p)
        states.append(z["states"])
        values.append(z["values"])
    states, values = np.concatenate(states), np.concatenate(values)
    teacher = teacher_values(a.teacher, states)
    p = horizon(states)
    target = np.where(p <= SOLVE_P, values, teacher).astype(np.float32)
    np.savez(a.out, states=states, target=target, horizon=p)
    print(
        f"{a.out}: {len(states):,} positions, {np.mean(p <= SOLVE_P):.1%} "
        f"with exact targets"
    )


def testset(a):
    z = np.load(a.data)
    states = z["states"]
    idx = np.load(a.exact_idx)
    with open(a.exact) as f:
        rows = {
            int(r["index"]): int(r["result"])
            for r in csv.DictReader(f, delimiter="\t")
        }
    exact = np.array([rows[i] for i in range(len(idx))], np.float32)
    np.savez(
        a.out,
        states=states,
        teacher=teacher_values(a.teacher, states),
        outcome=z["values"],
        horizon=horizon(states),
        exact_idx=idx,
        exact=exact,
    )
    print(f"{a.out}: {len(states):,} positions, {len(idx)} exact")


def symmetry_tables():
    """The 8 state symmetries as gather indices (x_k = x[sym[k]]), their
    inverses, and rel[l][k], the symmetry taking l to k (l^-1 k)"""
    sym = symmetries()[0]
    inv = np.argsort(sym, axis=1)
    key = {tuple(r): k for k, r in enumerate(sym)}
    # x[sym[m]][sym[k]] = x[sym[m][sym[k]]]: the product of m and k
    prod = np.array(
        [[key[tuple(sym[m][sym[k]])] for k in range(8)] for m in range(8)]
    )
    ident = key[tuple(range(sym.shape[1]))]
    inverse = [int(np.where(prod[m] == ident)[0][0]) for m in range(8)]
    rel = np.array([[prod[inverse[l]][k] for k in range(8)] for l in range(8)])
    return sym, inv, rel


class Small(nn.Module):
    """70 -> hidden -> 32 -> 1, clipped ReLU (NNUE-style, quantizable).

    tie: weights tied across the 8 board symmetries (entry 24). Units come
    in orbits of 8, one per symmetry: layer-1 unit (g, k) sees the board
    under symmetry k with group g's weights, layer 2 is equivariant
    (W2[(q, l), (g, k)] depends only on l^-1 k) and the output weight is
    shared by an orbit, so the value is exactly invariant. Same dense
    network at inference.

    mono_m, mono_m2: the last mono_m layer-1 units and the last mono_m2
    layer-2 units are reserve-aware; the others never see the reserves.
    Reserve-aware units take the side to move's reserves with weights >= 0
    and the opponent's with weights <= 0, and every weight downstream of
    them is >= 0, so the value never decreases with more of one's own
    pieces or fewer of the opponent's (entry 24, option B). Kept by
    projection after each step (project()).
    """

    def __init__(self, hidden, tie=False, mono_m=0, mono_m2=0):
        super().__init__()
        r = 8 if tie else 1
        assert hidden % r == 0 and 32 % r == 0
        assert mono_m % r == 0 and mono_m2 % r == 0
        assert mono_m <= hidden and mono_m2 <= 32
        assert (mono_m > 0) == (mono_m2 > 0), "mono needs both layers"
        self.hidden, self.tie, self.r = hidden, tie, r
        g, q = hidden // r, 32 // r
        self.mono = mono_m > 0
        self.m1, self.m2 = g - mono_m // r, q - mono_m2 // r  # first M group
        u = lambda shape, fan: nn.Parameter(  # noqa: E731
            torch.empty(shape).uniform_(-(fan**-0.5), fan**-0.5)
        )
        self.w1, self.b1 = u((g, 70), 70), u((g,), 70)
        self.w2, self.b2 = u((q, g, r), hidden), u((q,), hidden)
        self.w3, self.b3 = u((q,), 32), u((1,), 32)
        if tie:
            sym, inv, rel = symmetry_tables()
            self.register_buffer("inv", torch.from_numpy(inv))
            self.register_buffer("rel", torch.from_numpy(rel))
        self.project()

    @torch.no_grad()
    def project(self):
        if not self.mono:
            return
        m1, m2 = self.m1, self.m2
        self.w1[:m1, 64:70] = 0  # board-only units
        self.w1[m1:, 64:67].clamp_(min=0)  # side to move's reserves
        self.w1[m1:, 67:70].clamp_(max=0)  # the opponent's
        self.w2[:m2, m1:] = 0  # board-only layer-2 units
        self.w2[m2:, m1:].clamp_(min=0)
        self.w3[m2:].clamp_(min=0)

    def dense(self):
        """(W1 [h, 70], b1, W2 [32, h], b2, W3 [1, 32], b3)"""
        r = self.r
        if not self.tie:
            return (
                self.w1,
                self.b1,
                self.w2[:, :, 0],
                self.b2,
                self.w3[None],
                self.b3,
            )
        g, q = self.w1.shape[0], self.w2.shape[0]
        # Unit (g, k) = g * 8 + k: W1[(g, k), sym[k][j]] = w1[g, j]
        w1 = self.w1[:, self.inv].reshape(g * r, 70)
        w2 = self.w2[:, :, self.rel]  # [q, g, l, k]
        w2 = w2.permute(0, 2, 1, 3).reshape(q * r, g * r)
        rep = lambda v: v.repeat_interleave(r)  # noqa: E731
        return w1, rep(self.b1), w2, rep(self.b2), rep(self.w3)[None], self.b3

    def forward(self, x):
        w1, b1, w2, b2, w3, b3 = self.dense()
        h = torch.clamp(nn.functional.linear(x, w1, b1), 0, 1)
        h = torch.clamp(nn.functional.linear(h, w2, b2), 0, 1)
        return torch.tanh(nn.functional.linear(h, w3, b3)).squeeze(1)

    @torch.no_grad()
    def dense_state(self):
        """state_dict of the plain network (l1, l2, l3), for any variant"""
        w1, b1, w2, b2, w3, b3 = (t.detach().clone() for t in self.dense())
        return {
            "l1.weight": w1,
            "l1.bias": b1,
            "l2.weight": w2,
            "l2.bias": b2,
            "l3.weight": w3,
            "l3.bias": b3,
        }


@torch.no_grad()
def predict(net, states, batch=65536):
    net.eval()
    out = []
    for i in range(0, len(states), batch):
        x = states[i : i + batch].float() * 0.25
        out.append(net(x))
    return torch.cat(out).cpu().numpy()


BUCKETS = ((10, 22), (23, 27), (28, 35), (36, 48))


def report(pred, t):
    """Error by horizon P: against the teacher (P >= 28), the exact values
    (sample at P <= 27; MSE / sign right on decisive), and the outcomes"""
    out = {}
    p = t["horizon"]
    for lo, hi in BUCKETS:
        m = (p >= lo) & (p <= hi)
        key = f"P{lo}-{hi}"
        out[key + "_teacher_mse"] = float(
            np.mean((pred[m] - t["teacher"][m]) ** 2)
        )
        out[key + "_outcome_mse"] = float(
            np.mean((pred[m] - t["outcome"][m]) ** 2)
        )
    idx, ex = t["exact_idx"], t["exact"]
    pe = p[idx]
    for lo, hi in BUCKETS[:2]:
        m = (pe >= lo) & (pe <= hi)
        e, v = ex[m], pred[idx][m]
        d = e != 0
        out[f"P{lo}-{hi}_exact_mse"] = float(np.mean((v - e) ** 2))
        out[f"P{lo}-{hi}_exact_sign"] = float(np.mean(np.sign(v[d]) == e[d]))
    return out


def train(a):
    z = np.load(a.targets)
    t = dict(np.load(a.test))
    n = len(z["target"])
    order = np.random.default_rng(0).permutation(n)
    # Positions an alpha-beta engine solves instead of evaluating are left out
    order = order[z["horizon"][order] >= a.min_p][: a.rows or n]
    states = torch.from_numpy(z["states"][order]).to(DEV)
    target = torch.from_numpy(z["target"][order]).to(DEV)
    test_states = torch.from_numpy(t["states"]).to(DEV)
    sym = torch.from_numpy(symmetries()[0]).to(DEV)
    torch.manual_seed(a.seed)
    net = Small(a.hidden, a.tie, a.mono_m, a.mono_m2).to(DEV)
    # A tied layer-2 or output weight moves its 8 copies together, so the
    # output moves ~8x as far per step: at the full rate layer 2 sometimes
    # saturated for good (1 seed in 6 on a synthetic target; 0 at 1/8)
    lr2 = a.lr / net.r
    opt = torch.optim.AdamW(
        [
            {"params": [net.w1, net.b1], "lr": a.lr},
            {"params": [net.w2, net.b2, net.w3, net.b3], "lr": lr2},
        ],
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr, lr2], total_steps=a.steps, pct_start=0.05
    )
    g = torch.Generator(device=DEV).manual_seed(a.seed)
    t0 = time.time()
    best, best_step, best_state = 1e9, 0, None
    for step in range(1, a.steps + 1):
        net.train()
        idx = torch.randint(
            0, len(target), (a.batch,), device=DEV, generator=g
        )
        k = torch.randint(0, 8, (a.batch,), device=DEV, generator=g)
        x = torch.gather(states[idx], 1, sym[k]).float() * 0.25
        loss = ((net(x) - target[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if a.clip_w2 > 0:  # int8-friendly layer 2 (small_net.h: x 64)
            with torch.no_grad():
                net.w2.clamp_(-a.clip_w2, a.clip_w2)
        net.project()
        if step % a.eval_every == 0 or step == a.steps:
            pred = predict(net, test_states)
            m = t["horizon"] >= 28
            score = float(np.mean((pred[m] - t["teacher"][m]) ** 2))
            if score < best:
                best, best_step = score, step
                best_state = {
                    k: v.clone() for k, v in net.state_dict().items()
                }
    net.load_state_dict(best_state)
    pred = predict(net, test_states)
    params = sum(p.numel() for p in net.parameters())  # free, incl. zeros
    result = {
        "tag": a.tag,
        "hidden": a.hidden,
        "min_p": a.min_p,
        "clip_w2": a.clip_w2,
        "tie": a.tie,
        "mono_m": a.mono_m,
        "mono_m2": a.mono_m2,
        "seed": a.seed,
        "rows": int(len(target)),
        "steps": a.steps,
        "best_step": best_step,
        "params": params,
        "train_s": round(time.time() - t0, 1),
        **report(pred, t),
    }
    os.makedirs("runs/nnue/models", exist_ok=True)
    torch.save(
        {"hidden": a.hidden, "model": net.dense_state()},
        f"runs/nnue/models/{a.tag}.pt",
    )
    with open("runs/nnue/results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


def export_bin(a):
    """A trained model (.pt) as small_net.h's file: int32 hidden, W1
    input-major [70][h], b1, W2 [32][h], b2, W3 [32], b3, float32"""
    ck = torch.load(a.model, map_location="cpu")
    w = {k: v.float().numpy() for k, v in ck["model"].items()}
    h = np.array([ck["hidden"]], np.int32)
    with open(a.out, "wb") as f:
        h.tofile(f)
        for x in (
            w["l1.weight"].T,
            w["l1.bias"],
            w["l2.weight"],
            w["l2.bias"],
            w["l3.weight"][0],
            w["l3.bias"],
        ):
            np.ascontiguousarray(x, np.float32).tofile(f)
    print(f"{a.out}: hidden {ck['hidden']}")


def reference(a):
    """The teacher-style AZ network itself on the test set, for comparison"""
    t = dict(np.load(a.test))
    pred = teacher_values(a.model, t["states"])
    result = {"tag": a.tag, **report(pred, t)}
    os.makedirs("runs/nnue", exist_ok=True)
    with open("runs/nnue/results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("targets")
    p.add_argument("out")
    p.add_argument("teacher")
    p.add_argument("data", nargs="+")
    p = sub.add_parser("testset")
    p.add_argument("out")
    p.add_argument("teacher")
    p.add_argument("data")
    p.add_argument("exact")
    p.add_argument("exact_idx")
    p = sub.add_parser("train")
    p.add_argument("targets")
    p.add_argument("test")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--rows", type=int, default=0)
    p.add_argument("--min-p", type=int, default=0)
    p.add_argument("--clip-w2", type=float, default=0.0)
    p.add_argument("--tie", action="store_true")
    p.add_argument("--mono-m", type=int, default=0)
    p.add_argument("--mono-m2", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--tag", default="small")
    p = sub.add_parser("export")
    p.add_argument("model")
    p.add_argument("out")
    p = sub.add_parser("reference")
    p.add_argument("test")
    p.add_argument("model")
    p.add_argument("--tag", default="reference")
    a = ap.parse_args()
    {
        "targets": targets,
        "testset": testset,
        "train": train,
        "reference": reference,
        "export": export_bin,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
