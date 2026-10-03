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

DEV = "cuda"
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


class Small(nn.Module):
    """70 -> hidden -> 32 -> 1, clipped ReLU (NNUE-style, quantizable)"""

    def __init__(self, hidden):
        super().__init__()
        self.l1 = nn.Linear(70, hidden)
        self.l2 = nn.Linear(hidden, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, x):
        h = torch.clamp(self.l1(x), 0, 1)
        h = torch.clamp(self.l2(h), 0, 1)
        return torch.tanh(self.l3(h)).squeeze(1)


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
    net = Small(a.hidden).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05
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
                net.l2.weight.clamp_(-a.clip_w2, a.clip_w2)
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
    params = sum(p.numel() for p in net.parameters())
    result = {
        "tag": a.tag,
        "hidden": a.hidden,
        "min_p": a.min_p,
        "clip_w2": a.clip_w2,
        "rows": int(len(target)),
        "steps": a.steps,
        "best_step": best_step,
        "params": params,
        "train_s": round(time.time() - t0, 1),
        **report(pred, t),
    }
    os.makedirs("runs/nnue/models", exist_ok=True)
    torch.save(
        {"hidden": a.hidden, "model": net.state_dict()},
        f"runs/nnue/models/{a.tag}.pt",
    )
    with open("runs/nnue/results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


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
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--tag", default="small")
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
    }[a.cmd](a)


if __name__ == "__main__":
    main()
