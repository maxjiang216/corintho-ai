"""Fit the network on self-play samples: the Keras fit from main.pyx, in PyTorch.

    python fit.py --init models/gen_93 --samples runs/x/gen_1/samples \
                  --out runs/x/gen_1/model --lr 5e-6

What it reproduces from corintho_ai/python/main.pyx train_neural_network:
  loss       MSE(value) + 0.25 * cross-entropy(policy)   (wrapper.py weights)
  optimizer  Adam, Keras defaults (beta 0.9/0.999, epsilon 1e-7); its state
             carries over from --init when present (load_model restored it)
  data       validation = the LAST 30% of rows, as Keras validation_split
             takes them before shuffling; the rest shuffled every epoch
  schedule   ReduceLROnPlateau on validation loss (factor 0.5, patience 2)
  result     the epoch with the lowest validation loss (ModelCheckpoint
             save_best_only)

The whole generation sits on the GPU: states as bytes (they are multiples of
0.25), values and policies as float32, ~2.6 GB for 25k games.

Writes <out>.pt/.onnx/.mlp (model.save), <out>_fit.json and <out>_loss.csv.
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import model as M

VALUE_WEIGHT, POLICY_WEIGHT = 1.0, 0.25


def load_samples(dirs, device):
    states, values, policies = [], [], []
    for d in dirs:
        s = np.load(f"{d}/states.npy", mmap_mode="r")
        q = np.rint(np.asarray(s) * 4.0)
        assert np.array_equal(q / 4.0, s), "states are not multiples of 0.25"
        states.append(torch.from_numpy(q.astype(np.uint8)))
        values.append(torch.from_numpy(np.load(f"{d}/values.npy").reshape(-1)))
        policies.append(torch.from_numpy(np.load(f"{d}/policies.npy")))
    return (torch.cat(states).to(device), torch.cat(values).to(device),
            torch.cat(policies).to(device))


def losses(net, states_u8, values, policies):
    x = states_u8.float() * 0.25
    v, logits = net.heads(x)
    value_loss = F.mse_loss(v.squeeze(1), values)
    policy_loss = -(policies * F.log_softmax(logits, dim=1)).sum(1).mean()
    return value_loss, policy_loss


class GraphStep:
    """One training step recorded as a CUDA graph and replayed.

    A step of this small network is ~150 GPU operations taking ~6 ms, almost
    all of it Python and kernel-launch overhead: the time is the same at batch
    2048 and 8192 (entry 29). Replaying a recorded graph launches them all at
    once. Capture needs a few real warm-up steps; the weights, BatchNorm
    statistics and Adam state are restored afterwards, so training starts from
    exactly the loaded state.
    """

    def __init__(self, net, opt, train, batch):
        self.train, self.batch = train, batch
        dev = train[0].device
        self.idx = torch.zeros(batch, dtype=torch.long, device=dev)
        params = list(net.parameters()) + list(net.buffers())
        saved = [t.detach().clone() for t in params]
        had_state = len(opt.state) > 0
        saved_opt = {id(p): {k: v.clone() for k, v in st.items() if torch.is_tensor(v)}
                     for p, st in opt.state.items()}

        def step():
            vl, pl = losses(net, *(t[self.idx] for t in train))
            loss = VALUE_WEIGHT * vl + POLICY_WEIGHT * pl
            loss.backward()
            opt.step()
            return loss

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            self.loss = step()
        # Undo the warm-up (and the capture's own step)
        with torch.no_grad():
            for t, v in zip(params, saved):
                t.copy_(v)
            for p, st in opt.state.items():
                for k, v in st.items():
                    if not torch.is_tensor(v):
                        continue
                    if had_state and k in saved_opt.get(id(p), {}):
                        v.copy_(saved_opt[id(p)][k])
                    else:
                        v.zero_()  # fresh Adam: moments and step count 0

    def __call__(self, idx):
        self.idx.copy_(idx)
        self.graph.replay()
        return self.loss


def evaluate(net, data, batch):
    net.eval()
    totals = np.zeros(3)
    n = data[0].shape[0]
    with torch.no_grad():
        for i in range(0, n, batch):
            vl, pl = losses(net, *(t[i:i + batch] for t in data))
            k = min(batch, n - i)
            totals += k * np.array([VALUE_WEIGHT * vl.item() + POLICY_WEIGHT * pl.item(),
                                    vl.item(), pl.item()])
    return totals / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="model prefix to start from (.pt)")
    ap.add_argument("--samples", required=True, nargs="+", help="sample directories")
    ap.add_argument("--out", required=True, help="output model prefix")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--val-split", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--anneal", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-graph", dest="graph", action="store_false",
                    help="plain eager steps instead of a CUDA graph (checks)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32, as entry 20 requires
    torch.backends.cudnn.allow_tf32 = False

    net, state = M.load(args.init)
    net = net.to(device)
    # fused: one kernel for the whole update instead of several per tensor
    # capturable: the update reads the step count and learning rate from GPU
    # tensors, so it can be recorded in a CUDA graph
    opt = torch.optim.Adam(net.parameters(), lr=torch.tensor(args.lr, device=device),
                           eps=1e-7, fused=True, capturable=True)
    if "optimizer" in state:
        opt.load_state_dict(state["optimizer"])
    for group in opt.param_groups:  # the run's schedule sets the rate
        if torch.is_tensor(group["lr"]):
            group["lr"].fill_(args.lr)
        else:  # state saved before the optimizer was capturable
            group["lr"] = torch.tensor(args.lr, device=device)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=args.anneal, patience=args.patience)

    data = load_samples(args.samples, device)
    n = data[0].shape[0]
    n_val = int(n * args.val_split)
    train = tuple(t[:n - n_val] for t in data)
    val = tuple(t[n - n_val:] for t in data)
    t_loaded = time.perf_counter()
    net.train()
    graph = GraphStep(net, opt, train, args.batch) if args.graph else None

    rows, best = [], None
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(train[0].shape[0], device=device)
        # The running total stays on the GPU: calling .item() every step made
        # the CPU wait for the GPU, which then idled while the next step's
        # ~200 kernels were launched (entry 29)
        total = torch.zeros((), device=device, dtype=torch.float64)
        count = 0
        for i in range(0, perm.numel(), args.batch):
            idx = perm[i:i + args.batch]
            if graph is not None and idx.numel() == args.batch:
                loss = graph(idx)  # zeroing grads is part of the graph's pool
            else:
                if graph is not None:
                    # Gradients live in the graph's pool; the eager step
                    # must accumulate into zeroed tensors, not new ones
                    for p in net.parameters():
                        p.grad.zero_()
                else:
                    opt.zero_grad(set_to_none=True)
                vl, pl = losses(net, *(t[idx] for t in train))
                loss = VALUE_WEIGHT * vl + POLICY_WEIGHT * pl
                loss.backward()
                opt.step()
            total += loss.detach().double() * idx.numel()
            count += idx.numel()
        total = total.item()
        val_loss, val_value, val_policy = evaluate(net, val, 8192)
        lr = float(opt.param_groups[0]["lr"])
        sched.step(val_loss)
        rows.append({"epoch": epoch, "loss": total / count, "val_loss": val_loss,
                     "val_value_loss": val_value, "val_policy_loss": val_policy,
                     "lr": lr})
        if best is None or val_loss < best["val_loss"]:
            best = dict(rows[-1])
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            best_opt = copy.deepcopy(opt.state_dict())
        print(f"epoch {epoch}: loss {total / count:.5f}  val {val_loss:.5f} "
              f"(value {val_value:.5f}, policy {val_policy:.5f})  lr {lr:.3g}",
              flush=True)
    t_fit = time.perf_counter()

    net.load_state_dict(best_state)
    net = net.cpu().eval()
    M.save(net, args.out, optimizer=None, extra={"optimizer": best_opt})
    with open(f"{args.out}_loss.csv", "w") as f:
        f.write("epoch\tloss\tval_loss\tval_value_loss\tval_policy_loss\tlr\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in
                              ("epoch", "loss", "val_loss", "val_value_loss",
                               "val_policy_loss", "lr")) + "\n")
    info = {"samples": n, "train_rows": n - n_val, "val_rows": n_val,
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
            "load_seconds": t_loaded - t0, "fit_seconds": t_fit - t_loaded,
            "total_seconds": time.perf_counter() - t0, "init": args.init,
            "sample_dirs": args.samples}
    with open(f"{args.out}_fit.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
