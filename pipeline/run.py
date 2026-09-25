"""The generation loop of the GPU pipeline (worklog entry 29).

    python run.py --name overnight --generations 50

Each generation g, mirroring corintho_ai/python/wrapper.py and main.pyx:
  1. selfplay  the best model plays --games games (corintho_play train, GPU)
  2. fit       gen g-1's model trains on those samples (fit.py); as in the
               Cython pipeline, training continues from the latest
               generation, not from the best
  3. test      gen g plays the best model, --test-games games, alternating
               colours (corintho_play test); promoted when the Wilson lower
               bound on its decisive-game win rate exceeds 0.5
               (corintho_ai/python/promotion.py, entry 12)
  4. update    best generation, Elo-style rating, and the learning rate:
               halved after --patience consecutive generations that neither
               promoted nor improved the best validation loss
               (wrapper.py write_learning_rate)

Resumable: every step leaves gen_g/<step>.done, and state.json holds the run's
state. Rerunning the same command continues where it stopped. A run starts
from --init (default models/gen_93, the last cloud generation).

Samples are ~3.8 GB per 25k-game generation, so each generation's samples are
deleted once no later fit can use them (--old-gens, default 0).

--old-gens defaults to 0 because that is what the Cython pipeline actually
did: its get_samples() concatenated older generations with np.concatenate but
discarded the result, so gen_93's metadata ("num_old_gens": 2) never took
effect (entry 29).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "corintho_ai", "python"))
from promotion import describe, should_promote  # noqa: E402

PLAY = os.path.join(HERE, "build", "corintho_play")
PYTHON = sys.executable


def log(run_dir, msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(os.path.join(run_dir, "progress.log"), "a") as f:
        f.write(line + "\n")


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic: a crash never leaves half a state file


def sh(cmd, logfile):
    with open(logfile, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def pkg_temp():
    for z in sorted(os.listdir("/sys/class/thermal")):
        try:
            with open(f"/sys/class/thermal/{z}/type") as f:
                if f.read().strip() == "x86_pkg_temp":
                    with open(f"/sys/class/thermal/{z}/temp") as t:
                        return int(t.read()) // 1000
        except OSError:
            pass
    return None


def model_prefix(run_dir, g):
    return os.path.join(run_dir, f"gen_{g}", "model")


def step(run_dir, g, name, fn):
    marker = os.path.join(run_dir, f"gen_{g}", f"{name}.done")
    if os.path.exists(marker):
        return
    t = time.perf_counter()
    fn()
    seconds = time.perf_counter() - t
    times_path = os.path.join(run_dir, f"gen_{g}", "timing.json")
    times = load_json(times_path, {})
    times[name] = seconds
    save_json(times_path, times)
    open(marker, "w").close()
    log(run_dir, f"gen {g} {name} done in {seconds:.1f} s (package {pkg_temp()} C)")


def generation(run_dir, cfg, state, g):
    gen_dir = os.path.join(run_dir, f"gen_{g}")
    os.makedirs(gen_dir, exist_ok=True)
    samples = os.path.join(gen_dir, "samples")
    test_dir = os.path.join(gen_dir, "test")
    best = state["best_gen"]
    play_log = os.path.join(gen_dir, "play.log")
    common = ["--searches", str(cfg["searches"]), "--spe", str(cfg["spe"]),
              "--c-puct", str(cfg["c_puct"]), "--epsilon", str(cfg["epsilon"]),
              "--threads", str(cfg["threads"])]

    def selfplay():
        os.makedirs(samples, exist_ok=True)
        sh([PLAY, "train", "--model", model_prefix(run_dir, best) + ".onnx",
            "--games", str(cfg["games"]), "--in-flight", str(cfg["in_flight"]),
            "--groups", str(cfg.get("groups", 1)),
            "--seed", str(cfg["seed"] * 100003 + 2 * g), "--out", samples,
            "--logged", str(cfg["logged"])] + common, play_log)

    def fit():
        dirs = [samples] + [
            os.path.join(run_dir, f"gen_{h}", "samples")
            for h in range(max(1, g - cfg["old_gens"]), g)]
        sh([PYTHON, os.path.join(HERE, "fit.py"),
            "--init", model_prefix(run_dir, g - 1),
            "--samples", *[d for d in dirs if os.path.isdir(d)],
            "--out", model_prefix(run_dir, g),
            "--lr", repr(state["lr"]), "--epochs", str(cfg["epochs"]),
            "--batch", str(cfg["batch"]), "--patience", str(cfg["fit_patience"]),
            "--anneal", str(cfg["anneal"]), "--seed", str(g)],
           os.path.join(gen_dir, "fit.log"))

    def test():
        os.makedirs(test_dir, exist_ok=True)
        sh([PLAY, "test", "--new", model_prefix(run_dir, g) + ".onnx",
            "--best", model_prefix(run_dir, best) + ".onnx",
            "--games", str(cfg["test_games"]),
            "--seed", str(cfg["seed"] * 100003 + 2 * g + 1), "--out", test_dir]
           + common, play_log)

    step(run_dir, g, "selfplay", selfplay)
    step(run_dir, g, "fit", fit)
    step(run_dir, g, "test", test)

    if state["done_gen"] >= g:
        return
    # Update: promotion, rating, learning rate. Runs once per generation;
    # state.json is written atomically at the end.
    res = load_json(os.path.join(test_dir, "test.json"))
    promote, gate = should_promote(res["wins"], res["draws"], res["games"],
                                   cfg["confidence"])
    score = res["score"]
    best_rating = state["ratings"][str(best)]
    # main.pyx update_rating, unchanged
    rating = (best_rating - 400 * math.log10(1 / score - 1) if 0 < score < 1
              else best_rating + (400 if score >= 1 else -400))
    state["ratings"][str(g)] = rating
    fit_info = load_json(model_prefix(run_dir, g) + "_fit.json")
    state["val_losses"].append(fit_info["best_val_loss"])
    lr_before = state["lr"]
    if promote:
        state["best_gen"] = g
        state["fails"] = 0
    elif fit_info["best_val_loss"] <= min(state["val_losses"]):
        state["fails"] = 0  # loss improved: not a failure
    else:
        state["fails"] += 1
        if state["fails"] >= cfg["patience"]:
            state["lr"] *= cfg["anneal"]
            state["fails"] = 0
    state["done_gen"] = g
    with open(os.path.join(test_dir, "score.txt"), "w") as f:
        f.write(describe(gate) + "\n")
    times = load_json(os.path.join(gen_dir, "timing.json"), {})
    sp = load_json(os.path.join(samples, "selfplay.json"), {})
    row = {"gen": g, "best_before": best, "promoted": promote,
           "wins": res["wins"], "draws": res["draws"], "losses": res["losses"],
           "ci_low": round(gate["ci_low"], 4), "rating": round(rating, 1),
           "val_loss": round(fit_info["best_val_loss"], 5),
           "lr": lr_before, "turns_per_game": round(sp.get("turns", 0) / cfg["games"], 2),
           "first_player_score": round(sp.get("first_player_score", 0), 4),
           "selfplay_s": round(times.get("selfplay", 0), 1),
           "fit_s": round(times.get("fit", 0), 1),
           "test_s": round(times.get("test", 0), 1)}
    tsv = os.path.join(run_dir, "generations.tsv")
    new = not os.path.exists(tsv)
    with open(tsv, "a") as f:
        if new:
            f.write("\t".join(row) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")
    save_json(os.path.join(run_dir, "state.json"), state)
    log(run_dir, f"gen {g}: {describe(gate)} -> "
                 f"{'PROMOTED' if promote else 'kept gen ' + str(best)}; "
                 f"rating {rating:.0f}, val loss {fit_info['best_val_loss']:.5f}, "
                 f"lr {state['lr']:.3g}")

    # Samples no later fit will read
    for h in range(1, g - cfg["old_gens"] + 1):
        d = os.path.join(run_dir, f"gen_{h}", "samples")
        for name in ("states.npy", "values.npy", "policies.npy"):
            p = os.path.join(d, name)
            if not cfg["keep_samples"] and os.path.exists(p):
                os.remove(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--generations", type=int, default=1,
                    help="generations to run in this invocation")
    ap.add_argument("--init", default=os.path.join(HERE, "models", "gen_93"))
    ap.add_argument("--init-rating", type=float, default=5537.14343997563,
                    help="rating of --init (gen_93's)")
    # Defaults: gen_93's metadata.txt
    ap.add_argument("--games", type=int, default=25000)
    # Two groups of 2,000 games: the engine searches one while the GPU
    # evaluates the other; 25k games in ~336 s instead of ~407 s with one
    # group of 1,000, at ~3.7 GB peak (entry 29)
    ap.add_argument("--in-flight", type=int, default=2000)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--test-games", type=int, default=1600)
    ap.add_argument("--searches", type=int, default=1600)
    ap.add_argument("--spe", type=int, default=16)
    ap.add_argument("--c-puct", type=float, default=3.0)
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--fit-patience", type=int, default=2)
    ap.add_argument("--anneal", type=float, default=0.5)
    ap.add_argument("--old-gens", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--logged", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep-samples", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.join(HERE, "runs", args.name)
    cfg_path = os.path.join(run_dir, "config.json")
    state_path = os.path.join(run_dir, "state.json")
    if os.path.exists(cfg_path):
        cfg = load_json(cfg_path)  # an existing run keeps its settings
    else:
        os.makedirs(os.path.join(run_dir, "gen_0"), exist_ok=True)
        cfg = {k: v for k, v in vars(args).items() if k not in ("name", "generations")}
        for ext in (".pt", ".onnx", ".mlp"):
            shutil.copy(args.init + ext, model_prefix(run_dir, 0) + ext)
        save_json(cfg_path, cfg)
        save_json(state_path, {"best_gen": 0, "done_gen": 0, "lr": args.lr,
                               "fails": 0, "val_losses": [],
                               "ratings": {"0": args.init_rating}})
        log(run_dir, f"new run from {args.init}: {json.dumps(cfg)}")
    state = load_json(state_path)
    start = state["done_gen"] + 1
    for g in range(start, start + args.generations):
        generation(run_dir, cfg, state, g)
    log(run_dir, f"stopped after gen {state['done_gen']}; best gen {state['best_gen']}")


if __name__ == "__main__":
    main()
