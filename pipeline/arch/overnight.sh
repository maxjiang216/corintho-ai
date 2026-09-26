#!/usr/bin/env bash
# Overnight loop for the residual 512x4 network (worklog
# 2026-09-25-nn-architectures, entry 04), built only from tools already
# checked: corintho_play, arch/dataset.py, arch/sup.py, arch/export.py.
#
#   arch/overnight.sh            (from pipeline/; resumes where it stopped)
#   arch/overnight_supervise.sh  (reruns this after a failure, and logs it)
#
# Each iteration i = 1..ITERS:
#   1. self-play: DIRS x GAMES games with the current network (fp16 TensorRT)
#   2. dataset: the newest WINDOW sample directories (runs/full-2 gens 1-4
#      are the oldest, until pushed out), one row per position, checked
#   3. train a fresh network (arch/sup.py recipe) on all but the newest
#      directory, validate on the newest
#   4. export it and play matches, for the record only: against the previous
#      iteration's network (the Elo chain), iteration 0 (the first
#      supervised network) and gen 1 (the old pipeline's best). The new
#      network always becomes the self-play network (no gate).
# Every step leaves its output as a marker, so a rerun skips finished steps.
# Results: runs/$NAME/progress.log (events) and summary.tsv (one row each).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${NAME:-night-1}
ITERS=${ITERS:-8}
DIRS=${DIRS:-10}             # sample directories per iteration
GAMES=${GAMES:-25000}        # games per directory
WINDOW=${WINDOW:-14}         # newest sample directories in each dataset
EPOCHS=${EPOCHS:-40}
MATCH=${MATCH:-1600}
THREADS=${THREADS:-14}       # physical cores: as fast as 20 here, cooler
WIDTH=${WIDTH:-512}
DEPTH=${DEPTH:-4}
INIT=${INIT:-runs/sup/models/s4_512x4.pt}
ANCHOR=${ANCHOR:-$PWD/runs/full-2/gen_1/model.onnx}

R=$PWD/runs/$NAME
PLAY=$PWD/build/corintho_play
PY=$PWD/.venv/bin/python
mkdir -p "$R"
log() { echo "$(date '+%F %T') $*" | tee -a "$R/progress.log"; }
[ -f "$R/config.txt" ] || {
  for v in NAME ITERS DIRS GAMES WINDOW EPOCHS MATCH THREADS WIDTH DEPTH INIT ANCHOR; do
    echo "$v=${!v}"
  done > "$R/config.txt"
  echo "git $(git rev-parse --short HEAD)" >> "$R/config.txt"
}
[ -f "$R/summary.tsv" ] || printf 'iter\tselfplay_s\tpositions\tval_value_mse\tval_policy_ce\tval_top1\ttrain_s\tvs_prev\tvs_it0\tvs_gen1\telo_chain\n' > "$R/summary.tsv"

# All finished sample directories, oldest first (rebuilt from disk each time)
sample_dirs() {
  ls -d "$PWD"/runs/full-2/gen_{1,2,3,4}/samples
  for j in $(seq 1 "$ITERS"); do
    for d in $(seq 1 "$DIRS"); do
      [ -f "$R/it$j/samples_$d/selfplay.json" ] && echo "$R/it$j/samples_$d"
    done
  done
  return 0
}

# decisive win rate of a test.json, and W-D-L
rate() { $PY -c "
import json; t = json.load(open('$1'))
w, d, g = t['wins'], t['draws'], t['games']
print(f\"{w / max(1, g - d):.3f} ({w}-{d}-{g - w - d})\")"; }

# Iteration 0: the supervised network trained on runs/full-2
if [ ! -f "$R/it0/model.onnx" ]; then
  mkdir -p "$R/it0"
  $PY arch/export.py "$INIT" "$R/it0/model.onnx" > "$R/it0/export.log" 2>&1
  log "it0: exported $INIT"
fi

for i in $(seq 1 "$ITERS"); do
  P=$R/it$((i - 1))
  C=$R/it$i
  mkdir -p "$C"
  [ -f "$C/summary.done" ] && continue
  # 1. self-play
  for d in $(seq 1 "$DIRS"); do
    S=$C/samples_$d
    if [ ! -f "$S/selfplay.json" ]; then
      rm -rf "$S"
      mkdir -p "$S"
      "$PLAY" train --model "trt16:$P/model.onnx" --games "$GAMES" \
        --in-flight 2000 --groups 2 --threads "$THREADS" \
        --seed $((1000 * i + d)) --out "$S" > "$S.log" 2>&1
      log "it$i: self-play $d/$DIRS done in $(grep -o '"wall_seconds": [0-9.]*' "$S/selfplay.json" | grep -o '[0-9.]*$') s"
    fi
  done
  # 2. dataset over the window
  if [ ! -f "$C/data.npz" ]; then
    sample_dirs | tail -n "$WINDOW" > "$C/window.txt"
    $PY arch/dataset.py "$C/data.tmp.npz" $(cat "$C/window.txt") > "$C/dataset.log" 2>&1
    mv "$C/data.tmp.npz" "$C/data.npz"
    log "it$i: dataset of $(wc -l < "$C/window.txt") directories, $(tail -1 "$C/dataset.log" | grep -o '[0-9]* positions')"
    # Raw samples outside the window are not needed again; keep their logs
    for old in $(sample_dirs | grep "^$R/" | head -n -"$WINDOW"); do
      rm -f "$old"/*.npy
    done
    [ -d "$R/it$((i - 2))" ] && rm -f "$R/it$((i - 2))/data.npz"
  fi
  # 3. train from scratch
  if [ ! -f "$C/net.pt" ]; then
    $PY arch/sup.py --data "$C/data.npz" --model res --width "$WIDTH" \
      --depth "$DEPTH" --block post --lr 3.2e-2 --mask --epochs "$EPOCHS" \
      --tag net --models "$C" --out "$C/result.jsonl" > "$C/train.log" 2>&1
    log "it$i: trained: $(grep 'epoch' "$C/train.log" | tail -1)"
  fi
  # 4. export and matches (for the record)
  if [ ! -f "$C/model.onnx" ]; then
    $PY arch/export.py "$C/net.pt" "$C/model.onnx.tmp" > "$C/export.log" 2>&1
    mv "$C/model.onnx.tmp" "$C/model.onnx"
  fi
  for opp in prev it0 gen1; do
    M=$C/match_$opp
    if [ ! -f "$M/test.json" ]; then
      case $opp in
        prev) B="trt16:$P/model.onnx" ;;
        it0) B="trt16:$R/it0/model.onnx" ;;
        gen1) B="trt:$ANCHOR" ;;
      esac
      if [ "$opp" = it0 ] && [ "$i" = 1 ]; then continue; fi  # same as prev
      rm -rf "$M"
      mkdir -p "$M"
      "$PLAY" test --new "trt16:$C/model.onnx" --best "$B" --games "$MATCH" \
        --threads "$THREADS" --seed $((7000 + i)) --out "$M" > "$M.log" 2>&1
      log "it$i: vs $opp: decisive win rate $(rate "$M/test.json")"
    fi
  done
  # 5. summary row; the Elo chain adds up the previous-iteration matches
  $PY - "$R" "$i" <<'EOF' >> "$R/summary.tsv"
import glob, json, math, sys
R, i = sys.argv[1], int(sys.argv[2])
C = f"{R}/it{i}"
def rate(path):
    try:
        t = json.load(open(path))
    except FileNotFoundError:
        return None
    return t["wins"] / max(1, t["games"] - t["draws"])
def elo(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return 400 * math.log10(p / (1 - p))
chain = sum(elo(rate(f"{R}/it{j}/match_prev/test.json")) for j in range(1, i + 1))
res = json.loads(open(f"{C}/result.jsonl").read().splitlines()[-1])["final"]
sp = sum(json.load(open(f))["wall_seconds"]
         for f in glob.glob(f"{C}/samples_*/selfplay.json"))
train_s = json.loads(open(f"{C}/result.jsonl").read().splitlines()[-1])["seconds"]
positions = open(f"{C}/dataset.log").read().split()[-2]
fmt = lambda x: "-" if x is None else f"{x:.3f}"
it0 = rate(f"{C}/match_prev/test.json") if i == 1 else rate(f"{C}/match_it0/test.json")
print("\t".join([str(i), f"{sp:.0f}", positions, f"{res['value_mse']:.4f}",
                 f"{res['policy_ce']:.4f}", f"{res['top1']:.4f}", f"{train_s:.0f}",
                 fmt(rate(f"{C}/match_prev/test.json")), fmt(it0),
                 fmt(rate(f"{C}/match_gen1/test.json")), f"{chain:+.0f}"]))
EOF
  touch "$C/summary.done"
  log "it$i: done; summary: $(tail -1 "$R/summary.tsv")"
done
log "finished $ITERS iterations"
