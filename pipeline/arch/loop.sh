#!/usr/bin/env bash
# Small-generation self-play loop (worklog 2026-09-25-nn-architectures,
# entry 06): many small generations, few epochs, warm starts, a window of the
# last few generations. Built from checked tools: corintho_play,
# arch/dataset.py, arch/sup.py, arch/export.py.
#
#   arch/loop.sh                       (from pipeline/; resumes where it stopped)
#   SCRIPT=arch/loop.sh arch/overnight_supervise.sh    (reruns after failures)
#
# Generation g = 1..GENS:
#   1. self-play: GAMES games with gen g-1's network (fp16 TensorRT)
#   2. compact: arch/dataset.py -> gen_g/data.npz (checked, one row per
#      position, ~0.2 GB per 25k games); the raw samples are then deleted
#   3. train: continue gen g-1's network (warm start) for EPOCHS passes over
#      the last WINDOW generations' data, VAL_FRACTION held out
#   4. every MATCH_EVERY generations, matches for the record against fixed
#      anchors and against the network MATCH_EVERY generations back
# No gate: every generation's network plays the next one's games.
# Every step's output is its marker, so reruns resume.
# Results: runs/$NAME/progress.log and summary.tsv; matches.tsv.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${NAME:-loop-1}
GENS=${GENS:-60}
GAMES=${GAMES:-25000}
WINDOW=${WINDOW:-4}          # generations of data per training step
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-2}               # warm-start rate (entry 06: best of 1e-3..3.2e-2)
WARMUP=${WARMUP:-100}
VAL_FRACTION=${VAL_FRACTION:-0.02}
THREADS=${THREADS:-14}
LOGGED=${LOGGED:-50}         # full game logs written per generation
WIDTH=${WIDTH:-512}
DEPTH=${DEPTH:-4}
MATCH_EVERY=${MATCH_EVERY:-5}
MATCH=${MATCH:-1600}
# End self-play games by exact solution at horizon P <= SOLVE_P (entry 14;
# 0: off). Matches are played out as before. Off by default: at 27 the
# network got much weaker in 5 generations (entry 18).
SOLVE_P=${SOLVE_P:-0}
SOLVE_THREADS=${SOLVE_THREADS:-6}
INIT=${INIT:-runs/night-2/it1/net.pt}
# Compact data files standing in for the generations before gen 1
SEED_DATA=${SEED_DATA:-}
# Fixed anchors for the matches: name=model (driver syntax)
ANCHORS=${ANCHORS:-"n1it0=trt16:$PWD/runs/night-1/it0/model.onnx gen1=trt:$PWD/runs/full-2/gen_1/model.onnx"}

R=$PWD/runs/$NAME
PLAY=$PWD/build/corintho_play
PY=$PWD/.venv/bin/python
mkdir -p "$R"
log() { echo "$(date '+%F %T') $*" | tee -a "$R/progress.log"; }
[ -f "$R/config.txt" ] || {
  for v in NAME GENS GAMES WINDOW EPOCHS LR WARMUP VAL_FRACTION THREADS LOGGED WIDTH DEPTH MATCH_EVERY MATCH SOLVE_P SOLVE_THREADS INIT SEED_DATA ANCHORS; do
    echo "$v=${!v}"
  done > "$R/config.txt"
  echo "git $(git rev-parse --short HEAD)" >> "$R/config.txt"
}
[ -f "$R/summary.tsv" ] || printf 'gen\tselfplay_s\tpositions\ttrain_s\tval_value_mse\tval_policy_ce\tval_top1\tturns_per_game\tfirst_player_score\n' > "$R/summary.tsv"
[ -f "$R/matches.tsv" ] || printf 'gen\topponent\twins\tdraws\tlosses\tdecisive_win_rate\n' > "$R/matches.tsv"

# Generation 0: the starting network
if [ ! -f "$R/gen_0/model.onnx" ]; then
  mkdir -p "$R/gen_0"
  cp "$INIT" "$R/gen_0/net.pt"
  $PY arch/export.py "$R/gen_0/net.pt" "$R/gen_0/model.onnx.tmp" > "$R/gen_0/export.log" 2>&1
  mv "$R/gen_0/model.onnx.tmp" "$R/gen_0/model.onnx"
  log "gen 0: from $INIT"
fi

# Data files, oldest first: the seed data, then every finished generation's
data_files() {
  for s in $SEED_DATA; do echo "$s"; done
  for j in $(seq 1 "$1"); do
    [ -f "$R/gen_$j/data.npz" ] && echo "$R/gen_$j/data.npz"
  done
  return 0
}

match() {  # match NEW_GEN OPPONENT_NAME OPPONENT_MODEL
  local M=$R/gen_$1/match_$2
  [ -f "$M/test.json" ] && return 0
  rm -rf "$M"
  mkdir -p "$M"
  "$PLAY" test --new "trt16:$R/gen_$1/model.onnx" --best "$3" --games "$MATCH" \
    --threads "$THREADS" --seed $((7000 + $1)) --out "$M" > "$M.log" 2>&1
  local row
  row=$($PY -c "
import json; t = json.load(open('$M/test.json')); w, d, g = t['wins'], t['draws'], t['games']
print('\t'.join(map(str, [$1, '$2', w, d, g - w - d, round(w / max(1, g - d), 4)])))")
  echo "$row" >> "$R/matches.tsv"
  log "gen $1: vs $2: $(echo "$row" | cut -f3-6 | tr '\t' ' ')"
}

for g in $(seq 1 "$GENS"); do
  P=$R/gen_$((g - 1))
  C=$R/gen_$g
  [ -f "$C/gen.done" ] && continue
  mkdir -p "$C"
  # 1. self-play
  if [ ! -f "$C/samples/selfplay.json" ]; then
    rm -rf "$C/samples"
    mkdir -p "$C/samples"
    "$PLAY" train --model "trt16:$P/model.onnx" --games "$GAMES" \
      --in-flight 2000 --groups 2 --threads "$THREADS" --seed $((100000 + g)) \
      --logged "$LOGGED" --solve-p "$SOLVE_P" --solve-threads "$SOLVE_THREADS" \
      --out "$C/samples" > "$C/selfplay.log" 2>&1
  fi
  # 2. compact, then drop the raw samples (selfplay.json and logs stay)
  if [ ! -f "$C/data.npz" ]; then
    $PY arch/dataset.py "$C/data.tmp.npz" "$C/samples" > "$C/dataset.log" 2>&1
    mv "$C/data.tmp.npz" "$C/data.npz"
  fi
  rm -f "$C"/samples/*.npy
  # 3. train: warm start from the previous generation
  if [ ! -f "$C/net.pt" ]; then
    data_files "$g" | tail -n "$WINDOW" > "$C/window.txt"
    $PY arch/sup.py --data $(cat "$C/window.txt") --init "$P/net.pt" \
      --model res --width "$WIDTH" --depth "$DEPTH" --block post --mask \
      --lr "$LR" --epochs "$EPOCHS" --warmup "$WARMUP" \
      --val-fraction "$VAL_FRACTION" --seed "$g" --tag net.tmp --models "$C" \
      --out "$C/result.jsonl" > "$C/train.log" 2>&1
    mv "$C/net.tmp.pt" "$C/net.pt"
  fi
  if [ ! -f "$C/model.onnx" ]; then
    $PY arch/export.py "$C/net.pt" "$C/model.onnx.tmp" > "$C/export.log" 2>&1
    mv "$C/model.onnx.tmp" "$C/model.onnx"
  fi
  # summary row
  $PY - "$C" "$g" <<'EOF' >> "$R/summary.tsv"
import json, sys
C, g = sys.argv[1], sys.argv[2]
sp = json.load(open(f"{C}/samples/selfplay.json"))
r = json.loads(open(f"{C}/result.jsonl").read().splitlines()[-1])
f = r["final"]
print("\t".join([g, f"{sp['wall_seconds']:.0f}", str(r["train_positions"] + r["val_positions"]),
                 f"{r['seconds']:.0f}", f"{f['value_mse']:.4f}", f"{f['policy_ce']:.4f}",
                 f"{f['top1']:.4f}", f"{sp['turns'] / sp['games']:.2f}",
                 f"{sp['first_player_score']:.4f}"]))
EOF
  log "gen $g: $(tail -1 "$R/summary.tsv" | cut -f2- | tr '\t' ' ')"
  # 4. matches every MATCH_EVERY generations
  if [ $((g % MATCH_EVERY)) -eq 0 ]; then
    for entry in $ANCHORS; do
      match "$g" "${entry%%=*}" "${entry#*=}"
    done
    match "$g" gen0 "trt16:$R/gen_0/model.onnx"
    [ "$g" -gt "$MATCH_EVERY" ] &&
      match "$g" "gen$((g - MATCH_EVERY))" "trt16:$R/gen_$((g - MATCH_EVERY))/model.onnx"
  fi
  touch "$C/gen.done"
done
log "finished $GENS generations"
