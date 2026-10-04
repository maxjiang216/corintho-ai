#!/bin/bash
# Alpha-beta self-play from a blank network (worklog
# 2026-09-25-nn-architectures, entry 25).
#
#   arch/ab_loop.sh            (settings by environment, defaults below)
#
# Generation g: the network of g-1 plays POSITIONS positions (ab_selfplay),
# targets are mixed from the last WINDOW generations (exact where solved,
# else lambda x search value + (1 - lambda) x outcome), and network g is
# trained from network g-1 (warm start). Every EVAL generations, network g
# plays MCTS (ab_match, as in entry 23). Resumable: a step whose output
# exists is skipped, so the loop can be stopped and restarted, and
# POSITIONS or LAM changed between generations. Self-play progress goes to
# progress.log every REPORT seconds (30).
#
# Growing generations (ADAPT=1): each match against MCTS (MATCH_GAMES games)
# is compared with the best so far; after STALL_EVALS matches in a row
# without beating it by STALL_MARGIN, POSITIONS and STEPS double (up to
# MAX_POSITIONS). The schedule lives in $RUN/schedule (positions, steps,
# best score, stalls), so a restart continues it; delete the file to
# start the schedule over from POSITIONS and STEPS. KEEP_DATA=0 deletes each
# generation's positions once they leave the training window.
set -e
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/ab-1}
GENS=${GENS:-30}
POSITIONS=${POSITIONS:-200000}
DEPTH=${DEPTH:-5}
THREADS=${THREADS:-18}
WINDOW=${WINDOW:-4}
STEPS=${STEPS:-20000}
LAM0=${LAM0:-0.3}       # lambda at generation 1, rising by LAM_STEP a
LAM_STEP=${LAM_STEP:-0.05}  # generation up to LAM1
LAM1=${LAM1:-0.8}
EVAL=${EVAL:-5}
MATCH_GAMES=${MATCH_GAMES:-200}
ADAPT=${ADAPT:-0}
STALL_EVALS=${STALL_EVALS:-2}
STALL_MARGIN=${STALL_MARGIN:-0.01}
MAX_POSITIONS=${MAX_POSITIONS:-3200000}
KEEP_DATA=${KEEP_DATA:-1}
AZ=${AZ:-runs/solve-0/gen_5/model.onnx}
TEST=${TEST:-runs/nnue/test.npz}
NET_ARGS="--hidden 256 --mono-m 64 --mono-m2 16"
PY=.venv/bin/python
D=arch/nnue_distill.py
mkdir -p "$RUN"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/progress.log"; }

BEST=0 STALLS=0
if [ "$ADAPT" = 1 ]; then
  if [ -f "$RUN/schedule" ]; then
    read -r POSITIONS STEPS BEST STALLS < "$RUN/schedule"
  else
    echo "$POSITIONS $STEPS $BEST $STALLS" > "$RUN/schedule"
  fi
  log "schedule: $POSITIONS positions, $STEPS steps, best $BEST, stalls $STALLS"
fi

if [ ! -f "$RUN/gen_0/net.bin" ]; then
  mkdir -p "$RUN/gen_0"
  $PY $D init "$RUN/gen_0/net.pt" $NET_ARGS --seed "${SEED:-0}"
  $PY $D export "$RUN/gen_0/net.pt" "$RUN/gen_0/net.bin"
fi
for g in $(seq 1 "$GENS"); do
  G=$RUN/gen_$g P=$RUN/gen_$((g - 1))
  mkdir -p "$G"
  if [ ! -f "$G/data.npy" ]; then
    log "gen $g: self-play, $POSITIONS positions, depth $DEPTH"
    build/ab_selfplay --small "$P/net.bin" --out "$G/data.tmp.npy" \
      --positions "$POSITIONS" --threads "$THREADS" --depth "$DEPTH" \
      --seed "$g" --report-seconds "${REPORT:-30}" \
      2> >(tee -a "$G/selfplay.err" | sed "s/^/  gen $g: /" >> "$RUN/progress.log") \
      | tee "$G/selfplay.txt"
    mv "$G/data.tmp.npy" "$G/data.npy"
  fi
  if [ ! -f "$G/net.bin" ]; then
    lam=$(python3 -c "print(round(min($LAM1, $LAM0 + ($g - 1) * $LAM_STEP), 3))")
    gens=$(for k in $(seq $((g - WINDOW + 1)) "$g"); do
      [ "$k" -ge 1 ] && echo "$RUN/gen_$k/data.npy"; done)
    log "gen $g: train, lambda $lam, window $(echo $gens | wc -w)"
    $PY $D mix "$G/targets.npz" $gens --lam "$lam"
    $PY $D train "$G/targets.npz" "$TEST" $NET_ARGS --clip-w2 1.98 \
      --min-p 20 --steps "$STEPS" --init "$P/net.pt" --select last \
      --out "$G/net.pt" --results "$RUN/results.jsonl" --tag "gen_$g" \
      --seed "$g" --report-seconds "${REPORT:-30}" > "$G/train.txt" \
      2> >(sed "s/^/  gen $g train: /" >> "$RUN/progress.log")
    rm "$G/targets.npz"
    $PY $D export "$G/net.pt" "$G/net.bin"
    if [ "$KEEP_DATA" = 0 ] && [ $((g - WINDOW + 1)) -gt 1 ]; then
      rm -f "$RUN/gen_$((g - WINDOW))/data.npy"
    fi
  fi
  if [ $((g % EVAL)) -eq 0 ] && [ ! -f "$G/match.txt" ]; then
    log "gen $g: match against MCTS"
    build/ab_match --small "$G/net.bin" --az "$AZ" --games "$MATCH_GAMES" \
      --threads "$THREADS" --ab-seconds 2 --mcts-searches 11000 --seed 1 \
      2>/dev/null > "$G/match.tmp" && mv "$G/match.tmp" "$G/match.txt"
    log "gen $g: $(head -1 "$G/match.txt")"
    if [ "$ADAPT" = 1 ]; then
      score=$(sed -n 's/.*(score \([0-9.]*\),.*/\1/p' "$G/match.txt" | head -1)
      if python3 -c "import sys; sys.exit(0 if $score > $BEST + $STALL_MARGIN else 1)"; then
        BEST=$score STALLS=0
      else
        STALLS=$((STALLS + 1))
      fi
      if [ "$STALLS" -ge "$STALL_EVALS" ] && [ "$POSITIONS" -lt "$MAX_POSITIONS" ]; then
        POSITIONS=$((POSITIONS * 2)) STEPS=$((STEPS * 2)) STALLS=0
        log "stalled: now $POSITIONS positions, $STEPS steps per generation"
      fi
      echo "$POSITIONS $STEPS $BEST $STALLS" > "$RUN/schedule"
      log "schedule: best $BEST, stalls $STALLS"
    fi
  fi
done
