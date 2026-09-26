#!/usr/bin/env bash
# Extra anchor matches for an overnight run (worklog 2026-09-25-nn-architectures,
# entry 05): after each finished iteration, 3200 games against each anchor on
# the spare threads, for a sharper plateau measurement than the loop's own
# 1600-game matches. Results: runs/$NAME/extra/anchors.tsv.
# ANCHORS: space-separated name=model entries (driver syntax, e.g. trt16:path);
# default it0, it1 (of this run) and gen 1.
cd "$(dirname "$0")/.."
NAME=${NAME:-night-1}
ITERS=${ITERS:-8}
R=$PWD/runs/$NAME
X=$R/extra
mkdir -p "$X"
[ -f "$X/anchors.tsv" ] || printf 'iter\topponent\twins\tdraws\tlosses\tdecisive_win_rate\n' > "$X/anchors.tsv"
G1=$PWD/runs/full-2/gen_1/model.onnx
ANCHORS=${ANCHORS:-"it0=trt16:$R/it0/model.onnx it1=trt16:$R/it1/model.onnx gen1=trt:$G1"}
for i in $(seq 1 "$ITERS"); do
  until [ -f "$R/it$i/summary.done" ]; do
    grep -q "supervisor: overnight.sh failed (5" "$R/progress.log" 2>/dev/null && exit 1
    grep -q "supervisor: finished" "$R/progress.log" 2>/dev/null && [ ! -f "$R/it$i/summary.done" ] && exit 0
    sleep 60
  done
  for entry in $ANCHORS; do
    opp=${entry%%=*}
    B=${entry#*=}
    [ "$opp" = "it$i" ] && continue
    M=$X/it${i}_vs_${opp}_3200
    [ -f "$M/test.json" ] && continue
    rm -rf "$M"; mkdir -p "$M"
    build/corintho_play test --new "trt16:$R/it$i/model.onnx" --best "$B" \
      --games 3200 --threads 6 --seed $((9000 + i)) --out "$M" > "$M.log" 2>&1
    python3 -c "
import json; t = json.load(open('$M/test.json')); w, d, g = t['wins'], t['draws'], t['games']
print('\t'.join(map(str, [$i, '$opp', w, d, g - w - d, round(w / (g - d), 4)])))" >> "$X/anchors.tsv"
  done
done
