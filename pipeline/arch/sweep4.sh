#!/bin/bash
# Sweep 4 (worklog/2026-09-25-nn-architectures, entry 04): sizes x extra
# inputs, residual pre-activation BatchNorm, masked, random symmetries,
# 20 epochs. Settings from sweep 3 may be passed through: sweep4.sh --lr 1e-3
cd "$(dirname "$0")/.."
EXTRA=("$@")
run() { .venv/bin/python arch/sup.py --data runs/sup/full2.npz --epochs 20 --mask --model res "$@" 2>&1 | grep -E "^res|epoch 19"; }
for size in "256 4" "384 4" "512 4" "256 8"; do
  set -- $size
  w=$1 d=$2
  shift 2
  run --width $w --depth $d "${EXTRA[@]}" --tag s4_${w}x${d}
  run --width $w --depth $d "${EXTRA[@]}" --lines any --tag s4_${w}x${d}_lines
  run --width $w --depth $d "${EXTRA[@]}" --legal-input --tag s4_${w}x${d}_legal
done
