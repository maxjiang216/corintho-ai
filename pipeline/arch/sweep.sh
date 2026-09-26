#!/bin/bash
# Sweep 2, after the symmetry fix (worklog/2026-09-25-nn-architectures, entry 03): every run masked,
# 20 epochs, gens 1-3 -> gen 4. Appends to runs/sup/results.jsonl.
cd "$(dirname "$0")/.."
run() { .venv/bin/python arch/sup.py --data runs/sup/full2.npz --epochs 20 --mask "$@" 2>&1 | grep -E "^(mlp|res)|epoch 19"; }
run --no-sym --tag s2_nosym
run --lines any --tag s2_lines_any
run --lines type --tag s2_lines_type
run --legal-input --tag s2_legal_input
run --model res --width 256 --depth 4 --norm bn --tag s2_res256x4_bn
run --model res --width 256 --depth 4 --norm ln --tag s2_res256x4_ln
run --model res --width 256 --depth 4 --norm none --tag s2_res256x4_none
run --width 64 --depth 6 --tag s2_mlp64x6
run --width 256 --depth 6 --tag s2_mlp256x6
