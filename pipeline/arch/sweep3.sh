#!/bin/bash
# Sweep 3 (worklog/2026-09-25-nn-architectures, entry 04): the simple choices
# on residual 256x4 BatchNorm, masked, random symmetries, 20 epochs.
cd "$(dirname "$0")/.."
run() { .venv/bin/python arch/sup.py --data runs/sup/full2.npz --epochs 20 --mask --model res --width 256 --depth 4 "$@" 2>&1 | grep -E "^res|epoch 19"; }
run --seed 1 --tag s3_seed1
run --seed 2 --tag s3_seed2
run --block post --tag s3_post
run --lr 1e-3 --tag s3_lr1e-3
run --lr 4e-3 --tag s3_lr4e-3
run --wd 1e-3 --tag s3_wd1e-3
