#!/bin/bash
# Entry 24: symmetry-tied and reserve-monotone small networks against the
# plain h256-clip of entry 23, distilled on the same data, 3 seeds each.
#   arch/nnue_sweep.sh TARGETS.npz TEST.npz [extra train options]
# Results: runs/nnue/results.jsonl (tags e24_<variant>_s<seed>).
cd "$(dirname "$0")/.."
TARGETS=$1 TEST=$2
shift 2
EXTRA=("$@")
run() { .venv/bin/python arch/nnue_distill.py train "$TARGETS" "$TEST" --hidden 256 --clip-w2 1.98 --min-p 20 --steps 80000 --eval-every 4000 "${EXTRA[@]}" "$@"; }
for seed in 0 1 2; do
  run --seed $seed --tag e24_plain_s$seed
  run --seed $seed --tie --tag e24_tie_s$seed
  run --seed $seed --mono-m 64 --mono-m2 16 --tag e24_mono_s$seed
  run --seed $seed --tie --mono-m 64 --mono-m2 16 --tag e24_tiemono_s$seed
done
