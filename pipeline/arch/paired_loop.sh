#!/usr/bin/env bash
# Two arch/loop.sh runs advanced one generation at a time, in turn, so both
# see the same machine conditions (worklog 2026-09-25-nn-architectures,
# entry 16: solver on vs off from the same starting network).
#
#   arch/paired_loop.sh           (from pipeline/; resumes where it stopped)
#
# Each run keeps its own config.txt (written on its first start); only GENS
# is raised here, one generation at a time.
set -euo pipefail
cd "$(dirname "$0")/.."

GENS=${GENS:-20}
RUN_A=${RUN_A:-solve-27}
RUN_B=${RUN_B:-solve-0}
SOLVE_P_A=${SOLVE_P_A:-27}
SOLVE_P_B=${SOLVE_P_B:-0}

exec 9> runs/paired_loop.lock
flock -n 9 || { echo "paired_loop.sh is already running" >&2; exit 1; }

for g in $(seq 1 "$GENS"); do
  NAME=$RUN_A SOLVE_P=$SOLVE_P_A GENS=$g arch/loop.sh
  NAME=$RUN_B SOLVE_P=$SOLVE_P_B GENS=$g arch/loop.sh
done
