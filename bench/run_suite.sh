#!/usr/bin/env bash
# Record one labelled benchmark run for before/after comparison.
#
#   ./run_suite.sh <label> [suite] [reps]
#
#   label   name for the record, e.g. "before-bitboard"
#   suite   game | engine | all      (default: all)
#   reps    timed repetitions        (default: 5)
#
# Writes results/<label>.tsv. Timed metrics are reported as the median across
# reps, with min and max, because single runs on this machine vary by a few
# percent. Digests must be identical across reps or the run is rejected.
#
# Compare two records with ./compare.py results/before.tsv results/after.tsv

set -euo pipefail
cd "$(dirname "$0")"

LABEL="${1:?usage: ./run_suite.sh <label> [suite] [reps]}"
SUITE="${2:-all}"
REPS="${3:-5}"

# Engine benchmark parameters. Keep these fixed across all comparisons.
GAMES=200
SEARCHES=1600
PER_EVAL=16
THREADS=14
SEED=12345

# Microbenchmark parameters.
CORPUS=20000
MICRO_REPS=50

mkdir -p results
OUT="results/${LABEL}.tsv"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "building..."
make -s all

{
  echo -e "#label\t${LABEL}"
  echo -e "#suite\t${SUITE}"
  echo -e "#date\t$(date -Iseconds)"
  echo -e "#commit\t$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo -e "#dirty\t$(git diff --quiet 2>/dev/null && echo no || echo yes)"
  echo -e "#cxxflags\t$(grep -m1 '^CXXFLAGS' Makefile | cut -d= -f2- | xargs)"
  echo -e "#config\tgames=${GAMES} searches=${SEARCHES} per_eval=${PER_EVAL} threads=${THREADS} seed=${SEED} corpus=${CORPUS} reps=${REPS}"
} > "$OUT"

# --- Digests: must be reproducible, so any variation is a hard error. ---
echo "digests..."
./build/golden | grep '^#METRIC' | awk '{print $2"\t"$3}' > "$TMP/d0"
for _ in $(seq 2); do
  ./build/golden | grep '^#METRIC' | awk '{print $2"\t"$3}' > "$TMP/dn"
  if ! diff -q "$TMP/d0" "$TMP/dn" > /dev/null; then
    echo "FATAL: digests are not reproducible across runs" >&2
    exit 1
  fi
done
cat "$TMP/d0" >> "$OUT"

# --- Collect repeated samples, then reduce to median/min/max. ---
collect() {  # collect <binary> <args...>
  local bin="$1"; shift
  for _ in $(seq "$REPS"); do
    "$bin" "$@" | grep '^#METRIC' || true
  done
}

reduce() {  # reduce < raw metric lines
  python3 -c '
import sys, collections, statistics
vals = collections.defaultdict(list)
order = []
for line in sys.stdin:
    parts = line.split()
    if len(parts) != 3:
        continue
    _, key, value = parts
    try:
        v = float(value)
    except ValueError:
        continue
    if key not in vals:
        order.append(key)
    vals[key].append(v)
for key in order:
    xs = vals[key]
    print(f"{key}\t{statistics.median(xs):.4f}\t{min(xs):.4f}\t{max(xs):.4f}")
'
}

if [ "$SUITE" = "game" ] || [ "$SUITE" = "all" ]; then
  echo "microbenchmarks (${REPS} reps)..."
  collect ./build/micro_bench "$CORPUS" "$MICRO_REPS" | reduce >> "$OUT"
fi

if [ "$SUITE" = "engine" ] || [ "$SUITE" = "all" ]; then
  echo "engine benchmark (${REPS} reps)..."
  collect ./build/selfplay_bench "$GAMES" "$SEARCHES" "$PER_EVAL" "$THREADS" "$SEED" \
    | reduce >> "$OUT"
fi

echo "wrote $OUT"
column -t -s $'\t' "$OUT"
