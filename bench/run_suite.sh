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
#
# The engine is measured at BOTH one thread and many. They answer different
# questions and a change can easily improve one while hurting the other:
#
#   st  single thread -- pure algorithmic cost, no scheduling or allocator
#       contention. This is the number that should track the profiler.
#   mt  many threads, same game count -- st/mt gives parallel efficiency, so a
#       regression in scaling (false sharing, allocator contention, load
#       imbalance) shows up here even when st is unchanged.
#   big many threads at the headline game count -- overall throughput.
PAIR_GAMES=50          # st and mt use this, so the two are directly comparable
BIG_GAMES=200
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

# Capture provenance BEFORE touching the output file. Re-recording over an
# existing tracked record would otherwise dirty the tree and make this field
# permanently report "yes", which would defeat the point of recording it.
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if git diff --quiet 2>/dev/null; then DIRTY=no; else DIRTY=yes; fi
CXXFLAGS_USED="$(grep -m1 '^CXXFLAGS' Makefile | cut -d= -f2- | xargs)"

echo "building..."
make -s all

{
  echo -e "#label\t${LABEL}"
  echo -e "#suite\t${SUITE}"
  echo -e "#date\t$(date -Iseconds)"
  echo -e "#commit\t${COMMIT}"
  echo -e "#dirty\t${DIRTY}"
  echo -e "#cxxflags\t${CXXFLAGS_USED}"
  echo -e "#config\tpair_games=${PAIR_GAMES} big_games=${BIG_GAMES} searches=${SEARCHES} per_eval=${PER_EVAL} threads=${THREADS} seed=${SEED} corpus=${CORPUS} reps=${REPS}"
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

collect_prefixed() {  # collect_prefixed <prefix> <binary> <args...>
  local prefix="$1"; shift
  collect "$@" | sed "s/^#METRIC /#METRIC ${prefix}_/"
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
  echo "engine benchmark, single thread (${REPS} reps)..."
  collect_prefixed st ./build/selfplay_bench \
    "$PAIR_GAMES" "$SEARCHES" "$PER_EVAL" 1 "$SEED" | reduce >> "$OUT"

  echo "engine benchmark, ${THREADS} threads (${REPS} reps)..."
  collect_prefixed mt ./build/selfplay_bench \
    "$PAIR_GAMES" "$SEARCHES" "$PER_EVAL" "$THREADS" "$SEED" | reduce >> "$OUT"

  echo "engine benchmark, ${THREADS} threads at ${BIG_GAMES} games (${REPS} reps)..."
  collect_prefixed big ./build/selfplay_bench \
    "$BIG_GAMES" "$SEARCHES" "$PER_EVAL" "$THREADS" "$SEED" | reduce >> "$OUT"

  # Parallel efficiency, derived from the matched st/mt pair.
  python3 - "$OUT" "$THREADS" <<'PY' >> "$OUT"
import sys
path, threads = sys.argv[1], int(sys.argv[2])
vals = {}
for line in open(path, encoding="utf-8"):
    parts = line.split("\t")
    if len(parts) == 4:
        vals[parts[0]] = float(parts[1])
st, mt = vals.get("st_engine_seconds"), vals.get("mt_engine_seconds")
if st and mt:
    speedup = st / mt
    print(f"speedup_{threads}t\t{speedup:.4f}\t{speedup:.4f}\t{speedup:.4f}")
    eff = speedup / threads * 100.0
    print(f"parallel_efficiency_pct\t{eff:.4f}\t{eff:.4f}\t{eff:.4f}")
PY
fi

echo "wrote $OUT"
column -t -s $'\t' "$OUT"
