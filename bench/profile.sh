#!/usr/bin/env bash
# Tiered profiling.
#
#   ./profile.sh <label> [tier] [threads]
#
#   quick   gprofng, sampling, ~zero overhead, LARGE runs      ~90 s
#   deep    callgrind + cachegrind, exact, TINY runs           ~15 s
#   heap    dhat, allocation behaviour, TINY run               ~5 s
#   all     everything (default)                               ~2 min
#
# Writes results/profiles/<label>-*.txt
#
# ---------------------------------------------------------------------------
# Why tiers, and what transfers between them
# ---------------------------------------------------------------------------
#
# Heavyweight tools (callgrind, cachegrind, dhat) run 20-100x slower, so they
# get tiny runs. Sampling tools cost nothing, so they get large realistic runs.
# The question is whether the tiny run measures the same thing as the big one.
#
# Measured, by comparing requests/turn across run sizes:
#
#   games=3   searches=400    341.7 requests/turn
#   games=10  searches=400    341.8
#   games=40  searches=1600  1280.6
#   games=300 searches=1600  1271.9
#
# Tree shape is governed by SEARCH count, not game count. Changing games by 100x
# moves requests/turn by under 1%; changing searches by 4x moves it by 3.7x.
#
# Hence the rule every tier below follows: **fix the search count, vary only the
# game count.** A tier that changed searches would be measuring a different
# workload, not the same workload more precisely.
#
# What transfers across run sizes:
#   - instruction counts and attribution      yes
#   - branch prediction behaviour             yes
#   - allocation counts and sizes             yes
#
# What does NOT transfer:
#   - cache miss rates, especially last-level. These depend on the size of the
#     live search tree, which scales with games x threads. A 3-game
#     single-threaded run has a working set that fits in L2, so cachegrind will
#     report near-zero LL misses no matter how bad the real locality is.
#     Treat cachegrind's D1/DL numbers as a LOWER BOUND on the real problem.

set -euo pipefail
cd "$(dirname "$0")"

LABEL="${1:?usage: ./profile.sh <label> [quick|deep|heap|all] [threads]}"
TIER="${2:-all}"
THREADS="${3:-14}"

# Fixed across every tier. See the note above.
SEARCHES=1600
PER_EVAL=16
SEED=12345

# gprofng undersamples short runs badly: a 3-second run gave ~35 samples and
# percentages that swung 10 points between runs. Sized for ~300+ samples.
# Read only its percentages; its absolute seconds capture roughly a tenth of
# true CPU time.
GAMES_ST=300
GAMES_MT=400
# Heavyweight tools: small enough to stay under ~10 s each at 20-100x slowdown.
GAMES_HEAVY=3
GAMES_DHAT=2

OUTDIR="results/profiles"
mkdir -p "$OUTDIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "building..."
make -s all

header() {
  echo "# $1"
  echo "# label:    ${LABEL}"
  echo "# commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "# dirty:    $(git diff --quiet 2>/dev/null && echo no || echo yes)"
  echo "# cxxflags: $(grep -m1 '^CXXFLAGS' Makefile | cut -d= -f2- | xargs)"
  echo "# date:     $(date -Iseconds)"
  echo
}

run_quick() {
  echo "gprofng, 1 thread, ${GAMES_ST} games..."
  rm -rf "$TMP/st.er"
  gprofng collect app -o "$TMP/st.er" -p on \
    ./build/selfplay_bench "$GAMES_ST" "$SEARCHES" "$PER_EVAL" 1 "$SEED" \
    > /dev/null 2>&1
  {
    header "gprofng, 1 thread, ${GAMES_ST} games x ${SEARCHES} searches"
    echo "# Percentages only -- absolute seconds are unreliable."
    echo
    gprofng display text -functions -limit 30 "$TMP/st.er" 2>/dev/null
  } > "$OUTDIR/${LABEL}-st.txt"

  echo "gprofng, ${THREADS} threads, ${GAMES_MT} games..."
  rm -rf "$TMP/mt.er"
  gprofng collect app -o "$TMP/mt.er" -p on \
    ./build/selfplay_bench "$GAMES_MT" "$SEARCHES" "$PER_EVAL" "$THREADS" "$SEED" \
    > /dev/null 2>&1
  {
    header "gprofng, ${THREADS} threads, ${GAMES_MT} games x ${SEARCHES} searches"
    echo "# Compare against -st.txt: anything prominent here but not there"
    echo "# (libgomp, malloc internals, futex) is parallel overhead, not work."
    echo
    gprofng display text -functions -limit 30 "$TMP/mt.er" 2>/dev/null
  } > "$OUTDIR/${LABEL}-mt.txt"
}

run_deep() {
  echo "callgrind, ${GAMES_HEAVY} games (exact instruction counts)..."
  valgrind --tool=callgrind --callgrind-out-file="$TMP/cg.out" \
    ./build/selfplay_bench "$GAMES_HEAVY" "$SEARCHES" "$PER_EVAL" 1 "$SEED" \
    > /dev/null 2>&1
  {
    header "callgrind, 1 thread, ${GAMES_HEAVY} games x ${SEARCHES} searches"
    echo "# Exact and perfectly repeatable. Use to confirm a change removed the"
    echo "# instructions it was meant to remove."
    echo
    callgrind_annotate --auto=no "$TMP/cg.out" 2>/dev/null | sed -n '18,60p'
  } > "$OUTDIR/${LABEL}-callgrind.txt"

  echo "cachegrind, ${GAMES_HEAVY} games (cache + branch simulation)..."
  valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
    --cachegrind-out-file="$TMP/cach.out" \
    ./build/selfplay_bench "$GAMES_HEAVY" "$SEARCHES" "$PER_EVAL" 1 "$SEED" \
    > /dev/null 2>&1
  {
    header "cachegrind, 1 thread, ${GAMES_HEAVY} games x ${SEARCHES} searches"
    echo "# Columns: Ir I1mr ILmr Dr D1mr DLmr Dw D1mw DLmw Bc Bcm Bi Bim"
    echo "#"
    echo "# Bcm (conditional branch mispredicts) is the interesting column for"
    echo "# move generation; D1mr (L1 data read misses) for the search."
    echo "#"
    echo "# Cache numbers are a LOWER BOUND: this run's working set fits in L2."
    echo
    echo "== totals =="
    cg_annotate --auto=no "$TMP/cach.out" 2>/dev/null | grep "PROGRAM TOTALS"
    echo
    echo "== by branch mispredict =="
    cg_annotate --auto=no --sort=Bcm "$TMP/cach.out" 2>/dev/null \
      | awk '/file:function/,0' | head -14
    echo
    echo "== by L1 data read miss =="
    cg_annotate --auto=no --sort=D1mr "$TMP/cach.out" 2>/dev/null \
      | awk '/file:function/,0' | head -14
  } > "$OUTDIR/${LABEL}-cachegrind.txt"
}

run_heap() {
  echo "dhat, ${GAMES_DHAT} games (allocation behaviour)..."
  valgrind --tool=dhat --dhat-out-file="$TMP/dhat.out" \
    ./build/selfplay_bench "$GAMES_DHAT" "$SEARCHES" "$PER_EVAL" 1 "$SEED" \
    > /dev/null 2>"$TMP/dhat.log"
  {
    header "dhat, 1 thread, ${GAMES_DHAT} games x ${SEARCHES} searches"
    echo "# Allocation counts and peak heap. Relevant to the node arena and to"
    echo "# the to_eval_ oversizing (PLAN.md 10.2)."
    echo
    grep -Ei "total:|At t-gmax|At t-end|reads:|writes:" "$TMP/dhat.log" || cat "$TMP/dhat.log"
  } > "$OUTDIR/${LABEL}-dhat.txt"
}

case "$TIER" in
  quick) run_quick ;;
  deep)  run_deep ;;
  heap)  run_heap ;;
  all)   run_quick; run_deep; run_heap ;;
  *)     echo "unknown tier: $TIER" >&2; exit 1 ;;
esac

echo
echo "wrote:"
ls "$OUTDIR/${LABEL}"-*.txt | sed 's/^/  /'
