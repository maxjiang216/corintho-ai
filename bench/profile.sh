#!/usr/bin/env bash
# Capture profiles at one thread and at many, saved for the worklog.
#
#   ./profile.sh <label> [threads]
#
# Writes results/profiles/<label>-{st,mt,callgrind}.txt
#
# Single- and multi-threaded profiles answer different questions:
#
#   st          where the algorithmic time goes, with no scheduling or
#               allocator contention mixed in. Compare against callgrind.
#   mt          where the time goes under real contention. Anything that
#               appears here but not in st -- libgomp, malloc internals,
#               futex -- is parallel overhead rather than work.
#   callgrind   exact instruction counts, single-threaded. Use to confirm a
#               change removed the instructions it was meant to remove.

set -euo pipefail
cd "$(dirname "$0")"

LABEL="${1:?usage: ./profile.sh <label> [threads]}"
THREADS="${2:-14}"

# gprofng undersamples short runs badly: a 3-second run yielded ~35 samples,
# and the resulting percentages swung by 10 points between runs. Both profiles
# are therefore sized to run long enough for the percentages to settle
# (~300+ samples). This script is slow by design; it is not run every
# iteration. Treat gprofng's absolute seconds as unreliable and read only the
# percentages -- it captures roughly a tenth of the true CPU time here.
GAMES_ST=300
GAMES_MT=400
SEARCHES=1600
PER_EVAL=16
SEED=12345

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

# --- gprofng, single thread ---
echo "gprofng, 1 thread..."
rm -rf "$TMP/st.er"
gprofng collect app -o "$TMP/st.er" -p on \
  ./build/selfplay_bench "$GAMES_ST" "$SEARCHES" "$PER_EVAL" 1 "$SEED" \
  > /dev/null 2>&1
{
  header "gprofng, 1 thread, ${GAMES_ST} games x ${SEARCHES} searches"
  gprofng display text -functions -limit 30 "$TMP/st.er" 2>/dev/null
} > "$OUTDIR/${LABEL}-st.txt"

# --- gprofng, many threads ---
echo "gprofng, ${THREADS} threads..."
rm -rf "$TMP/mt.er"
gprofng collect app -o "$TMP/mt.er" -p on \
  ./build/selfplay_bench "$GAMES_MT" "$SEARCHES" "$PER_EVAL" "$THREADS" "$SEED" \
  > /dev/null 2>&1
{
  header "gprofng, ${THREADS} threads, ${GAMES_MT} games x ${SEARCHES} searches"
  gprofng display text -functions -limit 30 "$TMP/mt.er" 2>/dev/null
} > "$OUTDIR/${LABEL}-mt.txt"

# --- callgrind ---
echo "callgrind (slow)..."
valgrind --tool=callgrind --callgrind-out-file="$TMP/cg.out" \
  ./build/selfplay_bench 3 400 16 1 > /dev/null 2>&1
{
  header "callgrind, instruction counts, 1 thread, 3 games x 400 searches"
  callgrind_annotate --auto=no "$TMP/cg.out" 2>/dev/null | sed -n '18,60p'
} > "$OUTDIR/${LABEL}-callgrind.txt"

echo
echo "wrote:"
ls -la "$OUTDIR/${LABEL}"-*.txt | awk '{print "  "$NF}'
