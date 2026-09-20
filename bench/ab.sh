#!/usr/bin/env bash
# Interleaved A/B comparison of two builds.
#
#   ./ab.sh <labelA> "<flagsA>" <labelB> "<flagsB>" [reps]
#
# run_suite.sh records one build at a time, which is fine when the machine is
# steady. It is not fine here: an identical build measured 77% slower forty
# minutes later, because the laptop had moved from AC to battery. Any drift
# between recording A and recording B lands entirely in the comparison.
#
# This builds both once into separate directories, then ALTERNATES runs
# A,B,A,B,... so that slow drift affects both arms equally. What remains is the
# difference between the builds.
#
# It reports the median of each arm and the spread, and warns when the arms'
# own spreads are wide enough that the comparison is not trustworthy.

set -euo pipefail
cd "$(dirname "$0")"

LABEL_A="${1:?usage: ./ab.sh <labelA> \"<flagsA>\" <labelB> \"<flagsB>\" [reps]}"
FLAGS_A="${2:?}"
LABEL_B="${3:?}"
FLAGS_B="${4:?}"
REPS="${5:-7}"

GAMES=50
SEARCHES=1600
PER_EVAL=16
SEED=12345

echo "building both arms..."
rm -rf build-a build-b
make -s all BUILD=build-a CXXFLAGS="$FLAGS_A" LDFLAGS="$FLAGS_A" > /dev/null
make -s all BUILD=build-b CXXFLAGS="$FLAGS_B" LDFLAGS="$FLAGS_B" > /dev/null

echo "machine: ac=$(cat /sys/class/power_supply/AC*/online 2>/dev/null | head -1)" \
     "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
     "load=$(cut -d' ' -f1 /proc/loadavg)"
echo "interleaving ${REPS} reps per arm, single-threaded..."

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
for _ in $(seq "$REPS"); do
  ./build-a/selfplay_bench "$GAMES" "$SEARCHES" "$PER_EVAL" 1 "$SEED" \
    | awk '/#METRIC engine_seconds/{print $3}' >> "$TMP/a"
  ./build-b/selfplay_bench "$GAMES" "$SEARCHES" "$PER_EVAL" 1 "$SEED" \
    | awk '/#METRIC engine_seconds/{print $3}' >> "$TMP/b"
  printf "."
done
echo

python3 - "$TMP/a" "$TMP/b" "$LABEL_A" "$LABEL_B" <<'PY'
import sys, statistics
a = [float(x) for x in open(sys.argv[1])]
b = [float(x) for x in open(sys.argv[2])]
la, lb = sys.argv[3], sys.argv[4]
for label, xs in ((la, a), (lb, b)):
    med = statistics.median(xs)
    print(f"  {label:<24} median {med:7.3f}s   min {min(xs):7.3f}  "
          f"max {max(xs):7.3f}  spread {(max(xs)-min(xs))/med*100:5.1f}%")
ma, mb = statistics.median(a), statistics.median(b)
delta = (mb - ma) / ma * 100.0
worst = max((max(a)-min(a))/ma, (max(b)-min(b))/mb) * 100.0
print(f"\n  {lb} vs {la}: {delta:+.1f}%")
if abs(delta) < worst:
    print(f"  WARNING: delta is smaller than the worst arm's own spread "
          f"({worst:.1f}%). Not trustworthy -- quiet the machine or raise reps.")
else:
    print(f"  Delta exceeds the worst arm spread ({worst:.1f}%), so it is real.")
PY
