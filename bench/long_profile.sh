#!/usr/bin/env bash
# Long, realistic engine profile with the real network. Hours; unattended.
#
#   setsid nohup ./long_profile.sh <model.mlp> [out_dir] > /dev/null 2>&1 &
#
# Resumable: each step leaves <out>/<step>.done, and rerunning the script with
# the same out_dir skips finished steps. Everything it needs (binaries, model)
# is copied into out_dir first, so rebuilding bench/ mid-run changes nothing.
#
# Steps:
#   build      private copies of the benchmark binaries and the model
#   scaling    the same G_SCALE (1000) games at 1..20 threads, twice
#   gprof_mt   gprofng, RUNS_MT (8) x G_MT (2000) games x 20 threads
#   gprof_st   gprofng, 1 thread pinned to a P-core, G_ST (200) games
#   reports    gprofng text reports (function level), merged across runs
#   callgrind  callgrind with cache and branch simulation, engine only,
#              G_CG (48) games, 1 thread; dumped every 30 min so partial
#              results persist
#
# CPU care: runs at nice 10; waits before each run until the package is below
# COOL_C (default 85) degrees, for at most 10 minutes; logs package temperature
# and mean clock every minute to temps.tsv so throttled runs can be spotted.
# The CPU's own protection throttles at 100 C regardless.

set -u
MODEL_SRC=${1:?usage: long_profile.sh <model.mlp> [out_dir]}
BENCH=$(cd "$(dirname "$0")" && pwd)
OUT=${2:-$BENCH/results/long-profile-$(date +%F)}
COOL_C=${COOL_C:-85}
# Sizes; override for a quick smoke test of the whole pipeline
G_SCALE=${G_SCALE:-1000}
G_MT=${G_MT:-2000}
RUNS_MT=${RUNS_MT:-8}
G_ST=${G_ST:-200}
G_CG=${G_CG:-48}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$OUT"
renice -n 10 $$ > /dev/null

log() { echo "$(date '+%F %T') $*" >> "$OUT/progress.log"; }

pkg_zone=""
for z in /sys/class/thermal/thermal_zone*; do
  [ "$(cat "$z/type" 2>/dev/null)" = "x86_pkg_temp" ] && pkg_zone=$z
done
pkg_temp_c() {
  if [ -n "$pkg_zone" ]; then echo $(( $(cat "$pkg_zone/temp") / 1000 )); else echo 0; fi
}
mean_mhz() { awk '/cpu MHz/ {s += $4; n++} END {printf "%d", s / n}' /proc/cpuinfo; }

cooldown() {
  local waited=0
  while [ "$(pkg_temp_c)" -ge "$COOL_C" ] && [ $waited -lt 600 ]; do
    sleep 15
    waited=$((waited + 15))
  done
  [ $waited -gt 0 ] && log "cooldown waited ${waited}s, now $(pkg_temp_c)C"
}

# Temperature logger for the life of this script
[ -f "$OUT/temps.tsv" ] || echo -e "time\tpkg_c\tmean_mhz\tload1" > "$OUT/temps.tsv"
( while true; do
    echo -e "$(date '+%F %T')\t$(pkg_temp_c)\t$(mean_mhz)\t$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT/temps.tsv"
    sleep 60
  done ) &
TEMP_PID=$!
trap 'kill $TEMP_PID 2>/dev/null' EXIT

step() {  # step <name> <function>
  if [ -f "$OUT/$1.done" ]; then log "skip $1 (done)"; return; fi
  log "start $1"
  if "$2"; then touch "$OUT/$1.done"; log "done $1"; else log "FAILED $1"; fi
}

BIN=$OUT/bin
M=$OUT/model.mlp

do_build() {
  {
    echo "date:     $(date -Is)"
    echo "commit:   $(git -C "$BENCH" rev-parse HEAD)"
    echo "dirty:    $(git -C "$BENCH" diff --quiet && echo no || echo yes)"
    echo "cxxflags: $(make -s -C "$BENCH" -p 2>/dev/null | awk -F' := ' '/^CXXFLAGS :=/ {print $2; exit}')"
    echo "cpu:      $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "power:    $(cat /sys/class/power_supply/*/online 2>/dev/null | tr '\n' ' ')"
    echo "model:    $MODEL_SRC ($(sha256sum "$MODEL_SRC" | cut -c1-16))"
  } > "$OUT/provenance.txt"
  printf '%s\n' '*.er/' '*.er.ok' 'bin/' 'model.mlp' 'run.*' > "$OUT/.gitignore"
  make -s -C "$BENCH" BUILD="$OUT/build" "$OUT/build/selfplay_nn" > "$OUT/build.log" 2>&1 || return 1
  mkdir -p "$BIN" && cp "$OUT/build/selfplay_nn" "$BIN/" && rm -rf "$OUT/build"
  cp "$MODEL_SRC" "$M"
}

# One selfplay_nn run under /usr/bin/time; prints "engine wall requests turns peak_mb gather"
run_nn() {  # run_nn <games> <threads> <seed> [prefix command...]
  local games=$1 threads=$2 seed=$3; shift 3
  local tmp; tmp=$(mktemp "$OUT/run.XXXX")
  /usr/bin/time -f "#METRIC peak_kb %M" "$@" "$BIN/selfplay_nn" - "$M" "$games" 1600 16 "$threads" "$seed" > "$tmp" 2>&1
  awk '/#METRIC engine_seconds/ {e=$3} /#METRIC wall_seconds/ {w=$3}
       /#METRIC requests/ {r=$3} /#METRIC turns/ {t=$3} /#METRIC peak_kb/ {p=int($3/1024)}
       /#METRIC gather_seconds/ {g=$3}
       END {print e, w, r, t, p, g}' "$tmp"
  rm -f "$tmp"
}

do_scaling() {
  local f=$OUT/scaling.tsv
  [ -f "$f" ] || echo -e "rep\tthreads\tengine_s\twall_s\trequests\tturns\tpeak_mb\tgather_s\tpkg_c_before" > "$f"
  # Rep 1 descending, rep 2 ascending, so slow drift does not favour one end
  for rep in 1 2; do
    local order="20 14 12 8 6 4 2 1"
    [ $rep = 2 ] && order="1 2 4 6 8 12 14 20"
    for t in $order; do
      grep -q "^$rep"$'\t'"$t"$'\t' "$f" && continue   # resume inside the step
      cooldown
      local c; c=$(pkg_temp_c)
      local r; r=$(run_nn "$G_SCALE" "$t" 901)
      echo -e "$rep\t$t\t$(echo "$r" | tr ' ' '\t')\t$c" >> "$f"
      log "scaling rep $rep threads $t: $r"
    done
  done
}

do_gprof_mt() {
  for seed in $(seq 1 "$RUNS_MT"); do
    local exp=$OUT/gp-mt-$seed.er
    [ -f "$exp.ok" ] && continue
    rm -rf "$exp"
    cooldown
    gprofng collect app -p hi -o "$exp" "$BIN/selfplay_nn" - "$M" "$G_MT" 1600 16 20 "$seed" \
      > "$OUT/gp-mt-$seed.log" 2>&1 || return 1
    touch "$exp.ok"
    log "gprof_mt seed $seed: $(grep -E 'engine_seconds|wall_seconds' "$OUT/gp-mt-$seed.log" | awk '{printf "%s=%s ", $2, $3}')"
  done
}

do_gprof_st() {
  local exp=$OUT/gp-st.er
  rm -rf "$exp"
  cooldown
  taskset -c 0 gprofng collect app -p hi -o "$exp" "$BIN/selfplay_nn" - "$M" "$G_ST" 1600 16 1 1 \
    > "$OUT/gp-st.log" 2>&1
}

do_reports() {
  local mt; mt=$(ls -d "$OUT"/gp-mt-*.er)
  local r=$OUT/reports
  mkdir -p "$r"
  gprofng display text -limit 80 -functions $mt > "$r/mt-functions.txt" 2>&1
  gprofng display text -limit 200 -lines $mt > "$r/mt-lines.txt" 2>&1
  gprofng display text -limit 60 -calltree $mt > "$r/mt-calltree.txt" 2>&1
  # No main-thread report: the main thread also runs a share of every
  # parallel loop, so it does not isolate the serial work. gather_seconds in
  # the run output does.
  gprofng display text -limit 80 -functions "$OUT/gp-st.er" > "$r/st-functions.txt" 2>&1
  gprofng display text -limit 200 -lines "$OUT/gp-st.er" > "$r/st-lines.txt" 2>&1
  # No line-level source views: gprofng 2.42 records no line numbers for this
  # LTO build ("Source location not recorded"). Callgrind supplies lines.
  return 0
}

do_callgrind() {
  local cg=$OUT/callgrind.out
  cooldown
  valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes --collect-atstart=no \
    --toggle-collect='Trainer::doIteration*' --callgrind-out-file="$cg" \
    "$BIN/selfplay_nn" - "$M" "$G_CG" 1600 16 1 901 > "$OUT/callgrind.log" 2>&1 &
  local pid=$!
  # Periodic dumps: each writes callgrind.out.<n> with the counts so far
  ( while kill -0 $pid 2>/dev/null; do
      sleep 1800
      callgrind_control -d $pid > /dev/null 2>&1 && log "callgrind checkpoint dump"
    done ) &
  local dumper=$!
  wait $pid
  local rc=$?
  kill $dumper 2>/dev/null
  mkdir -p "$OUT/reports"
  # Each checkpoint dump zeroes the counters, and later dumps refer back to
  # names defined in earlier ones, so callgrind_annotate cannot read them one
  # at a time. callgrind_merge.py sums them in dump order.
  local dumps; dumps=$(ls "$cg".* 2>/dev/null | sort -t. -k3 -n)
  python3 "$BENCH/callgrind_merge.py" $dumps "$cg" > "$OUT/reports/callgrind-merged.txt" 2>&1
  return $rc
}

log "long_profile start, out=$OUT, pkg $(pkg_temp_c)C"
step build do_build
step scaling do_scaling
step gprof_mt do_gprof_mt
step gprof_st do_gprof_st
step reports do_reports
step callgrind do_callgrind
log "long_profile finished"
touch "$OUT/ALL.done"
