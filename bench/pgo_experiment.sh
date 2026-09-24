#!/usr/bin/env bash
# Profile-guided optimization (PGO) experiment. Hours; unattended.
#
#   setsid nohup ./pgo_experiment.sh <model.mlp> [out_dir] > /dev/null 2>&1 &
#
# Question: does GCC PGO, trained on real-network self-play, make the engine
# faster, and is it bit-identical? Three arms, all built from the same commit:
#
#   base     the normal bench build (-O3 -flto -march=native)
#   pgo      + -fprofile-use          (code the training never ran is
#                                      optimized for size)
#   partial  + -fprofile-use -fprofile-partial-training
#                                     (code the training never ran is
#                                      optimized normally)
#   engine   + -fprofile-use without the network's profile (mlp.gcda
#              removed), so the network's counts do not set the cutoff for
#              "hot" code
#   frac     + -fprofile-use with the hot cutoff lowered
#              (hot-bb-count-fraction, hot-bb-count-ws-permille)
#
# ARMS selects which run (default "base pgo partial"; base is required).
#
# Resumable like long_profile.sh: each step leaves <out>/<step>.done, timing
# rows are appended one per run and skipped on rerun. Everything it needs is
# built or copied into out_dir first, so editing bench/ mid-run changes nothing.
#
# Steps:
#   build    base binaries; instrumented binaries (-fprofile-generate)
#   train    instrumented selfplay_nn, real network, TRAIN_GAMES games,
#            TRAIN_THREADS (1) thread pinned to CPU 0, seed 777 (not a
#            timing or gate seed)
#   pgo      rebuild with the profile, every PGO arm; inspect.txt records
#            the size of TrainMC::doIteration and the calls left in it, and
#            inline-<arm>.txt GCC's missed-inlining report from the LTO link
#   gates    golden digests and verify for every arm; SAMPLE_DIGEST=1
#            selfplay_nn on seeds 1-4 (300 games, 1600 searches, 20 threads)
#            for every arm, compared with the recorded baselines
#   mt       timing: real network, 20 threads, G_MT games, REPS_MT seeds;
#            the three arms back to back per seed, order rotated per seed
#   st       timing: stub (selfplay_bench), 1 thread pinned to CPU 0 (a
#            P-core), G_ST games, REPS_ST seeds, same rotation
#   analyze  pgo_analyze.py -> summary.txt
#
# CPU care: nice 10; before each run waits until the package is below COOL_C
# (default 85) degrees, for at most 10 minutes; logs temperature and clock
# every minute to temps.tsv.

set -u
MODEL_SRC=$(realpath -e "${1:?usage: pgo_experiment.sh <model.mlp> [out_dir]}") || exit 1
BENCH=$(cd "$(dirname "$0")" && pwd)
OUT=${2:-$BENCH/results/pgo-$(date +%F)}
COOL_C=${COOL_C:-85}
# Sizes; override for a quick smoke test of the whole pipeline
TRAIN_GAMES=${TRAIN_GAMES:-300}
TRAIN_THREADS=${TRAIN_THREADS:-1}
GATE_GAMES=${GATE_GAMES:-300}
G_MT=${G_MT:-1000}
REPS_MT=${REPS_MT:-40}
G_ST=${G_ST:-200}
REPS_ST=${REPS_ST:-40}
ARMS=${ARMS:-base pgo partial}
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

[ -f "$OUT/temps.tsv" ] || echo -e "time\tpkg_c\tmean_mhz\tload1" > "$OUT/temps.tsv"
( while true; do
    echo -e "$(date '+%F %T')\t$(pkg_temp_c)\t$(mean_mhz)\t$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT/temps.tsv"
    sleep 60
  done ) &
TEMP_PID=$!
trap 'kill $TEMP_PID 2>/dev/null' EXIT

step() {  # step <name> <function>
  if [ -f "$OUT/$1.done" ]; then log "skip $1 (done)"; return 0; fi
  log "start $1"
  if "$2"; then touch "$OUT/$1.done"; log "done $1"; return 0; fi
  log "FAILED $1"
  return 1
}

M=$OUT/model.mlp
PROF=$OUT/pgo-data
# The bench Makefile's own flags; PGO flags are appended to these
BASE_CXX=$(make -s -C "$BENCH" -p 2>/dev/null | awk -F' := ' '/^CXXFLAGS :=/ {print $2; exit}')
BASE_LD=$(make -s -C "$BENCH" -p 2>/dev/null | awk -F' := ' '/^LDFLAGS :=/ {print $2; exit}')
TARGETS="selfplay_nn selfplay_bench golden verify"

# build_arm <arm> <extra flags> [link-only flags]: builds TARGETS into bin/<arm>. All arms use
# the same object directory, because GCC names each .gcda file after the
# object's full path: the -fprofile-use build must write its objects where
# the -fprofile-generate build wrote them.
build_arm() {
  local arm=$1 extra=$2 link=${3:-} b=$OUT/objs
  rm -rf "$b"
  local t targets=""
  for t in $TARGETS; do targets="$targets $b/$t"; done
  make -C "$BENCH" BUILD="$b" CXXFLAGS="$BASE_CXX $extra" LDFLAGS="$BASE_LD $extra $link" \
    $targets > "$OUT/build-$arm.log" 2>&1 || return 1
  mkdir -p "$OUT/bin/$arm"
  for t in $TARGETS; do cp "$b/$t" "$OUT/bin/$arm/"; done
  rm -rf "$b"
}

do_build() {
  {
    echo "date:      $(date -Is)"
    echo "commit:    $(git -C "$BENCH" rev-parse HEAD)"
    echo "dirty:     $(git -C "$BENCH" diff --quiet -- . ../corintho_ai/cpp && echo no || echo yes)"
    echo "compiler:  $(g++ --version | head -1)"
    echo "cxxflags:  $BASE_CXX"
    echo "ldflags:   $BASE_LD"
    echo "cpu:       $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
    echo "governor:  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "power:     $(cat /sys/class/power_supply/*/online 2>/dev/null | tr '\n' ' ')"
    echo "model:     $MODEL_SRC ($(sha256sum "$MODEL_SRC" | cut -c1-16))"
    echo "train:     $TRAIN_GAMES games, $TRAIN_THREADS threads, seed 777"
    echo "arms:      $ARMS"
    echo "sizes:     gate $GATE_GAMES, mt $REPS_MT x $G_MT, st $REPS_ST x $G_ST"
  } > "$OUT/provenance.txt"
  printf '%s\n' 'bin/' 'objs/' 'model.mlp' 'run.*' 'pgo-data*/' 'inline-*.txt' > "$OUT/.gitignore"
  cp "$MODEL_SRC" "$M" || return 1
  build_arm base "" || return 1
  # Training runs single-threaded with plain counters. Multi-threaded
  # training needs atomic counters (-fprofile-update=prefer-atomic), and 20
  # threads incrementing the same counters made the run several hundred times
  # slower (40 games: >10 minutes instead of ~2 s). One thread takes the same
  # branches in the same proportions, since every game runs the same code.
  rm -rf "$PROF"
  build_arm gen "-fprofile-generate=$PROF -fprofile-update=single"
}

do_train() {
  rm -rf "$PROF"
  cooldown
  taskset -c 0 "$OUT/bin/gen/selfplay_nn" - "$M" "$TRAIN_GAMES" 1600 16 "$TRAIN_THREADS" 777 > "$OUT/train.log" 2>&1 || return 1
  local n; n=$(find "$PROF" -name '*.gcda' | wc -l)
  log "train: $n .gcda files, $(grep -E 'engine_seconds|wall_seconds' "$OUT/train.log" | awk '{printf "%s=%s ", $2, $3}')"
  [ "$n" -gt 0 ]
}

arm_flags() {  # arm_flags <arm>: compiler flags for a PGO arm
  case $1 in
    pgo)     echo "-fprofile-use=$PROF" ;;
    partial) echo "-fprofile-use=$PROF -fprofile-partial-training" ;;
    engine)  echo "-fprofile-use=$OUT/pgo-data-engine" ;;
    frac)    echo "-fprofile-use=$PROF --param=hot-bb-count-fraction=1000000 --param=hot-bb-count-ws-permille=999" ;;
    *)       return 1 ;;
  esac
}

# Size of TrainMC::doIteration and the calls it still makes: inlining lost
# on the hot path shows up here (entry 27)
inspect() {  # inspect <arm>
  objdump -d --no-show-raw-insn -C "$OUT/bin/$1/selfplay_nn" | awk -v A="$1" '
    /^[0-9a-f]+ <.*>:$/ { f = ($0 ~ /<TrainMC::doIteration\(float\*, float\*\)>:$/) }
    f && /^ *[0-9a-f]+:/ { n++ }
    f && /call/ { sub(/.*call +[0-9a-f]+ /, ""); c[$0]++; calls++ }
    END { printf "== %s: TrainMC::doIteration %d instructions, %d call sites\n", A, n, calls
          for (k in c) printf "  %3d %s\n", c[k], k }'
}

do_pgo() {
  # The engine arm's profile: everything but the network's
  rm -rf "$OUT/pgo-data-engine"
  cp -r "$PROF" "$OUT/pgo-data-engine"
  find "$OUT/pgo-data-engine" -name 'mlp.gcda' -delete
  local arm
  for arm in $ARMS; do
    [ "$arm" = base ] && continue
    build_arm "$arm" "$(arm_flags "$arm") -Wno-missing-profile" \
      "-fopt-info-inline-missed=$OUT/inline-$arm.txt" || return 1
    # A profile that silently failed to apply would make the arm a copy of
    # base. Identical binaries are the tell.
    if cmp -s "$OUT/bin/base/selfplay_nn" "$OUT/bin/$arm/selfplay_nn"; then
      log "$arm binary identical to base: profile not applied"
      return 1
    fi
  done
  ls -l "$OUT"/bin/*/selfplay_nn | awk '{print $5, $9}' > "$OUT/binary-sizes.txt"
  for arm in $ARMS; do inspect "$arm"; done > "$OUT/inspect.txt"
}

# Recorded at 5a5616e and unchanged through 8f20319 (worklog entries 25-26)
declare -A BASELINE=([1]=2147e3071a27cc4d [2]=a988f67a0c984be7 [3]=dcb1745df12a1ba0 [4]=e0db31d1d69b6456)

do_gates() {
  local f=$OUT/gates.txt ok=0
  : > "$f"
  for arm in $ARMS; do
    "$OUT/bin/$arm/golden" 2>&1 | grep '#METRIC digest' | sed "s/^/$arm /" >> "$f"
    "$OUT/bin/$arm/verify" > "$OUT/verify-$arm.log" 2>&1 \
      && echo "$arm verify PASS" >> "$f" || { echo "$arm verify FAIL" >> "$f"; ok=1; }
  done
  for seed in 1 2 3 4; do
    for arm in $ARMS; do
      cooldown
      local d
      d=$(SAMPLE_DIGEST=1 "$OUT/bin/$arm/selfplay_nn" - "$M" "$GATE_GAMES" 1600 16 20 "$seed" 2>&1 \
          | awk '/#METRIC sample_digest/ {d=$3} /#METRIC turns/ {t=$3} END {print d, t}')
      local want=${BASELINE[$seed]}
      local verdict=MATCH
      # Baselines are for 300 games; other sizes compare across arms only
      [ "$GATE_GAMES" = 300 ] && [ "${d%% *}" != "$want" ] && { verdict=DIFF; ok=1; }
      echo "$arm sample seed $seed $d $verdict" >> "$f"
      log "gate $arm seed $seed $d $verdict"
    done
  done
  # Across arms: golden lines and sample digests must agree with base
  local arm
  for arm in $ARMS; do
    [ "$arm" = base ] && continue
    if ! diff <(grep "^base " "$f" | cut -d' ' -f2-) <(grep "^$arm " "$f" | cut -d' ' -f2-) > /dev/null; then
      echo "$arm DIFFERS FROM base" >> "$f"; ok=1
    fi
  done
  [ $ok = 0 ] && echo "ALL GATES PASS" >> "$f" || echo "GATE FAILURE" >> "$f"
  return 0   # record and continue: timing is still informative, analysis flags it
}

# rotate <rep>: arm order for this rep, rotated so no arm always goes first
rotate() {
  local a=($ARMS) n=${#a[@]} i out=""
  for i in $(seq 0 $((n - 1))); do out="$out ${a[$(( (i + $1) % n ))]}"; done
  echo $out
}

TIMING=$OUT/timing.tsv
[ -f "$TIMING" ] || echo -e "mode\tarm\tseed\tengine_s\twall_s\tturns\tgather_s\tpkg_c_before" > "$TIMING"

# One timed run; prints "engine wall turns gather"
run_timed() {  # run_timed <mode> <arm> <seed>
  local mode=$1 arm=$2 seed=$3 tmp
  tmp=$(mktemp "$OUT/run.XXXX")
  if [ "$mode" = mt ]; then
    "$OUT/bin/$arm/selfplay_nn" - "$M" "$G_MT" 1600 16 20 "$seed" > "$tmp" 2>&1
  else
    taskset -c 0 "$OUT/bin/$arm/selfplay_bench" "$G_ST" 1600 16 1 "$seed" > "$tmp" 2>&1
  fi
  awk '/#METRIC engine_seconds/ {e=$3} /#METRIC wall_seconds/ {w=$3}
       /#METRIC turns/ {t=$3} /#METRIC gather_seconds/ {g=$3}
       END {print e, w, t, (g == "" ? "NA" : g)}' "$tmp"
  rm -f "$tmp"
}

timing() {  # timing <mode> <reps> <seed0>
  local mode=$1 reps=$2 seed0=$3 rep
  for rep in $(seq 1 "$reps"); do
    local seed=$((seed0 + rep)) arm
    for arm in $(rotate "$rep"); do
      grep -q "^$mode"$'\t'"$arm"$'\t'"$seed"$'\t' "$TIMING" && continue
      cooldown
      local c r
      c=$(pkg_temp_c)
      r=$(run_timed "$mode" "$arm" "$seed")
      echo -e "$mode\t$arm\t$seed\t$(echo "$r" | tr ' ' '\t')\t$c" >> "$TIMING"
      log "$mode $arm seed $seed: $r"
    done
  done
}

do_mt() { timing mt "$REPS_MT" 3000; }
do_st() { timing st "$REPS_ST" 4000; }
do_analyze() { python3 "$BENCH/pgo_analyze.py" "$OUT" > "$OUT/summary.txt" 2>&1; }

log "pgo_experiment start, out=$OUT, pkg $(pkg_temp_c)C"
step build do_build || exit 1
step train do_train || exit 1
step pgo do_pgo || exit 1
step gates do_gates
step mt do_mt
step st do_st
step analyze do_analyze
log "pgo_experiment finished"
touch "$OUT/ALL.done"
