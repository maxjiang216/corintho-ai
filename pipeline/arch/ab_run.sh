#!/bin/bash
# One command for alpha-beta self-play runs (worklog entry 25).
#
#   arch/ab_run.sh start     build, stop any running loop, start ab_loop.sh in
#                            the background (survives closing the terminal)
#   arch/ab_run.sh stop      stop the loop and everything it started
#   arch/ab_run.sh status    schedule, matches, the latest progress line
#   arch/ab_run.sh log       follow progress.log
#
# Defaults (override in the environment, e.g. RUN=runs/ab-2 arch/ab_run.sh
# start): growing generations from 400k positions and 40k steps, 400-game
# matches against MCTS every 5 generations, 200 generations, old positions
# deleted, a progress line every 60 s. Once $RUN/schedule exists, it
# decides positions and steps (see ab_loop.sh).
cd "$(dirname "$0")/.."
export RUN=${RUN:-runs/ab-1}
export ADAPT=${ADAPT:-1}
export POSITIONS=${POSITIONS:-400000}
export STEPS=${STEPS:-40000}
export MATCH_GAMES=${MATCH_GAMES:-400}
export GENS=${GENS:-200}
export KEEP_DATA=${KEEP_DATA:-0}
export REPORT=${REPORT:-60}
PIDFILE=$RUN/loop.pid

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

stop() {
  if running; then
    # The loop runs in its own process group: stop it and its children
    kill -- "-$(cat "$PIDFILE")" 2>/dev/null
    sleep 2
    echo "stopped $RUN"
  else
    echo "no loop running for $RUN"
  fi
  rm -f "$PIDFILE"
  # Children of a loop started some other way (an older nohup)
  pkill -f "ab_loop.sh" 2>/dev/null
  pkill -f "ab_selfplay .*$RUN/" 2>/dev/null
  pkill -f "ab_match .*$RUN/" 2>/dev/null
  pkill -f "nnue_distill.py .*$RUN/" 2>/dev/null
  true
}

status() {
  echo "run: $RUN ($(running && echo "running, pid $(cat "$PIDFILE")" || echo "not running"))"
  [ -f "$RUN/schedule" ] && read -r p s b st < "$RUN/schedule" &&
    echo "schedule: $p positions, $s steps per generation, best $b, stalls $st"
  echo "matches against MCTS:"
  grep -h "MCTS:" "$RUN/progress.log" 2>/dev/null | sed 's/^/  /'
  grep -h "stalled" "$RUN/progress.log" 2>/dev/null | sed 's/^/  /'
  echo "latest:"
  tail -n 2 "$RUN/progress.log" 2>/dev/null | sed 's/^/  /'
}

case "$1" in
  start)
    make -s build/ab_selfplay build/ab_match 2>&1 | grep -E "error" && exit 1
    stop >/dev/null
    mkdir -p "$RUN"
    setsid nohup arch/ab_loop.sh >> "$RUN/loop.out" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 5
    if running; then
      echo "started $RUN (pid $(cat "$PIDFILE")); output in $RUN/loop.out"
      status
    else
      echo "the loop exited at once; see $RUN/loop.out:"
      tail -n 20 "$RUN/loop.out"
      exit 1
    fi
    ;;
  stop) stop ;;
  status) status ;;
  log) tail -f "$RUN/progress.log" ;;
  *)
    sed -n 2,9p "$0"
    exit 1
    ;;
esac
