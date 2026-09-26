#!/usr/bin/env bash
# Reruns $SCRIPT (default arch/overnight.sh; also arch/loop.sh) until it
# finishes, up to 5 failures in a row (each rerun resumes from the last
# finished step). Logs to runs/$NAME/.
cd "$(dirname "$0")/.."
NAME=${NAME:-night-1}
SCRIPT=${SCRIPT:-arch/overnight.sh}
mkdir -p "runs/$NAME"
# One run at a time: a second copy would write the same files
exec 9> "runs/$NAME/lock"
flock -n 9 || { echo "runs/$NAME is already running" >&2; exit 1; }
fails=0
while true; do
  if "$SCRIPT" >> "runs/$NAME/overnight.out" 2>&1; then
    echo "$(date '+%F %T') supervisor: finished" >> "runs/$NAME/progress.log"
    exit 0
  fi
  fails=$((fails + 1))
  echo "$(date '+%F %T') supervisor: $SCRIPT failed ($fails in a row); see overnight.out and the step's log" >> "runs/$NAME/progress.log"
  [ "$fails" -ge 5 ] && exit 1
  sleep 60
done
