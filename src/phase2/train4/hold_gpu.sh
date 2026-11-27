#!/bin/bash
# Hold ~/gpu.lock for train2. Trap removes it on EXIT/INT/TERM only if it is still ours.
# Polls; never kills another holder. Hard stop 4:45 PM PST.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$HOME/gpu.lock"
DONE="$HERE/.gpu-hold.done"
PIDF="$HERE/.gpu-hold.pid"
LOG="$HERE/logs/gpu-hold.log"
mkdir -p "$HERE/logs"
exec >>"$LOG" 2>&1

is_ours() {
  [ -f "$LOCK" ] || return 1
  awk -v p="$$" '$1=="output-model" && $2==p {found=1} END{exit !found}' "$LOCK"
}

release() {
  if is_ours; then
    rm -f "$LOCK"
    echo "$(TZ=America/Los_Angeles date '+%a %b %d %I:%M:%S %p %Z')  released gpu.lock pid $$"
  fi
  rm -f "$PIDF"
}

# Wait if another live PID holds the lock. Never kill it.
while [ -f "$LOCK" ]; do
  opid=$(awk '{print $2}' "$LOCK")
  if [ -n "${opid:-}" ] && kill -0 "$opid" 2>/dev/null; then
    echo "$(TZ=America/Los_Angeles date '+%a %b %d %I:%M:%S %p %Z')  waiting for lock: $(cat "$LOCK")"
    sleep 60
  else
    echo "$(TZ=America/Los_Angeles date '+%a %b %d %I:%M:%S %p %Z')  stale lock, removing: $(cat "$LOCK")"
    rm -f "$LOCK"
    break
  fi
done

echo "output-model $$ $(TZ=America/Los_Angeles date '+%a %b %d %I:%M:%S %p %Z') train2 voice-rewrite" > "$LOCK"
echo $$ > "$PIDF"
trap release EXIT INT TERM
echo "$(TZ=America/Los_Angeles date '+%a %b %d %I:%M:%S %p %Z')  holding gpu.lock pid $$"
echo "HOLDING $$ $(cat "$LOCK")"

while [ ! -f "$DONE" ]; do
  now=$(TZ=America/Los_Angeles date +%H%M)
  if [ "$now" -ge 1645 ]; then
    echo "$(TZ=America/Los_Angeles date '+%a %b %d %I:%M:%S %p %Z')  hard stop 4:45 PM PST"
    exit 0
  fi
  if ! is_ours; then
    echo "$(TZ=America/Los_Angeles date '+%a %b %d %I:%M:%S %p %Z')  lost lock: $(cat "$LOCK" 2>/dev/null || echo missing)"
    exit 1
  fi
  sleep 20
done
echo "$(TZ=America/Los_Angeles date '+%a %b %d %I:%M:%S %p %Z')  done file seen"
exit 0
