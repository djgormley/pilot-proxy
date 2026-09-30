#!/bin/bash
# Fire one approved science dump after a pre-flight, then start its reduction.
# usage: dump_at.sh <label> <planned unix start> <length s> <min TB> <max TB> [--dry-run]
#
# Approved 2026-09-16: three 3 s dumps, buffer cap 2 TB of ours resident, one at a time.
# Gates (all must pass, else postpone 30 min, up to 8 tries):
#   1. our earlier dumps hold at most 3.5 TB on the buffer host (author cap: 5 TB of ours resident, three dumps)
#   2. at least MIN_FREE_GB free on the buffer
#   3. the volume implied by the live frequency count falls in the approved band
# Reads the buffer through a key restricted to one read-only disk report; reads the live
# host count over coco's HTTP status. Never proceeds on a measurement it could not make.
# Site configuration: CHIME/FRB internal values, never committed; export them before running.
#   DUMP_DIR        working directory on the analysis host (holds trigger_dump.py and reduce/)
#   REMOTE_HOME     the operator's home directory on the analysis host, mounted into the container
#   BUFFER_HOST     ssh alias of the baseband buffer host that answers the read-only disk report
#   BUFFER_KEY      path to the ssh key restricted to that one read-only disk report
#   COCO_URL        coco base URL on the CHIME network, http://<host>:<port>
#   CHIME_FRB_USER  the operator's CHIME/FRB workflow account (read by trigger_dump.py)
#   BASEBAND_IMAGE  the CHIME/FRB baseband-analysis container image
# remote_wait_reduce.sh, started for each accepted event, also needs BASEBAND_RAW_ROOT and USER_DATA_DIR.
# Sending a dump writes every live frequency to CHIME's shared baseband buffer. Every dump needs CHIME/FRB
# authorization; never run this without it. See README.md.
: "${DUMP_DIR:?}" "${REMOTE_HOME:?}" "${BUFFER_HOST:?}" "${BUFFER_KEY:?}" "${COCO_URL:?}" "${CHIME_FRB_USER:?}" "${BASEBAND_IMAGE:?}"
LABEL=$1; PLANNED=$2; LEN=$3; MINTB=$4; MAXTB=$5; DRY=${6:-}
MIN_FREE_GB=6200
OURS_MAX_MB=3600000
FREQ_PER_HOST=4
D=$DUMP_DIR
LOG=$D/reduce/schedule_$LABEL.log
KEY=$BUFFER_KEY
say() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }
say "=== dump $LABEL: planned $(date -u -d @$PLANNED '+%F %T UTC'), length ${LEN}s, approved band ${MINTB}-${MAXTB} TB${DRY:+ [DRY RUN]}"

for attempt in $(seq 1 8); do
  now=$(date +%s)
  start=$PLANNED; [ $((start - now)) -lt 120 ] && start=$((now + 300))

  rep=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" "$BUFFER_HOST" 2>/dev/null)
  free_gb=$(echo "$rep" | awk '/^FREE_GB/ {print $2}')
  ours_mb=$(echo "$rep" | awk '/^OURS_MB/ {s+=$2} END {print s+0}')
  hosts=$(curl -s -m 30 "$COCO_URL/baseband-status" | python3 -c "
import sys, json
try: print(json.load(sys.stdin)['baseband'].get('200', 0))
except Exception: print(0)" 2>/dev/null)
  live=$((hosts * FREQ_PER_HOST))
  total_gb=$(python3 -c "print(f'{$LEN * 0.8 * $live:.0f}')")
  total_tb=$(python3 -c "print(f'{$LEN * 0.8 * $live / 1000:.2f}')")

  say "attempt $attempt: free ${free_gb:-?} GB | ours resident ${ours_mb} MB | live ${hosts} hosts = ${live} freqs | total ${total_tb} TB"
  ok=1
  [ -n "$free_gb" ] && [ "${free_gb:-0}" -ge "$MIN_FREE_GB" ] || { say "  BLOCKED: buffer free space unreadable or below ${MIN_FREE_GB} GB"; ok=0; }
  [ "${ours_mb:-99999999}" -le "$OURS_MAX_MB" ] || { say "  BLOCKED: our earlier dumps still hold ${ours_mb} MB on the buffer"; ok=0; }
  [ "${hosts:-0}" -gt 0 ] || { say "  BLOCKED: could not read the live host count"; ok=0; }
  awk "BEGIN {exit !($total_tb >= $MINTB && $total_tb <= $MAXTB)}" || { say "  BLOCKED: ${total_tb} TB is outside the approved ${MINTB}-${MAXTB} TB band"; ok=0; }

  if [ "$ok" = 1 ]; then
    if [ -n "$DRY" ]; then say "DRY RUN: all gates passed; would fire for $(date -u -d @$start '+%F %T UTC') with --confirm-total-tb $total_tb"; exit 0; fi
    say "pre-flight passed; firing for $(date -u -d @$start '+%F %T UTC')"
    out=$(docker run --rm -e HOME="$REMOTE_HOME" -e COCO_URL -e CHIME_FRB_USER -v "$REMOTE_HOME:$REMOTE_HOME" \
      "$BASEBAND_IMAGE" python $D/trigger_dump.py \
      --start "$start" --length "$LEN" --live-freq "$live" --free-gb "$free_gb" \
      --dump --confirm-total-tb "$total_tb" 2>&1)
    echo "$out" >> "$LOG"
    if echo "$out" | grep -q "work deposited: True"; then
      ev=$(date -u -d @$start '+%Y%m%d%H%M%S')
      say "ACCEPTED event $ev; starting the reduction watcher"
      nohup $D/reduce/remote_wait_reduce.sh "$ev" > /dev/null 2>&1 &
      exit 0
    fi
    say "trigger did not confirm acceptance; NOT retrying automatically"
    exit 1
  fi
  say "  postponing 30 min"
  sleep 1800
done
say "gave up after 8 attempts; dump $LABEL NOT sent"
exit 1
