#!/bin/bash
# Cadence campaign: N short dumps at fixed offsets from a common start, to measure the residual's coherence at
# lags between 15 s and 20 min. usage: cadence_fire.sh <unix start of the first dump> <length s>
# Approved by the author 2026-09-17 ("do whatever we need"). Pre-flights once (space, live count), then posts every
# dump ahead of time; each accepted event gets its own reduction waiter.
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
T0=$1; LEN=$2
OFFSETS="0 15 30 60 120 240 480 720 960 1200"
D=$DUMP_DIR; LOG=$D/reduce/cadence_$(date -u +%Y%m%dT%H%M%S).log
say() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
rep=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -i "$BUFFER_KEY" "$BUFFER_HOST" 2>/dev/null)
free_gb=$(echo "$rep" | awk '/^FREE_GB/ {print $2}'); ours_mb=$(echo "$rep" | awk '/^OURS_MB/ {s+=$2} END {print s+0}')
hosts=$(curl -s -m 30 "$COCO_URL/baseband-status" | python3 -c "import sys,json; print(json.load(sys.stdin)['baseband'].get('200',0))" 2>/dev/null)
live=$((hosts * 4)); per_tb=$(python3 -c "print(f'{$LEN*0.8*$live/1000:.2f}')"); n=$(echo $OFFSETS | wc -w)
say "pre-flight: free ${free_gb} GB, ours ${ours_mb} MB, live ${live} freqs; ${n} dumps of ${LEN} s = ${per_tb} TB each"
[ "${free_gb:-0}" -ge 6200 ] || { say "BLOCKED: free space"; exit 1; }
[ "${live:-0}" -ge 600 ] || { say "BLOCKED: live count ${live}"; exit 1; }
python3 -c "import sys; sys.exit(0 if $n*$per_tb <= 2.0 else 1)" || { say "BLOCKED: campaign total over 2 TB"; exit 1; }
for off in $OFFSETS; do
  start=$((T0 + off))
  out=$(docker run --rm -e HOME="$REMOTE_HOME" -e COCO_URL -e CHIME_FRB_USER -v "$REMOTE_HOME:$REMOTE_HOME" "$BASEBAND_IMAGE" \
        python $D/trigger_dump.py --start $start --length $LEN --live-freq $live --free-gb $free_gb --dump --confirm-total-tb $per_tb 2>&1)
  ev=$(echo "$out" | grep -oE "event_id [0-9]{14}" | head -1 | awk '{print $2}')
  if echo "$out" | grep -q "work deposited: True"; then say "offset ${off}s: ACCEPTED event $ev"; nohup $D/reduce/remote_wait_reduce.sh "$ev" > /dev/null 2>&1 &
  else say "offset ${off}s: NOT accepted: $(echo "$out" | tail -2 | tr '\n' ' ')"; fi
done
say "posted; events listed above"
