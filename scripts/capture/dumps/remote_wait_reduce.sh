#!/bin/bash
# on the analysis host: wait for one converted event to finish arriving, then reduce its DTV-band files.
# usage: remote_wait_reduce.sh <event_id>          (event_id is the UTC start, YYYYMMDDhhmmss)
# Site configuration: CHIME/FRB internal values, never committed; export them before running.
#   DUMP_DIR           working directory on the analysis host (holds reduce/)
#   REMOTE_HOME        the operator's home directory on the analysis host, mounted into the container
#   BASEBAND_RAW_ROOT  root of the converted raw baseband archive on the analysis host (read only)
#   USER_DATA_DIR      the operator's user-data directory on the analysis host (products go to pilot_reduce/)
#   BASEBAND_IMAGE     the CHIME/FRB baseband-analysis container image
# (REMOTE_HOME and BASEBAND_IMAGE are read by reduce_driver.sh; they are checked here so a missing one fails before the wait.)
# Reading CHIME baseband data requires CHIME/FRB authorization; see README.md.
: "${DUMP_DIR:?}" "${REMOTE_HOME:?}" "${BASEBAND_RAW_ROOT:?}" "${USER_DATA_DIR:?}" "${BASEBAND_IMAGE:?}"
EV=${1:?event id}
Y=${EV:0:4}; M=${EV:4:2}; D=${EV:6:2}
DIR=$BASEBAND_RAW_ROOT/$Y/$M/$D/baseband_$EV
OUT=$USER_DATA_DIR/pilot_reduce/$EV
LOG=$DUMP_DIR/reduce/wait_$EV.log
mkdir -p "$OUT"
prev=-1; stable=0
# up to 20 h: a 1.53 TB event needs about 1.5 h to convert and 8.5 h to transfer
for i in $(seq 1 1200); do
  n=$(ls "$DIR" 2>/dev/null | wc -l)
  echo "$(date -u +%T) files: $n" >> "$LOG"
  if [ "$n" -ge 630 ] && [ "$n" = "$prev" ]; then
    stable=$((stable+1))
    [ "$stable" -ge 2 ] && break
  else
    stable=0
  fi
  prev=$n
  sleep 60
done
if [ "${n:-0}" -lt 630 ]; then echo "$(date -u +%T) gave up waiting, only $n files" >> "$LOG"; exit 1; fi
echo "$(date -u +%T) transfer complete ($n files); launching driver" >> "$LOG"
"$DUMP_DIR/reduce/reduce_driver.sh" "$DIR" "$OUT" 6 6 0 >> "$LOG" 2>&1
echo "$(date -u +%T) driver finished: $(ls $OUT/*.npz 2>/dev/null | wc -l) products" >> "$LOG"
