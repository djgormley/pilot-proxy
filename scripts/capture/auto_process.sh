#!/bin/bash
# Pull and process every reduced event as it completes on the analysis host (280 products), once each.
# Site configuration: CHIME/FRB internal values, never committed; export them before running.
#   JUMP_HOST      ssh jump host into the CHIME network
#   ANALYSIS_HOST  ssh destination of the analysis host, <user>@<host>
#   USER_DATA_DIR  the operator's user-data directory on the analysis host (products are in pilot_reduce/)
# process_dump.sh, run for each completed event, reads the same three and its local configuration
# (REDUCE_DIR, DATASETS_DIR, ARCHIVE_PRODUCTS, PYTHON and optionally BACKUP_DIR); the ledger and the log are kept in
# REDUCE_DIR.
# Reading CHIME baseband data requires CHIME/FRB authorization; see dumps/README.md.
: "${JUMP_HOST:?}" "${ANALYSIS_HOST:?}" "${USER_DATA_DIR:?}" "${REDUCE_DIR:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
DONE=$REDUCE_DIR/done_events.txt; LOG=$REDUCE_DIR/auto_process.log
touch $DONE
for i in $(seq 1 300); do
  out=$(ssh -o ConnectTimeout=20 -J "$JUMP_HOST" "$ANALYSIS_HOST" 'for d in '"$USER_DATA_DIR"'/pilot_reduce/2026091*; do n=$(ls $d/*.npz 2>/dev/null | wc -l); [ "$n" -eq 280 ] && echo "$(basename $d)"; done' 2>/dev/null | grep -v conda)
  for ev in $out; do
    grep -q "^$ev$" $DONE && continue
    echo "$(date -u +%FT%TZ) processing $ev" >> $LOG
    (bash $HERE/process_dump.sh $ev > $REDUCE_DIR/process_$ev.log 2>&1); rc=$?
    echo "$(date -u +%FT%TZ) $ev exit $rc" >> $LOG
    [ $rc -eq 0 ] && echo "$ev" >> $DONE
  done
  sleep 120
done
