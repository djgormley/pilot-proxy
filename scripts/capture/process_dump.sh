#!/bin/bash
# Bring one reduced dump home and run everything on it: products, marker files, both detector banks, frame analysis,
# ladder placement. usage: process_dump.sh <event_id>   (needs the event reduced on the analysis host: 280 products)
# Site configuration: CHIME/FRB internal values, never committed; export them before running.
#   JUMP_HOST      ssh jump host into the CHIME network
#   ANALYSIS_HOST  ssh destination of the analysis host, <user>@<host>
#   USER_DATA_DIR  the operator's user-data directory on the analysis host (products are in pilot_reduce/)
# Local configuration:
#   REDUCE_DIR        working directory of the reduction (detector runs kernel_<event>_*, frame_analysis/,
#                     weights_ch33/, and the site's pull_pilots_ready.sh)
#   DATASETS_DIR      where pilot_reduce_<event>/ and pilot_dump_<event>/ are kept
#   ARCHIVE_PRODUCTS  the archive per-band products (<freq_id>.npz) the ladder is read from
#   PYTHON            the interpreter with pilot-proxy installed (or PYTHONPATH set to its src/)
#   BACKUP_DIR        optional: copy the detector runs and the frame residual under <BACKUP_DIR>/<event>/reduce
# Reading CHIME baseband data requires CHIME/FRB authorization; see dumps/README.md.
set -u
: "${JUMP_HOST:?}" "${ANALYSIS_HOST:?}" "${USER_DATA_DIR:?}" "${REDUCE_DIR:?}" "${DATASETS_DIR:?}" "${ARCHIVE_PRODUCTS:?}" "${PYTHON:?}"
EV=$1
R=$REDUCE_DIR
PY=$PYTHON
W=$R/weights_ch33/chime_dtv_weights_k128_ch33measured.bin
mkdir -p $DATASETS_DIR/pilot_reduce_$EV
rsync -a -e "ssh -o ConnectTimeout=20 -J $JUMP_HOST" "$ANALYSIS_HOST:$USER_DATA_DIR/pilot_reduce/$EV/" $DATASETS_DIR/pilot_reduce_$EV/ 2>&1 | grep -v conda
n=$(ls $DATASETS_DIR/pilot_reduce_$EV/*.npz 2>/dev/null | wc -l); echo "$EV products $n"; [ "$n" -ge 280 ] || { echo "not fully reduced"; exit 1; }
$R/pull_pilots_ready.sh $EV 2>&1 | tail -1
L=$DATASETS_DIR/pilot_dump_$EV; CH=""
for f in $(ls $L | sed 's/.*_//; s/.h5//'); do ch=$(python3 -c "f=800-$f*0.390625; print(14+int((f-470)//6))"); [ "$ch" -le 36 ] && CH="$CH --physical-channel $ch"; done
for tag in k230 k230_ch33measured; do O=$R/kernel_${EV}_$tag; rm -rf $O; extra=""; [ $tag = k230_ch33measured ] && extra="--weights-path $W"
  $PY -m pilot_proxy.cli chime-run --input-dir $L --output-dir $O $CH --frame-size-samples 16384 --frames-per-chunk 2 $extra 2>&1 | grep -iE "error" | head -2
  $PY -m pilot_proxy.cli validate-products --run-dir $O --output-json $O/product_validation.json 2>&1 | tail -1
done
FR=$R/frame_analysis/frame_residual_$EV.csv
$PY -m pilot_proxy.cli capture frame-residual --products $DATASETS_DIR/pilot_reduce_$EV --detector-run $R/kernel_${EV}_k230 --out $FR 2>&1 | grep -E "rows ->|Error"
$PY -m pilot_proxy.cli capture ladder-place --archive-products $ARCHIVE_PRODUCTS $R/kernel_${EV}_k230 2>&1 | awk "/kernel_${EV}/{p=1} p" | head -20
sha256sum $R/kernel_${EV}_k230/chime_detector_outputs.npz $R/kernel_${EV}_k230_ch33measured/chime_detector_outputs.npz $FR >> $R/frame_analysis/HASHES_frame_analysis.txt
if [ -n "${BACKUP_DIR:-}" ]; then
  B="$BACKUP_DIR/$EV/reduce"; mkdir -p "$B" && cp -r $R/kernel_${EV}_k230 $R/kernel_${EV}_k230_ch33measured "$B/" && cp $FR "$B/" && echo "$EV backed up"
fi
