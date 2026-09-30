#!/bin/bash
# reduce every frequency file of one converted event on the analysis host, N containers at a time.
# usage: reduce_driver.sh <event dir with baseband_*.h5> <out dir> [containers=5] [threads per container=8] [max frames=0]
# Site configuration: CHIME/FRB internal values, never committed; export them before running.
#   DUMP_DIR           working directory on the analysis host (holds reduce/)
#   REMOTE_HOME        the operator's home directory on the analysis host, mounted into the container
#   BASEBAND_RAW_ROOT  root of the converted raw baseband archive on the analysis host (read only)
#   USER_DATA_DIR      the operator's user-data directory on the analysis host (products go to pilot_reduce/)
#   BASEBAND_IMAGE     the CHIME/FRB baseband-analysis container image
# Reading CHIME baseband data requires CHIME/FRB authorization; see README.md.
# $DUMP_DIR/reduce/reduce_dump.py is a copy of src/pilot_proxy/instruments/chime/capture/reduce_dump.py.
set -u
: "${DUMP_DIR:?}" "${REMOTE_HOME:?}" "${BASEBAND_RAW_ROOT:?}" "${USER_DATA_DIR:?}" "${BASEBAND_IMAGE:?}"
EV=$1; OUT=$2; NC=${3:-5}; NT=${4:-8}; MF=${5:-0}
IMG=$BASEBAND_IMAGE
RUN="docker run --rm -w /tmp --security-opt seccomp=unconfined --user $(id -u):$(id -g) -e HOME=/tmp -e OPENBLAS_NUM_THREADS=$NT -v $REMOTE_HOME:$REMOTE_HOME -v $BASEBAND_RAW_ROOT:$BASEBAND_RAW_ROOT:ro -v $USER_DATA_DIR:$USER_DATA_DIR $IMG"
mkdir -p "$OUT"
# common frame grid: the latest time0 over all files, so every file has j0 >= 0
T0=$($RUN python -c "
import h5py, sys, glob
fs = sorted(glob.glob('$EV/baseband_*.h5'))
t = [int(h5py.File(f, 'r').attrs['time0_fpga_count']) for f in fs]
n = [int(h5py.File(f, 'r')['baseband'].shape[0]) for f in fs]
print(max(t), len(fs), min(t), min(n), max(n), file=sys.stderr)
print(max(t))
" 2>"$OUT/grid.txt")
echo "grid origin $T0; $(cat "$OUT/grid.txt")" | tee "$OUT/driver.log"
# only the DTV band: freq_id 477..844 covers channels 14..37 (channel 37 is the reference)
ls "$EV"/baseband_*.h5 | awk -F_ '{ id=$NF; sub(/\.h5$/, "", id); if (id+0 >= 477 && id+0 <= 844) print }' | xargs -P "$NC" -I{} sh -c "$RUN python $DUMP_DIR/reduce/reduce_dump.py {} $OUT --grid-fpga $T0 --max-frames $MF --workers $NT >> $OUT/driver.log 2>&1"
echo "done: $(ls $OUT/*.npz 2>/dev/null | wc -l) products" | tee -a "$OUT/driver.log"
