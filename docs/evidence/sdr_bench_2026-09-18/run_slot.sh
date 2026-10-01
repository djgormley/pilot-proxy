#!/bin/bash
# usage: run_slot.sh <rung> <slot_index> <mode> <pad|term> [notes]
set -u
R=$1; N=$(printf %02d $2); M=$3; PAD=$4; NOTE=${5:-}
cd /home/djg/rail/output/sdr-bench-2026-09-17
export PYTHONPATH=/home/djg/rail/pilot-proxy/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/djg/rail/output/fisher-fixes-2026-09-08/analysis-venv/bin/python
AMP=$(python3 -c "import json;print(json.load(open('rungs/$R/protocol_spec.json'))['radio']['tx_rms'])")
D=session/$R/slot-$N-$M; mkdir -p session/$R analysis/$R; rm -rf $D
TX=""; [ "$M" = "noise" ] || TX="--transmit"
timeout 300 $PY /home/djg/rail/pilot-proxy/tools/lime_reference_capture_v1.py prepare --output $D \
  --build-manifest build-v2.json --serial 0x1d423d9108f273 --mode $M --tone-component $AMP --record-seconds 2 >/dev/null 2>&1 \
  || { echo "PREPARE FAILED"; exit 1; }
timeout 600 $PY /home/djg/rail/pilot-proxy/tools/lime_reference_capture_v1.py capture --output $D \
  --hardware-authorized --rf-confined-authorized $TX >/dev/null 2>&1
python3 log_slot.py $R $D $PAD "$NOTE" || exit 1
timeout 900 $PY -c "
import sys; sys.path.insert(0,'/home/djg/rail/pilot-proxy/tools')
from analyze_sdr_antenna_controls_v1 import analyze_capture
analyze_capture('$D','analysis/$R/slot-$N-$M')" >/dev/null 2>&1 \
  && python3 -c "
import json, math
null=4.158e-4
r=json.load(open('analysis/$R/slot-$N-$M/record.json'))
t=r['natural_mean_frame_term_powers']; sat=r['adapter_metadata']['sample_quantization']['saturated_component_count']
print('  target %.4e  ratio %.3f  over null %+.1f dB  vs 30cm ref %+.1f dB  saturated %d' % (
  t[0], r['natural_ratio']['finite_median'], 10*math.log10(t[0]/null), 10*math.log10(t[0]/null)-29.5, sat))" \
  || echo "  (analysis skipped)"
