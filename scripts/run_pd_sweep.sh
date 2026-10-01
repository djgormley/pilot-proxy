#!/usr/bin/env bash
# Publication P_d sweep (docs/PUBLICATION_VALIDATION.md@30e50c2 item 2), CPU-reference
# backend, sharded across cores with resume support.
#
#   bash run_pd_sweep.sh [TRIALS] [JOBS] [OUT]
#
# Defaults: TRIALS=300, JOBS=$(nproc), OUT=results/pd_curves_cpu.
# Env overrides: IQ (capture path), SNR_LIST, OFF_LIST.
# Rerunning skips finished shards, so an interrupted sweep resumes cleanly.
# Needs the evaluate_snr cpu-reference summary fix applied (see the patch).
set -u
TRIALS="${1:-300}"
JOBS="${2:-$(nproc)}"
OUT="${3:-results/pd_curves_cpu}"
IQ="${IQ:-generated/atsc/atsc_8vsb_complex64.cfile}"
SNR_LIST="${SNR_LIST:-$(seq -38 1 -24)}"
OFF_LIST="${OFF_LIST:--1000 0 1000}"

mkdir -p "$OUT/shards"
echo "sweep: trials=$TRIALS jobs=$JOBS out=$OUT iq=$IQ"
i=0
for snr in $SNR_LIST; do
  for off in $OFF_LIST; do
    d="$OUT/shards/snr${snr}_off${off}"
    if [ -f "$d/dtv_snr_summary.csv" ]; then
      echo "skip (done): $d"
      continue
    fi
    mkdir -p "$d"
    (
      pilot-proxy evaluate-snr --input-iq "$IQ" \
        --physical-channel 14 --frame-size-samples 16384 \
        --num-input-streams 4 \
        --requested-data-shelf-snr-db "$snr" --frequency-offset-hz "$off" \
        --threshold-data-shelf-snr-db -32 \
        --detector-backend cpu-reference --noise-source python \
        --noise-trials "$TRIALS" --output-dir "$d" \
        > "$d/shard.log" 2>&1 \
      && echo "done: $d" || echo "FAILED: $d (see $d/shard.log)"
    ) &
    i=$((i + 1))
    if [ $((i % JOBS)) -eq 0 ]; then wait; fi
  done
done
wait

python3 - "$OUT" <<'PY'
import csv, glob, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows, header = [], None
shards = sorted(glob.glob(str(out / "shards" / "*" / "dtv_snr_summary.csv")))
for p in shards:
    with open(p) as fh:
        r = list(csv.reader(fh))
    if not r:
        continue
    if header is None:
        header = r[0]
    rows += r[1:]
if header is None:
    raise SystemExit("no shard summaries found")
with open(out / "dtv_snr_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header)
    w.writerows(rows)
print(f"merged {len(rows)} rows from {len(shards)} shards -> "
      f"{out/'dtv_snr_summary.csv'}")
PY

pilot-proxy plot-results --input-csv "$OUT/dtv_snr_summary.csv" \
  --output-png "$OUT/dtv_snr_sweep.png"
echo "figure: $OUT/dtv_snr_sweep.png"
