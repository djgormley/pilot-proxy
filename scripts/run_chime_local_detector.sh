#!/usr/bin/env bash
# Local CHIME workflow: one positive-excess detector run.
set -euo pipefail

INPUT_DIR="${PILOT_PROXY_CHIME_INPUT_DIR:-$HOME/dataset/canfar_pilots_10s}"
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "CHIME input directory not found: $INPUT_DIR" >&2
  echo "Set PILOT_PROXY_CHIME_INPUT_DIR to your CHIME baseband HDF5 directory." >&2
  exit 1
fi
ROOT=generated/chime_real

ANALYSIS_RUN=${PILOT_PROXY_CHIME_OUTPUT_DIR:-${ROOT}/canfar_pilots_10s_positive_excess_full}
FRAMES_PER_CHUNK=${PILOT_PROXY_FRAMES_PER_CHUNK:-2}
MAX_FRAMES=${PILOT_PROXY_MAX_FRAMES:-}

if [[ "${PILOT_PROXY_CLEAN_RUNS:-0}" == "1" ]]; then
  rm -rf "$ANALYSIS_RUN"
fi

echo "Using CHIME input directory: $INPUT_DIR"
echo "Using frames per chunk: ${FRAMES_PER_CHUNK}"
echo "Using CHIME output directory: ${ANALYSIS_RUN}"
if [[ -n "$MAX_FRAMES" ]]; then
  echo "Using max frames: ${MAX_FRAMES}"
fi

CHIME_RUN_EXTRA_ARGS=()
if [[ -n "$MAX_FRAMES" ]]; then
  CHIME_RUN_EXTRA_ARGS+=(--max-frames "$MAX_FRAMES")
fi

PYTHONPATH=src python -m pilot_proxy.cli chime-run \
  --input-dir "$INPUT_DIR" \
  --physical-channel-range 14:36 \
  --frames-per-chunk "$FRAMES_PER_CHUNK" \
  --output-dir "$ANALYSIS_RUN" \
  --plot \
  "${CHIME_RUN_EXTRA_ARGS[@]}"

PYTHONPATH=src python -m pilot_proxy.cli validate-products \
  --run-dir "$ANALYSIS_RUN" \
  --output-json "$ANALYSIS_RUN/product_validation.json"
