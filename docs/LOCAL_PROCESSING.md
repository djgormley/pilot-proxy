# Local archive processing

This is the sole executable authority for streaming the frozen CHIME inventory
from CADC and processing it on the local WSL workstation. Scientific and
product pins are in [`RERUN_PARAMETER_REGISTER.md`](RERUN_PARAMETER_REGISTER.md),
mandatory gates are in [`VALIDATION_GATES.md`](VALIDATION_GATES.md), and actual
launch values go in a run ledger (the template of the 2026-09 run is
`docs/LOCAL_ARCHIVE_RUN_LEDGER_TEMPLATE.md@30e50c2`).
[`CANFAR_RUNBOOK.md`](CANFAR_RUNBOOK.md) documents the separately qualified
CANFAR path and cannot override this file; neither file overrides the other,
and the operator launches exactly one path.

The run is historical estimation and sufficient-statistic reprocessing. The
coarse positive-excess flag is retained as a bootstrap diagnostic; the fine
rank/eta decision is inactive and no calibrated detection policy is applied.

## Measured machine profile

- Host CPU: 24 physical hybrid cores, 32 hardware threads
- WSL CPU: 32 virtual CPUs
- WSL memory: 62 GiB plus 16 GiB swap
- GPU: one RTX 5000 Ada, 16 GiB, compute capability 8.9
- WSL ext4 free space: measure and record it at launch
- Frozen input: 165,682 files, 25.64 TiB

Keep active staging on WSL ext4. Do not stage through `/mnt/c`, and do not run several detector processes against the single GPU.

## Where a run writes

Storage on the workstation is by pipeline stage, one producer each:
`~/rail/datasets/` holds raw inputs, `~/rail/products/` holds everything
pilot-proxy produces, `~/rail/results/` holds RFIsher's results. A scan
therefore lives entirely under one directory of `~/rail/products/`, named for
the run, with the layout the September 2026 campaign left behind
(`~/rail/products/chime_pilots_rebuild_20260829/`):

    ~/rail/products/<run>/products/   the chime-scan output directory (_per_pilot/ inside)
    ~/rail/products/<run>/staging/    staged downloads; delete only after `stop` confirms exit
    ~/rail/products/<run>/logs/       scan and supervisor logs

Smoke tests and rehearsals go under a dated `~/rail/products/dev_rehearsals_<YYYY-MM>/`.
The former `~/rail/pilot_proxy_runs`, `pilot_proxy_staging` and
`pilot_proxy_logs` directories were retired on 2026-09-06 and must not be
recreated; the blocks below are written for the new layout.

## Freeze one source revision

Finish all source changes before any final gate. Commit and push them, then
record one clean revision and package-source digest:

```bash
cd /home/djg/rail/pilot-proxy
set -euo pipefail
git fetch origin --prune --tags
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
SOURCE_REVISION=$(git rev-parse HEAD)
SOURCE_SHORT=$(git rev-parse --short=12 HEAD)
PACKAGE_SOURCE_SHA256=$(PYTHONPATH=src python3 - <<'PY'
from pilot_proxy.provenance import package_source_sha256
print(package_source_sha256())
PY
)
printf 'source_revision=%s\npackage_source_sha256=%s\n' \
  "$SOURCE_REVISION" "$PACKAGE_SOURCE_SHA256"
```

Use that revision for environment setup, final gates, smoke, rehearsal, launch,
and every resume. Any source change invalidates the later gate evidence.

## Environment

```bash
cd /home/djg/rail/pilot-proxy
VENV_DIR=/home/djg/rail/venvs/archive-local \
PILOT_PROXY_DIR=/home/djg/rail/pilot-proxy \
PILOT_PROXY_SKIP_REGISTRY=1 \
PILOT_PROXY_USE_SYSTEM_PACKAGES=0 \
PYTHON=python3.12 \
bash scripts/setup_env.sh
source /home/djg/rail/venvs/archive-local/bin/activate
umask 077
```

Set `umask 077` in every launch and resume shell so staged data and products
remain owner-only.

Freeze the complete installed runtime before the final gates. This downloads or
builds one wheel for every installed distribution, rebuilds direct Git sources
from their exact commits, builds PilotProxy from a Git archive of the clean
revision, and writes a hash-locked offline bundle:

```bash
set -euo pipefail
SOURCE_SHORT=$(git rev-parse --short=12 HEAD)
RUNTIME_BUNDLE="/home/djg/rail/runtime_freezes/archive-local-${SOURCE_SHORT}"
test ! -e "$RUNTIME_BUNDLE"
python scripts/freeze_archive_runtime.py create \
  --output-dir "$RUNTIME_BUNDLE"
python scripts/freeze_archive_runtime.py verify \
  --bundle-dir "$RUNTIME_BUNDLE"
sha256sum "$RUNTIME_BUNDLE/SHA256SUMS"
```

Record the bundle path, `SHA256SUMS` hash, source revision, package-source hash,
Python version, and package count. The bundle contains `requirements.lock`, the
wheelhouse, the exact source archive, direct-source commits, and hashes for
every payload. It deliberately does not contain the preserved detector library;
verify that separate binary with the pinned command below.

Test an offline restore before launch, using a fresh path outside the checkout:

```bash
RESTORE_VENV=/home/djg/rail/venvs/archive-local-restore
test ! -e "$RESTORE_VENV"
python3.12 -m venv "$RESTORE_VENV"
"$RESTORE_VENV/bin/python" -m pip install \
  --no-index \
  --only-binary=:all: \
  --find-links "$RUNTIME_BUNDLE/wheelhouse" \
  --require-hashes \
  -r "$RUNTIME_BUNDLE/requirements.lock"
"$RESTORE_VENV/bin/python" -m pip check
"$RESTORE_VENV/bin/python" scripts/freeze_archive_runtime.py verify \
  --bundle-dir "$RUNTIME_BUNDLE"
```

Keep the original production environment for the run. The restore proves that
the sealed runtime can be recreated without a package index or source host.

Renew and inspect the CADC certificate:

```bash
set -euo pipefail
/home/djg/rail/venvs/canfar-client/bin/canfar login cadc --force
chmod 600 /home/djg/.ssl/cadcproxy.pem
test "$(stat -c '%a' /home/djg/.ssl/cadcproxy.pem)" = "600"
openssl x509 -in /home/djg/.ssl/cadcproxy.pem -noout -checkend 259200
openssl x509 -in /home/djg/.ssl/cadcproxy.pem -noout -dates
```

## Preserved detector library

The run is pinned to the already preserved SM89 library. Do not replace or
rebuild it during this run.

```bash
set -euo pipefail
KERNEL_LIB=/home/djg/rail/kernels/pilotproxy-detector-core-2.3.0-sm89-f6cd8529ca4b.so
KERNEL_SHA256=f6cd8529ca4b4581aaa37a6007a372d5afb4afa8c730d8a4372a8eaf25e807f2
KERNEL_MANIFEST=/home/djg/rail/kernels/pilotproxy-detector-core-2.3.0-sm89-f6cd8529ca4b.manifest.json
KERNEL_MANIFEST_SHA256=d781d3d4dfbe15dd336b1c89e412a91522376f69a30cfc9543fd52ff6a954cf0
printf '%s  %s\n' \
  "$KERNEL_SHA256" "$KERNEL_LIB" \
  "$KERNEL_MANIFEST_SHA256" "$KERNEL_MANIFEST" | sha256sum --check --strict
export KERNEL_LIB KERNEL_SHA256

python - "$KERNEL_LIB" <<'PY'
import sys
from pilot_proxy.kernel import FStatKernel

kernel = FStatKernel(sys.argv[1])
assert kernel.version.as_string() == "2.3.0"
assert kernel.supports_fine_powers()
assert kernel.supports_fused_fine()
print("kernel", kernel.version.as_string(), kernel.get_fine_specs())
PY
```

Run every command in [`VALIDATION_GATES.md`](VALIDATION_GATES.md), including
the complete test tree, CPU reference, SM89 CUDA test, and zero-skip kernel
suite. The later detection campaign is not a prerequisite for this acquisition,
but every archive and GPU gate must pass on the frozen source.

## One-file GPU gate

The downloaded file has 2048 input streams. Confirm that shape, run one full
detector chunk with no other GPU compute process active, and record peak VRAM.
The production run is blocked unless the product validates and peak use stays
below 13,900 MiB, leaving at least 15 percent of device memory free.

```bash
SMOKE_INPUT=/home/djg/rail/studies_2026-08/data_checks/datatrail_pull_844
SMOKE_FILE="$SMOKE_INPUT/data/chime/baseband/raw/2020/07/15/astro_100058001/baseband_100058001_844.h5"
SOURCE_SHORT=$(git rev-parse --short=12 HEAD)
REHEARSALS="/home/djg/rail/products/dev_rehearsals_$(date -u +%Y-%m)"
mkdir -p "$REHEARSALS"
SMOKE_OUTPUT="$REHEARSALS/local_file_smoke_844_${SOURCE_SHORT}"
VRAM_LOG="$REHEARSALS/local_file_smoke_844_${SOURCE_SHORT}.vram.csv"

test ! -e "$SMOKE_OUTPUT" || exit 1
test ! -e "$VRAM_LOG" || exit 1
python - "$SMOKE_FILE" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as handle:
    baseband = handle["baseband"]
    assert baseband.shape[1] == 2048, baseband.shape
    assert str(baseband.dtype) == "uint8", baseband.dtype
    print("baseband", baseband.shape, baseband.dtype)
PY

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -lms 250 > "$VRAM_LOG" &
vram_monitor_pid=$!

pilot-proxy chime-scan \
  --input-dir "$SMOKE_INPUT" \
  --output-dir "$SMOKE_OUTPUT" \
  --source local \
  --instrument chime \
  --analyzer pilot-proxy-detector \
  --select 844 \
  --weights-path "$PWD/weights/chime_dtv_weights_k128.bin" \
  --weight-coordinate-system post_spectral_sense_normalization \
  --lib-path "$KERNEL_LIB" \
  --set fine_products=on \
  --max-files 1 \
  --max-chunks-per-file 1 \
  --allow-partial

smoke_status=$?
kill "$vram_monitor_pid" 2>/dev/null || true
wait "$vram_monitor_pid" 2>/dev/null || true
test "$smoke_status" -eq 0 || exit "$smoke_status"

peak_vram_mib=$(awk -F, '
  { gsub(/[[:space:]]/, "", $2); if (($2 + 0) > peak) peak = $2 + 0 }
  END { print peak + 0 }
' "$VRAM_LOG")
printf 'peak_vram_mib=%s\n' "$peak_vram_mib"
test "$peak_vram_mib" -lt 13900 || exit 1
```

Validate the smoke output before continuing:

```bash
pilot-proxy validate-products --run-dir "$SMOKE_OUTPUT"

python - "$SMOKE_OUTPUT/_per_pilot/844.npz" \
  "$PACKAGE_SOURCE_SHA256" "$KERNEL_SHA256" <<'PY'
import json
import sys

import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as product:
    version = str(product["detector_version"])
    decision = json.loads(str(product["decision_contract_json"]))
    fine = product["fine_power_u64"]
    unit_count = product["unit_order"].size
    scopes = np.asarray(product["unit_scope"]).astype(str)
    tags = np.asarray(product["unit_git_version_tag"]).astype(str)
    input_hashes = np.asarray(product["unit_input_map_sha256"]).astype(str)
    assert str(product["schema_version"]) == "pilotproxy_per_pilot_product_v5"
    assert f"source={sys.argv[2]}" in version
    assert f"kernel_sha256={sys.argv[3]}" in version
    assert str(product["fine_status"]) == "enabled"
    assert fine.dtype == np.uint64 and fine.ndim == 3
    assert fine.shape[1:] == (3, 256)
    assert scopes.size == tags.size == input_hashes.size == unit_count
    assert all(scopes) and all(tags) and all(input_hashes)
    assert all(len(value) == 64 and set(value) <= set("0123456789abcdef")
               for value in input_hashes)
    assert decision["fine_candidate_decision"]["active"] is False
print("smoke product identity, receiver state, and exact terms pass")
PY
```

Record the HDF5 shape, peak VRAM, validation result, output path, source digest,
and library digest in the run ledger. Earlier smoke evidence used a different
source digest and is not a final-source gate.

## Measured transfer settings

Four equal-size archive objects were downloaded through the same `cadcget` path used by the scan. Every file matched its archive MD5.

| Workers | Elapsed | Throughput |
|---:|---:|---:|
| 1 | 72.863 s | 4.781 MiB/s |
| 2 | 45.397 s | 7.673 MiB/s |
| 4 | 36.785 s | 9.469 MiB/s |

A reverse `4,2,1` sequence measured 21.001, 24.956, and 14.209 MiB/s. Across
both sequences, aggregate throughput was 13.053 MiB/s with four workers, 11.737
MiB/s with two, and 7.154 MiB/s with one. Transfer conditions varied, so four
workers are the best tested starting value, not a settled optimum. Use four
download workers and eight staged-file slots for the production-profile
rehearsal. Analysis remains in frozen inventory order even when transfers finish
out of order. Eight slots bound frozen-inventory scratch use to about 8.35 GiB.
Do not raise either value without another bounded measurement.

## Production-profile resume rehearsal

This rehearsal uses the production archive source, full file chunks, ordered
four-worker prefetch, the preserved library, and fine products. Its short
checkpoint interval exists only to force an interruption and resume. Keep its
capped product separate from production.

```bash
BUNDLE_DIR=/home/djg/rail/archive_inputs/chime-pilots-v5
SOURCE_SHORT=$(git rev-parse --short=12 HEAD)
REHEARSALS="/home/djg/rail/products/dev_rehearsals_$(date -u +%Y-%m)"
mkdir -p "$REHEARSALS"
REHEARSAL_OUTPUT="$REHEARSALS/local_archive_rehearsal_844_${SOURCE_SHORT}"
REHEARSAL_STAGING="$REHEARSALS/local_archive_rehearsal_844_${SOURCE_SHORT}.staging"

test ! -e "$REHEARSAL_OUTPUT" || exit 1
test ! -e "$REHEARSAL_STAGING" || exit 1

REHEARSAL_ARGS=(
  chime-scan
  --source cadc-datatrail
  --inventory "$BUNDLE_DIR/inventory.jsonl"
  --output-dir "$REHEARSAL_OUTPUT"
  --staging-dir "$REHEARSAL_STAGING"
  --instrument chime
  --analyzer pilot-proxy-detector
  --select 844
  --download-workers 4
  --max-staged-files 8
  --checkpoint-every 2
  --weights-path "$PWD/weights/chime_dtv_weights_k128.bin"
  --weight-coordinate-system post_spectral_sense_normalization
  --lib-path "$KERNEL_LIB"
  --set fine_products=on
  --max-files 8
  --allow-partial
)

pilot-proxy "${REHEARSAL_ARGS[@]}"
```

Run that command in one terminal. In another, wait for
`$REHEARSAL_OUTPUT/_per_pilot/844.npz` to appear, then interrupt the first
terminal with `Ctrl-C` and wait for it to exit. Run the exact same command again:

```bash
pilot-proxy "${REHEARSAL_ARGS[@]}"
pilot-proxy validate-products --run-dir "$REHEARSAL_OUTPUT"

python - "$REHEARSAL_OUTPUT/_per_pilot/844.npz" \
  "$REHEARSAL_OUTPUT/scan_scope.json" \
  "$PACKAGE_SOURCE_SHA256" "$KERNEL_SHA256" <<'PY'
import json
import sys

import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as product:
    version = str(product["detector_version"])
    units = np.asarray(product["unit_keys"]).astype(str)
    frames = np.asarray(product["frame_index"])
    fine = product["fine_power_u64"]
    scopes = np.asarray(product["unit_scope"]).astype(str)
    tags = np.asarray(product["unit_git_version_tag"]).astype(str)
    input_hashes = np.asarray(product["unit_input_map_sha256"]).astype(str)
    assert units.size == 8 and np.unique(units).size == 8
    assert np.unique(frames).size == frames.size
    assert f"source={sys.argv[3]}" in version
    assert f"kernel_sha256={sys.argv[4]}" in version
    assert str(product["fine_status"]) == "enabled"
    assert fine.dtype == np.uint64 and fine.shape[1:] == (3, 256)
    assert scopes.size == tags.size == input_hashes.size == units.size
    assert all(scopes) and all(tags) and all(input_hashes)
    assert all(len(value) == 64 and set(value) <= set("0123456789abcdef")
               for value in input_hashes)
with open(sys.argv[2], encoding="utf-8") as handle:
    scope = json.load(handle)
execution = scope["execution"]
assert execution["download_workers"] == 4
assert execution["max_staged_files"] == 8
assert execution["checkpoint_every"] == 2
print("resume identity, receiver state, order, and exact terms pass")
PY

test -z "$(find "$REHEARSAL_STAGING" -type f -name '*.h5' -print -quit)"
```

Record the interruption point, resume output, unique unit and frame counts,
validation, empty-staging check, source digest, and kernel digest in the run
ledger. Never reuse this capped output for production. Earlier rehearsal
evidence used a different source digest and is not a final-source gate.

## WSL host controls

Finish Windows and WSL updates, reboot once, and rerun the gates before launch.
Use AC power and a `tmux` session inside WSL. Do not depend on an open terminal
window to keep the scan alive.

Inspect and record the active Windows power scheme and plugged-in sleep policy
from WSL:

```bash
powershell.exe -NoProfile -Command 'powercfg /getactivescheme'
powershell.exe -NoProfile -Command 'powercfg /query SCHEME_CURRENT SUB_SLEEP'
```

If the workstation owner approves disabling plugged-in sleep for the run,
record the current values first, then use an elevated Windows PowerShell:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

Restore the recorded values after closeout. Do not change battery policy. Keep
automatic restart risk, planned maintenance, and the current Windows update
state in the launch decision; a long scan still needs controlled checkpointed
stops and resumes.

## Production command

Set `EXPECTED_SOURCE_REVISION` and `EXPECTED_PACKAGE_SOURCE_SHA256` from the
filled external run ledger. Do not derive either value from the current tree.

```bash
cd /home/djg/rail/pilot-proxy
source /home/djg/rail/venvs/archive-local/bin/activate
set -euo pipefail
umask 077

: "${EXPECTED_SOURCE_REVISION:?set from the external run ledger}"
: "${EXPECTED_PACKAGE_SOURCE_SHA256:?set from the external run ledger}"
test "$(stat -c '%a' /home/djg/.ssl/cadcproxy.pem)" = "600"
openssl x509 -in /home/djg/.ssl/cadcproxy.pem -noout -checkend 259200

BUNDLE_DIR=/home/djg/rail/archive_inputs/chime-pilots-v5
INVENTORY_PATH="$BUNDLE_DIR/inventory.jsonl"
RUN_ROOT=/home/djg/rail/products/chime_pilots_local_v5
OUTPUT_DIR="$RUN_ROOT/products"
STAGING_DIR="$RUN_ROOT/staging"
LOG_DIR="$RUN_ROOT/logs"
RUN_LOG="$LOG_DIR/chime_pilots_local_v5.log"
WEIGHTS_PATH="$PWD/weights/chime_dtv_weights_k128.bin"
KERNEL_LIB=/home/djg/rail/kernels/pilotproxy-detector-core-2.3.0-sm89-f6cd8529ca4b.so
KERNEL_SHA256=f6cd8529ca4b4581aaa37a6007a372d5afb4afa8c730d8a4372a8eaf25e807f2
KERNEL_MANIFEST=/home/djg/rail/kernels/pilotproxy-detector-core-2.3.0-sm89-f6cd8529ca4b.manifest.json
KERNEL_MANIFEST_SHA256=d781d3d4dfbe15dd336b1c89e412a91522376f69a30cfc9543fd52ff6a954cf0

git fetch origin --prune --tags
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE_REVISION"
test "$(git rev-parse origin/main)" = "$EXPECTED_SOURCE_REVISION"
SOURCE_REVISION=$(git rev-parse HEAD)
PACKAGE_SOURCE_SHA256=$(python - <<'PY'
from pilot_proxy.provenance import package_source_sha256
print(package_source_sha256())
PY
)
test "$PACKAGE_SOURCE_SHA256" = "$EXPECTED_PACKAGE_SOURCE_SHA256"
printf 'source_revision=%s\npackage_source_sha256=%s\n' \
  "$SOURCE_REVISION" "$PACKAGE_SOURCE_SHA256"

printf '%s  %s\n' \
  b2cfef752a6f2cf88317141a974161439299668a37a5464d710b3800b9a872d8 "$INVENTORY_PATH" \
  d0ab78db9dec847fb5270fb94e3fc45efd6bbbc71adc376830fb71467693079a "$BUNDLE_DIR/exclusions.jsonl" \
  077915779d234ad9cca1e9b52171e28025a60474f20fad7aff5684f55ce0c4c7 "$BUNDLE_DIR/pending_resolution.json" \
  5695e1cc9c007cb2c79ad39535cf9ed2fb20848ad00d3cf0215933670d0707e5 "$BUNDLE_DIR/inventory_manifest.json" \
  bc59e77442a4c15f74c716d14eaeea4f10a69517d3bfb8c88ce10a7a42ea1e15 "$PWD/configs/receiver_profiles/chime_dtv_fengine.json" \
  1383c6d0ca521a26b317d008feb6e09eb41427155bda9a320f70bca62e0e6259 "$WEIGHTS_PATH" \
  d0ccc8162a350e9d3266e6acf3b38d2fe5982c474b73ef0715b8b838954e81a7 "$WEIGHTS_PATH.manifest.json" \
  "$KERNEL_SHA256" "$KERNEL_LIB" \
  "$KERNEL_MANIFEST_SHA256" "$KERNEL_MANIFEST" | sha256sum --check --strict
if pgrep -af '[p]ilot-proxy chime-scan'; then
  exit 1
fi

RUN_ARGS=(
  chime-scan
  --source cadc-datatrail
  --inventory "$INVENTORY_PATH"
  --output-dir "$OUTPUT_DIR"
  --staging-dir "$STAGING_DIR"
  --instrument chime
  --analyzer pilot-proxy-detector
  --select 506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844
  --download-workers 4
  --max-staged-files 8
  --checkpoint-every 250
  --weights-path "$WEIGHTS_PATH"
  --weight-coordinate-system post_spectral_sense_normalization
  --lib-path "$KERNEL_LIB"
  --set fine_products=on
)
```

For the initial launch only, require fresh output and staging paths, copy the run
ledger beside the output path, fill every before-launch field, then start:

```bash
test ! -e "$OUTPUT_DIR" || exit 1
test ! -e "$STAGING_DIR" || exit 1
install -d -m 700 "$LOG_DIR"
test ! -e "$RUN_LOG" || exit 1
pilot-proxy "${RUN_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
```

Keep `--allow-partial` off. The scan records its resolved worker, staging, and
checkpoint settings in `scan_scope.json` before downloading data.

The 250-unit checkpoint interval reduces repeated whole-product writes about
fivefold compared with 50. A stop can repeat up to 249 successfully analyzed
units plus downloads that were prefetched but not represented by the last
checkpoint. Complete the production-profile rehearsal before launch.

`fine_products=on` retains exact fine powers. It does not enable fine detection.

## Live run checks

Run the health check from another `tmux` pane at launch and about every five
minutes during unattended operation:

```bash
python scripts/monitor_local_archive.py \
  --run-dir "$OUTPUT_DIR" \
  --staging-dir "$STAGING_DIR" \
  --log "$RUN_LOG" \
  --expect-running \
  --min-free-gib 100 \
  --stale-minutes 120 \
  --min-gpu-free-mib 2500 \
  --max-gpu-temperature-c 85 \
  --certificate /home/djg/.ssl/cadcproxy.pem \
  --min-certificate-hours 72
```

The command is read-only and exits nonzero for a missing or duplicate scan,
less than 100 GiB free, no output/log/staging heartbeat for two hours, low GPU
headroom, excessive GPU temperature, an unavailable GPU, or a concerning log
pattern. It also requires the CADC certificate to remain mode `0600` with at
least 72 hours left. Use `--json` for machine-readable output. `watch -n 300`
may be placed before the command for a continuously visible pane.

Treat low space, a stale heartbeat, new quarantine activity, a fetch failure,
an expired certificate, a traceback, or a GPU memory error as a stop condition.
Wait for a checkpoint when possible, press `Ctrl-C`, and wait for every worker
to exit. Inspect `df -h`, `nvidia-smi`, the last log lines, the newest
`_per_pilot/*.npz` timestamp, and staging contents before deciding whether to
resume. Do not delete staging files while any old worker is alive.

## First-checkpoint tripwire

After the first 250-unit product checkpoint appears, stop the scan cleanly and
inspect it before continuing. The terminal combined files do not exist yet, so
run the deep per-pilot audit and verify the frozen identities and exact terms:

```bash
set -euo pipefail
FIRST_PRODUCT="$OUTPUT_DIR/_per_pilot/506.npz"
if pgrep -af '[p]ilot-proxy chime-scan'; then
  exit 1
fi
python tools/audit_per_pilot.py "$OUTPUT_DIR/_per_pilot"

python - "$FIRST_PRODUCT" "$OUTPUT_DIR/scan_scope.json" \
  "$PACKAGE_SOURCE_SHA256" "$KERNEL_SHA256" <<'PY'
import json
import sys

import numpy as np
from pilot_proxy.product_contract import validate_current_product_identity

with np.load(sys.argv[1], allow_pickle=False) as product:
    validate_current_product_identity(product)
    version = str(product["detector_version"])
    decision = json.loads(str(product["decision_contract_json"]))
    target = product["p_target_u64"]
    lower = product["p_ref_lower_u64"]
    upper = product["p_ref_upper_u64"]
    reference = product["p_ref_sum_u64"]
    fine = product["fine_power_u64"]
    frames = product["frame_index"].shape[0]
    unit_count = product["unit_order"].size
    scopes = np.asarray(product["unit_scope"]).astype(str)
    tags = np.asarray(product["unit_git_version_tag"]).astype(str)
    input_hashes = np.asarray(product["unit_input_map_sha256"]).astype(str)
    assert str(product["schema_version"]) == "pilotproxy_per_pilot_product_v5"
    assert f"source={sys.argv[3]}" in version
    assert f"kernel_sha256={sys.argv[4]}" in version
    assert all(x.dtype == np.uint64 and x.shape == (frames, 1)
               for x in (target, lower, upper, reference))
    assert all(int(a) + int(b) == int(c)
               for a, b, c in zip(lower[:, 0], upper[:, 0], reference[:, 0]))
    assert fine.dtype == np.uint64 and fine.shape == (frames, 3, 256)
    assert scopes.size == tags.size == input_hashes.size == unit_count
    assert all(scopes) and all(tags) and all(input_hashes)
    assert all(len(value) == 64 and set(value) <= set("0123456789abcdef")
               for value in input_hashes)
    assert str(product["fine_status"]) == "enabled"
    assert decision["fine_candidate_decision"]["active"] is False
    print("frames", frames)
    print("designated_bins", product["fine_designated_bins"])
    print("census_exclusions", product["fine_census_excluded_bins"])
with open(sys.argv[2], encoding="utf-8") as handle:
    scope = json.load(handle)
execution = scope["execution"]
assert execution["download_workers"] == 4
assert execution["max_staged_files"] == 8
assert execution["checkpoint_every"] == 250
print("first-checkpoint tripwire passes")
PY
```

Record the result in the run ledger. Resume only after every assertion passes.

## Quick resume

For a planned stop, wait for the active `_per_pilot/<freq_id>.npz` checkpoint to
change, press `Ctrl-C`, and wait for the command to exit. For an unplanned stop,
first confirm that no prior scan or download worker remains. Never remove the
staging directory while an old worker can still write to it.

Resume only from the recorded source revision and preserved library. Do not
rebuild, change options, or change the inventory. In a new shell:

```bash
cd /home/djg/rail/pilot-proxy
source /home/djg/rail/venvs/archive-local/bin/activate
set -euo pipefail
umask 077
: "${KERNEL_LIB:?set the path recorded in the run ledger}"
: "${EXPECTED_SOURCE_REVISION:?set from the run ledger}"
: "${EXPECTED_PACKAGE_SOURCE_SHA256:?set from the run ledger}"
: "${EXPECTED_KERNEL_SHA256:?set from the run ledger}"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE_REVISION"
test "$(PYTHONPATH=src python - <<'PY'
from pilot_proxy.provenance import package_source_sha256
print(package_source_sha256())
PY
)" = "$EXPECTED_PACKAGE_SOURCE_SHA256"
test "$(stat -c '%a' /home/djg/.ssl/cadcproxy.pem)" = "600"
openssl x509 -in /home/djg/.ssl/cadcproxy.pem -noout -checkend 259200
printf '%s  %s\n' "$EXPECTED_KERNEL_SHA256" "$KERNEL_LIB" | \
  sha256sum --check --strict
pgrep -af '[p]ilot-proxy chime-scan' && exit 1
```

Recreate `RUN_ARGS`, `LOG_DIR`, and `RUN_LOG` from the production block with the
ledger values and rerun `pilot-proxy "${RUN_ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"`.
Keep the same output directory, staging directory, inventory, selection, worker
and slot counts, checkpoint interval, weights, coordinate system, library, and
fine-product setting. The analyzer loads the last checkpoint and skips its
recorded unit keys. Keep `pipefail` active so a scan failure is not hidden by
`tee`.

## Certificate renewal during production

Record the certificate expiry at launch. At least 72 hours before expiry, wait
for the next checkpoint, stop as described above, and wait for every worker to
exit. Renew and inspect the certificate:

```bash
source /home/djg/rail/venvs/canfar-client/bin/activate
set -euo pipefail
canfar login cadc --force
chmod 600 /home/djg/.ssl/cadcproxy.pem
test "$(stat -c '%a' /home/djg/.ssl/cadcproxy.pem)" = "600"
openssl x509 -in /home/djg/.ssl/cadcproxy.pem -noout -checkend 259200
openssl x509 -in /home/djg/.ssl/cadcproxy.pem -noout -dates
```

Return to the production environment and follow the quick-resume procedure with
the identical command. Repeat this planned cycle whenever the new expiry is
shorter than the remaining run.

## Runtime expectations

At the slower first-sequence four-worker rate, 25.64 TiB represents about 33
days of transfer service time if that short benchmark rate remains constant.
The reverse sequence was much faster, which shows why 33 days is a conservative
planning projection rather than an ETA. Download and GPU analysis overlap, so
wall time is governed by the slower stage, with checkpoint pauses, outages, and
other stalls added. Follow the planned certificate renewal cycle rather than
allowing expiry to stop the run.

Multiple CPU threads are available for support work, but the current detector is one ordered GPU consumer. Additional GPU batching and append-only checkpoints may improve throughput later; both change deeper execution paths and need separate parity and interruption testing before production use.

## After-run acceptance

Use the same source revision and environment recorded in the run ledger. Set the
three expected values from that ledger, then run the closeout with no scan or
download worker active:

```bash
cd /home/djg/rail/pilot-proxy
source /home/djg/rail/venvs/archive-local/bin/activate
set -euo pipefail
umask 077

: "${EXPECTED_SOURCE_REVISION:?set from the run ledger}"
: "${EXPECTED_PACKAGE_SOURCE_SHA256:?set from the run ledger}"
: "${EXPECTED_KERNEL_SHA256:?set from the run ledger}"

BUNDLE_DIR=/home/djg/rail/archive_inputs/chime-pilots-v5
INVENTORY_PATH="$BUNDLE_DIR/inventory.jsonl"
RUN_ROOT=/home/djg/rail/products/chime_pilots_local_v5
OUTPUT_DIR="$RUN_ROOT/products"
STAGING_DIR="$RUN_ROOT/staging"
WEIGHTS_PATH="$PWD/weights/chime_dtv_weights_k128.bin"
PROFILE_PATH="$PWD/configs/receiver_profiles/chime_dtv_fengine.json"
KERNEL_LIB=/home/djg/rail/kernels/pilotproxy-detector-core-2.3.0-sm89-f6cd8529ca4b.so
KERNEL_MANIFEST=/home/djg/rail/kernels/pilotproxy-detector-core-2.3.0-sm89-f6cd8529ca4b.manifest.json

test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE_REVISION"
test "$(PYTHONPATH=src python - <<'PY'
from pilot_proxy.provenance import package_source_sha256
print(package_source_sha256())
PY
)" = "$EXPECTED_PACKAGE_SOURCE_SHA256"
test "$EXPECTED_KERNEL_SHA256" = \
  f6cd8529ca4b4581aaa37a6007a372d5afb4afa8c730d8a4372a8eaf25e807f2
printf '%s  %s\n' \
  b2cfef752a6f2cf88317141a974161439299668a37a5464d710b3800b9a872d8 "$INVENTORY_PATH" \
  d0ab78db9dec847fb5270fb94e3fc45efd6bbbc71adc376830fb71467693079a "$BUNDLE_DIR/exclusions.jsonl" \
  077915779d234ad9cca1e9b52171e28025a60474f20fad7aff5684f55ce0c4c7 "$BUNDLE_DIR/pending_resolution.json" \
  5695e1cc9c007cb2c79ad39535cf9ed2fb20848ad00d3cf0215933670d0707e5 "$BUNDLE_DIR/inventory_manifest.json" \
  bc59e77442a4c15f74c716d14eaeea4f10a69517d3bfb8c88ce10a7a42ea1e15 "$PROFILE_PATH" \
  1383c6d0ca521a26b317d008feb6e09eb41427155bda9a320f70bca62e0e6259 "$WEIGHTS_PATH" \
  d0ccc8162a350e9d3266e6acf3b38d2fe5982c474b73ef0715b8b838954e81a7 "$WEIGHTS_PATH.manifest.json" \
  "$EXPECTED_KERNEL_SHA256" "$KERNEL_LIB" \
  d781d3d4dfbe15dd336b1c89e412a91522376f69a30cfc9543fd52ff6a954cf0 "$KERNEL_MANIFEST" | \
  sha256sum --check --strict
if pgrep -af '[p]ilot-proxy chime-scan'; then
  exit 1
fi
test -d "$STAGING_DIR"
test -z "$(find "$STAGING_DIR" -type f -print -quit)"

python tools/audit_per_pilot.py "$OUTPUT_DIR/_per_pilot" | \
  tee "$OUTPUT_DIR/per_pilot_audit.txt"

# This reads the inventory, scope, products, staging, and quarantine state. It
# changes no product and writes only the requested audit report.
python tools/audit_local_archive_run.py \
  --inventory "$INVENTORY_PATH" \
  --run-dir "$OUTPUT_DIR" \
  --staging-dir "$STAGING_DIR" \
  --package-source-sha256 "$EXPECTED_PACKAGE_SOURCE_SHA256" \
  --kernel-sha256 "$EXPECTED_KERNEL_SHA256" \
  --weight-bank-sha256 \
    1383c6d0ca521a26b317d008feb6e09eb41427155bda9a320f70bca62e0e6259 \
  --weight-manifest-sha256 \
    d0ccc8162a350e9d3266e6acf3b38d2fe5982c474b73ef0715b8b838954e81a7 \
  --expected-inventory-sha256 \
    b2cfef752a6f2cf88317141a974161439299668a37a5464d710b3800b9a872d8 \
  --expected-units 165682 \
  --freq-ids 506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844 \
  --download-workers 4 \
  --max-staged-files 8 \
  --checkpoint-every 250 \
  --output-json "$OUTPUT_DIR/final_inventory_audit.json"

COMBINE_STATUS=$(python - "$OUTPUT_DIR/scan_scope.json" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text())["terminal_combine"]["status"])
PY
)
if [[ "$COMBINE_STATUS" == "combined" ]]; then
  pilot-proxy validate-products \
    --run-dir "$OUTPUT_DIR" \
    --output-json "$OUTPUT_DIR/product_validation.json"
else
  echo "canonical validation not applicable: empty all-channel intersection"
fi

(
  cd "$OUTPUT_DIR"
  mapfile -d '' PRODUCTS < <(
    find _per_pilot -maxdepth 1 -type f -name '*.npz' -print0 | sort -z
  )
  test "${#PRODUCTS[@]}" -eq 23
  MANIFEST_FILES=(
    "${PRODUCTS[@]}"
    scan_scope.json
    per_pilot_audit.txt
    final_inventory_audit.json
  )
  if [[ -f product_validation.json ]]; then
    MANIFEST_FILES+=(product_validation.json)
  fi
  sha256sum "${MANIFEST_FILES[@]}" > product_sha256s.txt
  sha256sum product_sha256s.txt > product_sha256s.txt.sha256
  sha256sum --check --strict product_sha256s.txt
  sha256sum --check --strict product_sha256s.txt.sha256
  cat product_sha256s.txt.sha256
)
```

Record the audit paths, inventory and product totals, combine disposition,
manifest hash, quarantine result, empty-staging result, and UTC acceptance time
in the run ledger. Do not accept the run if any command or check fails.
