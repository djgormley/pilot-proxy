# PilotProxy archive integration

CHIME archive runs need bounded staging, restartable products, and one detector
product per selected pilot. PilotProxy now owns that runtime. Archive discovery
uses the pinned `datatrail-cli` client; file inspection and transfer use CADC
clients directly.

The bundled archive engine enumerates files, limits the staging area, streams
arrays, requests checkpoints, and fans a run out by channel. PilotProxy supplies
the packed CHIME reader, detector and control analyzers, CUDA detector call, and
product writers.

## Runtime layout

| Part | Implementation |
|---|---|
| instrument | bundled CHIME target geometry: 800 MHz band top, 1024 coarse channels, `nfft=16384`, inverted spectral sense |
| source | bundled `local` and `cadc-datatrail` sources |
| reader | bundled packed CHIME reader for detector runs |
| analyzer | bundled detector and control analyzers |
| discovery client | tested revision pinned in `requirements/archive.txt`; indexed release required by the `archive` extra |

`pilot-proxy chime-scan` writes one `<freq_id>.npz` product for each selected
pilot with usable input. When products share `(event, frame-in-file)` identities,
the combine step aligns the common identities and writes:

- `chime_detector_outputs.npz`
- `chime_spectrogram_cache.npz`
- `chime_reductions_10s.npz`
- CSV summaries under `tables/`
- JSON provenance in `run_config.json`, `stats.json`, and `input_manifest.json`

If the intersection across all completed pilots is empty, the scan leaves the
completed per-pilot products under `_per_pilot/`. Use
`pilot-proxy chime-combine --report` to inspect compatible subsets.

The `nfft=16384` value is the target frame size for the CHIME F-engine upgrade.
It is not a statement that every deployed CHIME product uses this size. Verify
the frame size and stream layout against the data product used for each run.

## Setup

Clone this repository and install its archive workflow:

```bash
git clone https://github.com/WVURAIL/pilot-proxy.git ~/pilot-proxy
cd ~/pilot-proxy
python -m pip install -r requirements/archive.txt
python -m pip install -e ".[archive,test]"
```

`scripts/setup_env.sh` performs the same installation in a guarded virtual
environment, checks the bundled source, reader, and analyzers directly, and
builds the CUDA kernel when a GPU and `nvcc` are available. See
[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md#environment-setup) for the exact
CANFAR sequence.

The detector has no CPU fallback in `chime-scan`. Run detector scans on a GPU
node with CuPy matched to the CUDA runtime. `chime-survey`, `chime-inventory`,
and the default `chime-control-scan` path do not require a GPU.

## Selection uses CHIME `freq_id`

`pilot-proxy chime-scan --select` accepts CHIME coarse-channel identifiers,
called `freq_id`s. It does not accept ATSC physical-channel numbers. For the
`cadc-datatrail` source, `--select` is optional. When omitted, the command reads
the distinct `freq_id`s from the inventory and prints the resolved set before
staging any file. A local source has no inventory from which to infer the scope,
so `--select` is required.

Archive-style filenames use:

```text
baseband_<event>_<freq_id>.h5
```

For the default DTV physical-channel range 14-36, use the 23 pilot `freq_id`s in
[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md#selection-convention). Under the
bundled geometry, the physical-channel 14 pilot maps to `freq_id=844`.

## Local source

The default local filename expression is `_(\d+)\.h5$`. Inspect the identifiers
before selecting one:

```bash
find "$LOCAL_H5" -maxdepth 1 -name "*.h5" \
  | sed -E 's/.*_([0-9]+)\.h5$/\1/' \
  | sort -n | uniq
```

Then run one file and one chunk:

```bash
pilot-proxy chime-scan \
  --input-dir "$LOCAL_H5" \
  --output-dir "$HOME/pilot_proxy_runs/detector_smoke_844" \
  --source local \
  --select 844 \
  --max-files 1 \
  --max-chunks-per-file 1 \
  --allow-partial
```

For another filename layout, pass
`--source-freq-id-regex '<regex-with-one-capturing-group>'`. An explicit
`--set 'source_freq_id_regex=...'` still takes precedence.

## CADC inventory and scan

Renew the CADC proxy certificate, then build a bounded inventory:

```bash
cadc-get-cert -u <your-cadc-username>

pilot-proxy chime-survey \
  --freq-ids 506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844 \
  --name chime-pilots \
  --max-events 5
```

The managed inventory is written under the canonical inventory root and the
command prints the exact `inventory.jsonl` path. The default remains:

```text
~/datatrawl-inventories/chime-pilots/inventory.jsonl
```

Set `PILOT_PROXY_INVENTORY_ROOT` to an absolute path to change that root.

Inspect the inventory without staging baseband data:

```bash
pilot-proxy chime-inventory --inventory-name chime-pilots
```

For an inventory stored elsewhere, pass `--inventory <path>` or
`--source-root <survey-root>`.

First run one selected channel, one file, and one chunk:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/detector_smoke_844" \
  --inventory-name chime-pilots \
  --select 844 \
  --max-files 1 \
  --max-chunks-per-file 1 \
  --allow-partial
```

`--allow-partial` is required because the cap deliberately leaves inventoried
files unprocessed. Omit the caps and that acknowledgement for a production run:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/chime-pilots" \
  --inventory-name chime-pilots
```

## Control channels

Control bins use the bundled control analyzer and complex CHIME reader:

```bash
pilot-proxy chime-survey \
  --name chime-controls \
  --freq-ids 484,491,515,545,591,745

pilot-proxy chime-control-scan \
  --output-dir "$HOME/pilot_proxy_runs/chime-controls-smoke" \
  --inventory-name chime-controls \
  --select 484,491,515,545,591,745 \
  --max-files 2 \
  --max-frames-per-file 4 \
  --allow-partial
```

Drop the caps and `--allow-partial`, and use a fresh output directory for the
full pass. The default control path is CPU-only; pass `--gpu` only when CuPy
acceleration is wanted.

## Post-processing

Validate and plot combined detector products:

```bash
pilot-proxy validate-products --run-dir <detector_run>
pilot-proxy chime-plot --run-dir <detector_run>
```

For ragged archives, inspect and combine a compatible channel subset:

```bash
pilot-proxy chime-combine --report --work-dir <detector_run>/_per_pilot
pilot-proxy chime-combine \
  --work-dir <detector_run>/_per_pilot \
  --drop 598,690 \
  --output-dir <subset_run>
```

## Order-safety constraint

The detector analyzer appends frames in file delivery order. `chime-scan` tags
downloads with their inventory position and buffers early completions, so
concurrent transfers cannot change `frame_index` or `relative_time_s`. The
defaults remain one worker and one staged file. Raise the bounds explicitly with
`--download-workers` and `--max-staged-files`.

## Inventory and resume compatibility

Completed `inventory.jsonl` files remain readable. This includes managed
inventories under the default `~/datatrawl-inventories` path and inventories
whose sidecar still records the canonical unpacked CHIME reader. The supported
commands choose the packed reader for detector scans and the complex reader for
control scans.

Do not resume survey state created by the earlier runtime. Start the survey with
a fresh `--name` or explicit `--out`. Reusing a completed `inventory.jsonl` is
safe; reusing its in-progress survey state is not.

Detector checkpoints bind the PilotProxy source tree. After source changes, an
in-progress detector checkpoint fails closed. Start a fresh output directory.
Completed current products remain valid inputs to `chime-combine`.

Compatibility fields and storage tokens are retained so completed inventories
and current products remain readable. They are data-format identifiers, not a
runtime dependency. Source integrations import the supported
`pilot_proxy.archive` namespace directly.

## Verification

The offline archive suite checks array layout, orchestration, resume behavior,
product combination, selection, metadata inference, and parity against the
earlier PilotProxy runner. It does not replace these operational gates:

- CUDA detector execution on a GPU node;
- authenticated streaming through `cadc-datatrail`;
- real-data parity against a separately validated CHIME run.

Run the archive checks with:

```bash
python -m pip install -r requirements/archive.txt
python -m pip install -e ".[archive,test]"
make archive-integration-check
```

## Main files

```text
requirements/archive.txt
src/pilot_proxy/archive/
src/pilot_proxy/archive/detector.py
src/pilot_proxy/archive/control.py
src/pilot_proxy/archive/packed_reader.py
src/pilot_proxy/archive/scan.py
src/pilot_proxy/archive/combine.py
src/pilot_proxy/chime/baseband_reader.py
tests/archive/
```

## Exact profile documents

Receiver profiles and detector-core profiles are current-only contract
documents. The receiver profile has one nested representation; the detector
core has one `kernel_contract` representation. Every authoritative field is
required, unknown structural fields are rejected, and serialization emits
exactly the accepted shape.

The receiver profile declares both the normalized coarse-channel center and the
normalized physical data-DC location. It also declares whether the adapter
reverses detector windows before the post-spectral-sense weight bank. There is
no implicit center-at-Nyquist or spectral-sense fallback.
