# PilotProxy

<p align="center">
  <a href="https://github.com/WVURAIL/pilot-proxy/actions/workflows/tests.yml"><img src="https://github.com/WVURAIL/pilot-proxy/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="license: MIT"></a>
</p>

`pilot-proxy` evaluates a local-reference power ratio detector for Advanced Television Systems
Committee (ATSC) 1.0 digital television (DTV) signals. We use the narrow ATSC
pilot tone as a measurable proxy for the broadband data shelf. This provides a
narrow-band observable when the shelf is below the instantaneous noise level.
The repository includes a standalone CUDA detector and a GNU Radio validation
testbench.

The package has two main workflows:

1. **Standalone synthetic/testbench mode** for clean (noise-free) ATSC
   generation, quantization, CUDA-kernel evaluation, and controlled SNR sweeps.
2. **CHIME real-data mode** for baseband HDF5 data. The recommended archive-scale
   entry point is `pilot-proxy chime-scan`. This command runs the PilotProxy
   analyzer through the bundled archive engine, then attempts to combine the
   per-pilot products into the canonical CHIME outputs.

The detector core is telescope-independent. A receiver integration supplies the
metadata and arrays needed to satisfy the CUDA kernel contract; it does not
change that contract.

---

## Choose your workflow

The repository supports several workflows with different dependencies. Choose
the row that matches the result you need:

| Goal | Guide to follow | Needs GPU? | Needs archive extras? |
| --- | --- | :---: | :---: |
| Run the frozen CHIME archive inventory on the local WSL workstation | [docs/LOCAL_PROCESSING.md](docs/LOCAL_PROCESSING.md) | Yes | Yes |
| Run a bounded CHIME archive job on remote CANFAR (`chime-scan`) | [docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md) | Yes | Yes |
| Check the package installs and the CLI loads | This README: minimal CPU-only smoke test (below) | No | No |
| Generate / audit synthetic ATSC | This README: standalone testbench (needs GNU Radio) | No | No |
| Run the CUDA detector / SNR evaluation | This README: standalone CUDA path | Yes | No |
| Publication SNR sweeps without a GPU | This README: `pilot-proxy evaluate-snr --input-iq <capture> --detector-backend cpu-reference --noise-source python` (the capture is generated once with GNU Radio) | No | No |

For the frozen local archive run, begin with
[the local processing guide](docs/LOCAL_PROCESSING.md); it owns the local
revision, gate, staging, launch, resume, and closeout procedure. For a bounded
remote A100 run, begin with the [CANFAR guide](docs/CANFAR_RUNBOOK.md). The
standalone sections below are not prerequisites for either archive workflow.
`setup_env.sh` performs its own sanity checks, so the README smoke test is also
optional on those paths.

---

## Environment

We run the standalone workflows in one Python virtual environment. The virtual
environment can be in any writable, persistent directory; it does **not** need
to be inside the source checkout. The selected Python must include `venv` and
`ensurepip`. On a minimal Debian or Ubuntu installation, install
`python3-venv`, or set `PYTHON_BIN` to a Miniconda or session Python that already
provides them.

```bash
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/pilot-proxy-archive}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$(dirname "$VENV_DIR")"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
```

Shared and image-managed Python installations may be read-only or marked as
PEP 668 externally managed. In those environments, install into the virtual
environment rather than using bare `pip`. The `--system-site-packages` option
keeps a session image's CuPy and CUDA packages visible to the GPU workflows.
Activate the environment in every **new** session. It persists only when
`VENV_DIR` is on persistent storage, such as `/arc` on CANFAR. The CANFAR
runbook uses `setup_env.sh`, which creates and recreates its own configured
environment; do not create that environment by hand.

---

## Fresh clone and minimal CPU-only smoke test

This smoke test checks the Python package, CLI entry point, and reference
detector metadata. It does **not** require GNU Radio, CADC
credentials, CUDA, or CHIME HDF5 data. Run it inside the
[environment](#environment) above before using a standalone workflow.

```bash
export REPO_DIR="${REPO_DIR:-$PWD/pilot-proxy}"
git clone https://github.com/WVURAIL/pilot-proxy.git "$REPO_DIR"
cd "$REPO_DIR"
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[test]"

pilot-proxy --help
python -m pytest tests/test_cli.py -q
pilot-proxy check-profile \
    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
pilot-proxy check-layout \
    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
```

### Repository test targets

The smoke test runs only `tests/test_cli.py`. To run the repository's Python
test target, install the test extra and use:

```bash
python -m pip install -e ".[test]"
make test-python
```

The `test` extra includes `h5py`, Matplotlib, and PyYAML because the checked-in
tests exercise the CHIME HDF5 adapters, plotting paths, and YAML configuration
loaders. These tests generate or mock their inputs, so they do not require CHIME
data files. CADC-backed checks are skipped when archive credentials are not
available.

`make test` runs `test-kernel` before `test-python`; therefore, it is **not** a
CPU-only target. Use it on a CUDA build host after checking both the driver and
compiler:

```bash
command -v nvidia-smi && nvidia-smi --query-gpu=name,compute_cap --format=csv
command -v nvcc && nvcc --version
```

If `nvidia-smi` is unavailable (it ships with the NVIDIA driver; on WSL,
through the Windows driver's WSL CUDA support), use `make test-python` instead
of `make test` or `make test-kernel`. The
`make release-check` target adds CPU C/C++ reference checks, profile and layout
checks, and runtime-bundle validation. It requires a C++ compiler but not a GPU.

Use `make test-gpu DETECTOR_WINDOW_SAMPLES=128` for the required GPU gate,
including temporary K=64 and K=128 builds and archive interrupt/resume tests.
Missing GPU prerequisites fail this gate. See [test coverage](docs/TEST_COVERAGE.md)
for the component map, GNU Radio checks, and remaining validation limits.

---

## Contents

The repository separates the detector, integration code, configuration, and
operating documentation as follows:

- `cuda/` - CUDA kernel, public C header, CPU C++ reference, and C++ tests.
- `src/pilot_proxy/` - Python package for kernel loading, detector geometry,
  reference channelization, DTV unit conversion, CHIME adapters, and testbench
  workflows.
- `src/pilot_proxy/testbench/` - GNU Radio ATSC generation, waveform audit,
  AWGN generation, quantization, and SNR evaluation.
- `src/pilot_proxy/archive/` - bounded inventory, staging, streaming, CHIME
  sources and readers, detector and control analyzers, scan, and combine support.
- `weights/` - prebuilt ATSC reference detector weights for
  `detector_window_samples=128`, `num_weight_terms=3`,
  `skipped_guard_bins=1`, `reference_offset_bins=2`, physical channels 14-36,
  and 4+4 bit samples.
- `configs/` - detector-core, receiver-profile, and stream-map JSON examples for
  standalone and integration workflows.
- `scripts/setup_env.sh` - one-shot CANFAR archive setup script for the
  integrated CHIME workflow.
- `scripts/launch_gpu_session.py` - launch, reuse, or tear down a CANFAR CUDA GPU
  notebook session (skaha / `canfar` client) for the detector path.
- `INTEGRATION.md` - detailed standalone archive integration notes.
- `docs/METHOD_SPEC.md` - equation-first method contract for CHIME products.
- `docs/PRODUCT_SCHEMA.md` - the only supported per-pilot product contract.
- `docs/PER_PILOT_PRODUCT_FIELDS.md` - detailed shared-field reference.
- `docs/FINE_REDUCTION_PRODUCTS.md` - fine-measurement and diagnostic fields.
- `docs/CHIME_RUN_WORKFLOW.md` - staged-data workflow (`chime-run`) for
  already-staged HDF5 directories; choose the local or CANFAR guide below for
  archive-scale `chime-scan` runs.
- `docs/DATA_PRODUCTS.md` - emitted file, array, and table definitions.
- `docs/LOCAL_PROCESSING.md` - authoritative local WSL archive procedure.
- `docs/CANFAR_RUNBOOK.md` - bounded CANFAR operating procedure.
- `docs/KOTEKAN_INTERFACE_PREP.md` - runtime-bundle and Kotekan handoff notes.
- `docs/DESIGN_DECISIONS.md` - recorded detector and integration decisions.
- `docs/PilotProxy_DS001_Data_Sheet.tex` - formal data sheet (build to PDF).
- `docs/PilotProxy_UG001_User_Guide.tex` - formal user guide (build to PDF).
- `examples/quickstart.sh` - standalone release sanity-check workflow (CUDA +
  GNU Radio; environment-specific defaults --- override `SM`, `CUDA_PYTHON`,
  `GNURADIO_PYTHON`).

We commit documentation sources and the small, explicitly named artifacts
needed for reproducibility: shipped weight banks and manifests, frozen test
fixtures and dated scientific-provenance snapshots.
Ordinary run products, rebuilt plots, formal-document PDFs, generated TeX
facts, and CUDA shared libraries are ignored build artifacts. The active and
historical weight-bank boundary is documented in `weights/README.md`. The
2026-07 manuscript and its artifact policy are kept in the history
(`paper/README.md@30e50c2`).

Built wheels include the shipped receiver profiles, stream map, weight bank, and
weight manifest. The CUDA shared library is architecture-specific and is not
included. Before running the GPU detector, build the library from a source
checkout and stage it under ``~/.cache/pilot_proxy/libfstatistic.so``.

---

## Setup for the CHIME / CANFAR workflow

The integrated workflow uses this repository and the pinned archive client in
`requirements/archive.txt`. The `scripts/setup_env.sh` script recreates a
virtual environment and installs `.[archive,test]`. It checks the bundled
source, reader, and analyzer classes directly.
On a GPU node, it also requires `nvcc`, builds the CUDA kernel, and checks that
the kernel loads. Because the script **removes and recreates** the target
environment, do not set `VENV_DIR` to an environment you need to preserve. The
script refuses the home directory, the checkout, overlapping directories,
and unowned non-empty targets. On the first guarded rerun of a genuine virtual
environment created by an older checkout, explicitly acknowledge adoption with
`PILOT_PROXY_ADOPT_LEGACY_VENV=1`; subsequent rebuilds use the durable ownership
record beside the environment and remain retryable after an interruption.

The full procedure is in
[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md#environment-setup).
It gives the exact `setup_env.sh` invocation, a manual setup that does not
recreate an environment, GPU-session launch, Harbor registry credentials,
required inputs, and bounded run sequences. Integration-specific details are in
[INTEGRATION.md](INTEGRATION.md#setup).

For a manual installation in an active virtual environment:

```bash
python -m pip install -r requirements/archive.txt
python -m pip install -e ".[archive,test]"
```

---

## Receiver integration contract

A receiver integration translates telescope data into the detector's fixed
input contract. It provides:

- `receiver_profile.json` describing RF band, channelizer geometry, spectral
  sense, frame size, input streams, quantization policy, bin ENBW, and pilot
  capture efficiency;
- optional `stream_map.json` describing the input-stream ordering;
- channelized complex input arrays or packed detector matrices;
- a generated weight bank built from the receiver profile.

After this translation, the CUDA kernel sees only:

- packed int4 detector rows;
- packed int4 weights;
- uint64 target/reference powers.

CHIME run products record the detector contract in
`run_config.json` and `stats.json` with:

- `schema_version = pilotproxy_detector_contract_v1`;
- K, weight-term, skipped-guard, and reference-offset geometry;
- packed input and uint64 accumulator metadata;
- the all-row summation rule;
- the positive-excess mask policy;
- reference-placement summary metadata.

Validate integration metadata with:

```bash
pilot-proxy check-profile   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json

pilot-proxy check-layout   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json   --stream-map configs/stream_maps/chime_feed_pol_example.json
```

The shipped receiver profiles have different roles and evidential status:

- `reference_800mhz_pfb.json` is the single-stream detector-coordinate reference
  profile used for shipped reference weights and tests.
- `chime_dtv_fengine.json` is the target CHIME DTV adapter profile used by the
  current integration: 2048 feed-polarization streams, inverted spectral sense,
  descending RF channel order, and `frame_size_samples=16384`. The 16384-sample
  frame is the target for the CHIME F-engine upgrade rather than a claim about the
  currently deployed frame. The profile is frame-verified
  (`profile_status` in the file); still verify these values
  against the data product used for an operational run.
- `chord_dtv_fengine.json` and `chord_pathfinder_dtv_fengine.json` are the
  CHORD / CHORD-pathfinder adapter profiles for the kotekan (`chord` branch)
  integration. Both share the CHORD channelization: 3.2 GS/s ADC, first
  Nyquist zone, 16384-point PFB, 8192 coarse channels of exactly 195312.5 Hz,
  upright spectral sense, ascending channel order, channel RF center =
  `coarse_channel_index * 195312.5 Hz` (the kotekan `freq_id` namespace) --
  and differ only in stream count: 1024 dish-polarization streams (512
  dishes) for full CHORD, 128 (64 dishes) for the pathfinder. The 8192-sample
  detector frame is exactly one kotekan GPU frame (41.94304 ms) and, at
  K=64, gives the frozen 128-windows-per-stream fine-reduction geometry.
  The ATSC 14-36 pilots land in CHORD channels 2408-3084; channels 14 and
  21 are the adaptive reference cases (ch14's upper reference wraps the
  frame edge; ch21's lower reference shifts off the forbidden DC bin,
  wrapping the frame edge). These profiles declare
  `metadata.channel_id_map`, so exported runtime bundles carry a populated
  `chord_channel_id` for kotekan first-frame profile selection. Both are
  marked `example_requires_data_product_verification`: the grid matches the
  kotekan chord-branch source, and the frame convention must still be
  verified against CRS F-engine data before an operational run
  (`docs/KOTEKAN_INTERFACE_PREP.md`).

---

## Weight bank

We generate the default CHIME DTV weight bank from the receiver and detector
profiles:

```bash
pilot-proxy make-weights   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json   --detector-core-profile configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json   --physical-channel-range 14:36   --weight-coordinate-system post_spectral_sense_normalization   --output weights/chime_dtv_weights_k128.bin
```

By default, the detector looks for:

```text
weights/chime_dtv_weights_k128.bin
```

Leave the weights file at the default path, or pass the path explicitly to a
command that consumes it:

```bash
pilot-proxy list-channels --weights-path weights/chime_dtv_weights_k128.bin
pilot-proxy chime-run --weights-path weights/chime_dtv_weights_k128.bin --help
```

`list-channels` reports the reference placement for each physical channel.
Adaptive cases also print an explanatory `NOTE` without changing the CSV
output. The shipped banks use `reference_offset_bins=2` and
`skipped_guard_bins=1`. We skip one fine bin on each side of the target and use
the next fine bin as the lower or upper reference. Placement works in the
periodic normalized-frequency coordinate of the coarse-channel FFT, so a
reference requested past the baseband-frame origin (nu = 0) wraps modulo 1;
the wrapped weights are bit-identical to the unwrapped ones. The manifest
records this as `edge_wrapped`, a name inherited from the earlier
center-at-Nyquist frame convention: under the verified center-at-DC CHIME
profile, nu = 0 is the coarse-channel *center* (the forbidden DC tone), not the
channel edge, and the manifest's `frame_origin_description` says which. A
reference that collides with the forbidden coarse-channel DC tone moves one bin
farther from the target, and the manifest records a placement warning. A
target-DC collision stops weight generation because the target cannot be moved.
Separately, the manifest's `*_crosses_channel_edge` fields (and a
`list-channels` `NOTE`) report a reference requested beyond +-fs/2 of the
channel center: it aliases to the opposite channel edge and samples the
channelizer roll-off. Placement and detection arithmetic are unchanged by that
note.

For the shipped ATSC 14-36 CHIME bank, physical channel 14 is the only adaptive
case: its pilot sits one fine bin above the coarse-channel center (DC), so its
lower reference lands one bin below DC with the forbidden DC tone in the
skipped guard between them (`forbidden_tone_in_skipped_guard`), and
`list-channels` prints the corresponding placement `NOTE`. No shipped channel
collides with the forbidden DC tone or requires a DC shift; the remaining
channels use the nominal reference placement. Channel 21's upper reference is
the one that crosses the physical coarse-channel half-width (+196.7 kHz against
195.3 kHz); `list-channels` prints a `NOTE` for it.

For deployment, export the profiles and weights as a compact runtime bundle,
then validate that bundle:

```bash
pilot-proxy export-runtime-weight-bundle   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json   --detector-core-profile configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json   --weight-coordinate-system post_spectral_sense_normalization   --physical-channel-range 14:36   --output-dir generated/deploy/chime_dtv_k128

pilot-proxy validate-runtime-weight-bundle   --bundle-dir generated/deploy/chime_dtv_k128
```

---

## CUDA kernel

On a GPU host:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

The build uses the integer `SM` form of the compute capability. Remove the
decimal point to obtain it:

| Compute capability | `SM` |
|---:|---:|
| 8.0 | 80 |
| 8.6 | 86 |
| 8.9 | 89 |
| 9.0 | 90 |

The Python GPU path uses CuPy. In the integrated CHIME/CANFAR workflow,
`setup_env.sh` resolves CuPy through the bundled archive accelerator API. For a standalone
CUDA 12.x workflow, use `python -m pip install -e ".[cuda]"`. The `cuda` extra
installs `cupy-cuda12x`; use the corresponding `cupy-cudaXXx` package when the
runtime is not CUDA 12.x.

Both `setup_env.sh` and the `make` targets detect `SM` from the first GPU visible
to `nvidia-smi`. Pass `SM=<arch>` only when detection fails or when
cross-compiling for another architecture. The build records the architecture
and kernel configuration in its build stamp. If either value changes, the next
build recompiles the kernel.

### CUDA toolchain (`nvcc`)

The kernel build requires the CUDA compiler, `nvcc`. Check that it is on
`PATH`:

```bash
command -v nvcc && nvcc --version
```

On some CANFAR CUDA images, `nvcc` is installed under `/usr/local/cuda/bin` but
that directory is not on `PATH`. If `command -v nvcc` returns a path such as
`/usr/bin/nvcc`, use `make build-kernel` without setting `NVCC`. Otherwise,
locate the compiler:

```bash
find /usr/local -maxdepth 4 -path '*/bin/nvcc' -type f 2>/dev/null
```

Then add its directory to `PATH`:

```bash
export PATH=/path/to/cuda/bin:$PATH
```

Alternatively, pass the compiler path directly. `cuda/Makefile` honors
`NVCC=`:

```bash
make build-kernel NVCC=/path/to/cuda/bin/nvcc
```

If `nvcc` is absent, use a CANFAR image that includes the CUDA *toolkit*, such as
`skaha/astroml-cuda`, or install a toolkit that matches the runtime. A
runtime-only image may run CuPy while still lacking the compiler needed for this
kernel. In that case, `make test` and `make test-kernel` stop at the `nvcc`
step. On a CPU-only host, use `make test-python` or `make release-check`; neither
target requires a GPU.

Build and stage the kernel:

```bash
make build-kernel
```

This builds `cuda/libfstatistic.so` and stages a copy to:

```text
~/.cache/pilot_proxy/libfstatistic.so
```

After building, load the library through Python and print its compile-time
contract:

```bash
PYTHONPATH=src python - <<'PY'
from pilot_proxy.kernel import FStatKernel
kernel = FStatKernel()
print(kernel.specs.as_descriptive_dict())
print(kernel.features.as_dict())
print(kernel.version.as_string())
PY
```

This output describes the CUDA kernel rather than the complete receiver frame.
`detector_window_samples=128` is `K`. It is both the detector-row length and the
number of coefficients in each packed weight vector: target, lower reference,
and upper reference. It is **not** the receiver frame length.

The receiver profile supplies `frame_size_samples`. Both shipped profiles set
`frame_size_samples=16384`, so the tested configuration gives:

```text
windows_per_stream = frame_size_samples / detector_window_samples
                   = 16384 / 128
                   = 128

detector_rows_per_frame = num_input_streams * num_selected_channels * windows_per_stream
```

For the target CHIME profile and one selected coarse channel, the row count is:

```text
2048 feed-pol streams * 1 selected channel * 128 windows_per_stream = 262144 rows
```

Thus, one frame in this profile is packed as `(262144, 128)`, and a batch is
packed as `(frames_in_chunk, 262144, 128)`. The kernel sums the target and
reference powers over every detector row before it forms one local-reference power ratio for the
frame or block. This geometry is conditional on the profile's 16384-sample
frame. Use `check-layout` to print and validate the geometry derived from the
profile:

```bash
pilot-proxy check-layout \
    --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
    --stream-map configs/stream_maps/chime_feed_pol_example.json
```

This command checks the configuration rather than a CHIME file. The CHIME
profile records the baseband-frame verification performed in 2026-07, but its
planned F-engine frame size, stream count, and ordering must still be compared
with the operational data product for each run.

The shared library is loaded by PilotProxy and is not executed directly.

Run the compiled CUDA/C++ regression tests with:

```bash
make test-kernel
```

---

## CHIME archive workflow

To retain a fixed staging bound during an archive-scale run, we use
`pilot-proxy chime-scan`. It runs the PilotProxy analyzer through the bundled
archive engine, writes one product for each selected pilot with usable input,
and attempts to combine those products into the canonical CHIME outputs. If no
`(event, frame-in-file)` identity is common to every completed pilot, the scan
preserves the per-pilot products and defers stacking until a compatible channel
subset is chosen with `pilot-proxy chime-combine`.

The archive command sequence is:

```bash
pilot-proxy chime-survey --name chime-pilots --freq-ids <freq_ids>
pilot-proxy chime-inventory --inventory-name chime-pilots
pilot-proxy chime-scan --inventory-name chime-pilots --output-dir <detector_run>
pilot-proxy chime-control-scan --inventory-name chime-controls --select <freq_ids> --output-dir <control_run>
```

Two constraints determine how we run it:

- **Selection uses the CHIME `freq_id` coarse-channel namespace** instead
  of ATSC physical-channel numbers. For `--source cadc-datatrail`, omitting `--select`
  scans each `freq_id` present in the inventory. The command prints that set
  before staging begins. Pass `--select` to restrict it. The
  `--inventory` and `--inventory-name` flags also let `chime-scan` infer the
  archive source. The 23 `freq_id`s for the default ATSC 14-36 pilot range are
  listed in
  [docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md#selection-convention); `844` is
  the single-channel smoke-test value for the ATSC 14 pilot.
- **Use `chime-scan` for the supported path.** Downloads may run concurrently,
  while the analyzer still receives files in inventory order. The defaults are
  one worker and one staged file; use `--download-workers` and
  `--max-staged-files` to raise the bounded prefetch depth.

The following documents give the complete selection rules, local and
CADC/CANFAR sequences, order constraint, and post-processing commands:

- **[docs/RERUN_PARAMETER_REGISTER.md](docs/RERUN_PARAMETER_REGISTER.md)** --- frozen scientific and product settings.
- **[docs/VALIDATION_GATES.md](docs/VALIDATION_GATES.md)** --- mandatory local launch gates.
- **[docs/LOCAL_PROCESSING.md](docs/LOCAL_PROCESSING.md)** --- measured local workstation profile and sole production command.
- **[INTEGRATION.md](INTEGRATION.md)** --- standalone archive integration contract.
- **[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md)** --- alternate bounded remote workflow.

---

## Standalone synthetic/testbench workflow

For the dissertation's staged, paired fixed-versus-floating sensitivity
measurement at the current 2048-stream geometry, see
[`docs/CURRENT_GEOMETRY_SENSITIVITY.md`](docs/CURRENT_GEOMETRY_SENSITIVITY.md).
The compact commands below remain useful for waveform and detector smoke
tests; their reduced examples are not publication sensitivity estimates.

<p align="center">
  <img src="docs/figures/validation_flow.svg" alt="Standalone validation flow: ATSC generation, waveform audit, AWGN injection, reference PFB, packed int4 matrix, CUDA local-reference power ratio kernel, report" width="560">
</p>

This is the same diagram the Data Sheet and User Guide carry; all three
render from the single source `docs/figures/validation_flow.tikz` (see the
header of that file for the one-command svg re-render).

When GNU Radio is installed only in the system Python, the standalone workflow
uses two interpreters:

- GNU Radio Python for ATSC generation and GNU Radio AWGN;
- CUDA Python for CuPy and the CUDA local-reference power ratio library.

Generate an ATSC waveform without injected noise:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3   -m pilot_proxy.testbench.generate_atsc_signal   --output-iq generated/atsc/atsc_8vsb_complex64.cfile   --num-iq-samples 600000
```

Audit the generated waveform:

```bash
PYTHONPATH=src python   -m pilot_proxy.testbench.audit_atsc_signal   --input-iq generated/atsc/atsc_8vsb_complex64.cfile   --fail-on-quality
```

The audit writes `generated/atsc/atsc_waveform_audit.json` and reports five
measured properties: pilot frequency error, pilot level relative to the data
shelf, occupied bandwidth, shelf flatness, and channel-edge rolloff. A waveform
that meets all configured bounds prints `quality_passed=True (5/5)` and the
margin for each check. With `--fail-on-quality`, the command exits nonzero when
any bound is not met. This result validates the generated waveform against those
five checks; it does not validate a receiver implementation.

Pack detector input:

```bash
PYTHONPATH=src python   -m pilot_proxy.testbench.quantize   --input-iq generated/atsc/atsc_8vsb_complex64.cfile   --physical-channel 14   --frame-size-samples 16384   --num-input-streams 1
```

Run a small SNR evaluation:

```bash
PYTHONPATH=src python   -m pilot_proxy.testbench.evaluate_snr   --input-iq generated/atsc/atsc_8vsb_complex64.cfile   --physical-channel 14   --frame-size-samples 16384   --num-input-streams 1   --requested-data-shelf-snr-db -26   --noise-trials 10
```

This command writes `generated/dtv_snr_eval/dtv_snr_summary.csv`. The
`snr_error_db_mean` and `snr_error_db_std` columns summarize estimated shelf SNR
minus the injected or measured reference. For the packed fixed-point path,
`cpu_gpu_abs_diff_max` measures agreement between the CUDA detector and the CPU
reference and is expected to be zero. Ten noise trials provide a smoke test,
not a final uncertainty estimate. Increase `--noise-trials` when the uncertainty
of the reported mean and standard deviation matters.

---

## Figures

We use LaTeX-style fonts for figures. By default, Matplotlib renders Computer
Modern with mathtext, which does not require a TeX installation. Set
`PILOT_PROXY_USE_TEX=1` to use external TeX when `latex`, `dvipng`, and the
`cm-super` fonts are installed. CI leaves this option disabled. Figure writers
emit 300 dpi PNG files by default. Set
`PILOT_PROXY_FIGURE_FORMATS=png,pdf` to write a vector PDF with the same stem.

## Build documentation

Generated PDFs are ignored by git. Build them locally with:

```bash
make docs        # latexmk; scratch in docs/auxil/, PDFs in docs/out/
make docs-specs-check  # validate canonical documentation facts without writing
```

The package set tested on Debian/Ubuntu is:

```bash
sudo apt-get install --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-pictures lmodern latexmk
```

Figures rendered with `PILOT_PROXY_USE_TEX=1` also require
`dvipng cm-super ghostscript`. The tested CANFAR session images do not provide
TeX or root access, so build the documentation outside the session.

---

## Commit hygiene

Before committing, remove generated products and local build artifacts, then
check the remaining tree:

```bash
make release-clean
make commit-check
```

## Inventory migration note

Completed `inventory.jsonl` files remain readable, including inventories under
the default `~/datatrawl-inventories` compatibility path. Start an old
in-progress survey with a fresh `--name` or `--out`; its saved survey state is
not resumed across the runtime change. Detector checkpoints fail closed after
source changes. [INTEGRATION.md](INTEGRATION.md#inventory-and-resume-compatibility)
records the full compatibility contract.

## Citation

We provide a machine-readable software citation in
[`CITATION.cff`](CITATION.cff).

Use `pilot-proxy --version` to report the installed package version.

Survey provenance, legacy compatibility, exclusions and interrupted-run recovery
are documented in [the survey trust contract](docs/survey_trust.md). The
[September 8 fine-gain correction](docs/evidence/fine_gain_report_correction_2026-09-08/README.md)
keeps corrected reports separate from the original evidence.
