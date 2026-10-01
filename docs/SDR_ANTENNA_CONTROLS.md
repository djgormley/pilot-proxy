# Finite antenna-connected controls

`tools/run_sdr_antenna_controls.py` prepares and, only with explicit launch
flags, executes five triplets of receiver-only ambient, TX-active zero-IQ and
commanded steady-tone records. All fifteen records are descriptive repeats.
They provide receiver/TX-state comparisons, not a thermal null, absolute RF
calibration, fitted F-law or statistically accepted observing policy.

The exact schedule is frozen before hardware. Python `random.Random(20260909)`
shuffles `[noise, txzero, tone]` independently for each of five triplets. Every
record retains its triplet and chronological identity. The schedule is not
redrawn after a failure or after inspecting IQ. Adjacent frames within a record
are not counted as independent acquisitions.

Each attempt retains exactly four million complex64 samples (two seconds) at
2 MHz, 500 MHz nominal LO and native RX gain 30. TX-active modes use requested
and expected readback gain 50. Tone peak component is 0.005 at +100 kHz digital
offset. The existing capture helper provides settling, fixed time intervals,
before/after state readbacks, transport counters and shutdown receipts. Its
initialization can perform internal gain calibration before the measurement.
TX-disabled and TX-active zero-IQ remain distinct states.

Analysis is frozen to the single-input upgrade adapter: input rate 2 MHz,
mixer +100 kHz, target bin zero, sample scale 500, 390625 Hz output and
16384-sample frames. Scale 500 is taken from the separate earlier ambient
adapter validation and its original metadata hash is bound in the new plan.
It is not tuned to these records. Clipping, undefined/infinite ratios, reference
imbalance and time variation must remain visible. No record-dependent scaling,
whitening, gain search or spectral selection is allowed by this protocol.

## Preparation and launch

Prepare only after the analyzer and orchestration checks are complete. The
`prepare` command requires a new empty output directory, an authenticated
existing capture build and one or more `--analysis-source` paths. Supply the
analyzer, its tests and every implementation file it uses for the declared
analysis. Preparation hashes those files and the existing capture helper,
worker, library, build inputs, orchestration source and tests. It then creates
all fifteen immutable attempt plans and binds their exact payload hashes in
one final study plan. Preparation never opens the SDR.
Exact copies of all bound inputs are retained under `source_snapshots/`, including
the prior scale-origin metadata and summary, worker/library and implementation
sources. The plan records Python, NumPy and SciPy versions and executable; launch
refuses a different recorded numerical runtime or changed source snapshot.

Review the resulting plan and retain its SHA256 independently of the directory.
`run` requires that exact `--expected-plan-sha256`, plus
`--hardware-authorized --transmit --rf-confined-authorized`. It refuses any
existing study launch/results or unexpected artifact in a prepared attempt.
Exclusive launch-file creation prevents concurrent invocations. Only one
finite capture runs at a time, with numerical-library thread limits set to one.

Any failed acquisition, missing or mismatched receipt, incomplete accepted
interval/count, unconfirmed cleanup, or changed frozen source stops the entire
sequence. Complete earlier records and the failed attempt are retained, while
all remaining attempts stay unattempted. There are no retries, shortened
intervals, changed gains, resumed schedules or extensions. `run-events.jsonl`
records each start/outcome; `run-receipt.json` records the completed/failed and
unattempted counts. The existing worker/parent deadlines remain 20/30 seconds,
with a five-second SIGTERM cleanup allowance.

If every attempt succeeds, the accepted total is 30 seconds. The ten TX-active
payloads total 23 seconds including fixed guards; this is not a measured RF-on
duration or a TX-power calibration. Failed and initialization intervals remain
separately recorded by the capture helper. No GPU operation, firmware change,
USB reset, fine-detector campaign edit or process modification is included.

## Hardware-free checks

`python -m pytest tests/testbench/test_sdr_antenna_controls.py -q` uses an
injected fake helper and non-executable fixture files. It verifies the fixed
randomized schedule, source and reviewed-plan binding, explicit launch flags,
single-use execution, fixed counts and stopping at failures including cleanup,
receipt identity, incomplete data, missing receipt and sources changing during
acquisition. It does not enumerate or open a radio.
