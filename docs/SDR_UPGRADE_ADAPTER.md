# Single-input SDR digital adapter

`pilot_proxy.integration.sdr_upgrade_adapter` prepares an existing contiguous
complex IQ record for the upgrade detector geometry on the CPU. The adapter
has no device or GPU API. It matches the **390625 Hz output rate, 16384-sample
frame, K=128 samples per projection, and L=128 rows per input**. It supplies one
input. Its FIR is a declared engineering filter, not an emulation or measured
calibration of the CHIME analog chain or polyphase filter bank.

## Digital contract

The caller must provide the nominal input rate of **2000000 Hz**, a signed
digital mixer frequency, an integer target bin, and a positive fixed sample
quantization scale. A positive digital tone is
`exp(+2j*pi*f*n/input_rate)`. Mixing multiplies the IQ by
`exp(-2j*pi*mixer_hz*n/input_rate)`, so the output digital frequency is the
input digital frequency minus the mixer frequency. Neither the API nor its
receipt assigns a physical RF frequency. LO readback alone does not establish
spectral orientation or an RF calibration.

The anti-alias stage interpolates by 25 and decimates by 128. Its FIR has 8193
real coefficients, a Kaiser window with beta 8.6, and a 165 kHz cutoff at the
50 MHz interpolated rate. The unity-DC coefficients are generated explicitly
with `scipy.signal.firwin`; `upfirdn` applies their interpolation gain of 25.
The receipt saves every coefficient and its canonical little-endian float64
SHA-256. The numerical acceptance checks require amplitude error below 5e-5
through 135 kHz and amplitude below 4e-5 from the output Nyquist frequency
(195312.5 Hz) to 25 MHz. The frequency-grid spacing is recorded. These are
digital transfer checks, not a bound on the analog receiver response.

The three natural projectors are target, lower digital reference, and upper
digital reference, at bins `[b, b-2, b+2]` with bin width 3051.7578125 Hz.
All three centers must lie within the qualified +/-135 kHz passband. Natural
weights have positive phase and projection uses `x dot conjugate(w)`. The
configuration refuses translated bands crossing the raw input Nyquist
boundary. It does not infer or reverse an unknown physical spectral sense.

Samples and weights use symmetric signed integer rails **[-7,7]**, nearest
rounding with ties to even, and packed two's-complement components: real in
the high nibble and imaginary in the low nibble. Weight scale is 7. Sample
scale is explicit and fixed by the caller; the adapter does not fit it to
each frame. Receipts count components that require clipping and save the
weight norms. Integer row projections use the existing CPU kernel reference.
The reported coarse ratio is exactly `2*sum(target power)/sum(reference
powers)` without weight-norm renormalization. Zero denominators remain
explicitly infinite or undefined rather than becoming noise observations.

## Timing and framing

For untrimmed resampler output index `j`, the sample-center time is
`(128*j - 4096)/50000000` seconds relative to the input record start. The
adapter keeps only samples with complete FIR support inside the record.
The first retained sample is at 81.92 microseconds. Only complete 16384-sample
frames are processed, and unused final samples are counted. The frame
duration is **0.04194304 seconds** and the row spacing is **0.00032768 seconds**.
The final frame endpoint is the next sample center, exclusive. Device clock
readback and absolute UTC calibration are separate from this nominal digital
time lattice.

This is a contiguous-record batch adapter. Calling it independently on
successive chunks loses FIR state and drops boundaries; that is not a live
streaming implementation. A future stream implementation needs stateful FIR
history, mixer phase and output-lattice state, and explicit transport-gap
handling before it can feed a live detector.

An ideal unfiltered, unquantized, equal-norm IID Gaussian input would produce
an F(256,512) coarse frame ratio for this one-input geometry. Filtering,
quantized weights, rounding, clipping and measured receiver behavior require
their own validation. One SDR stream does not become 2048 independent CHIME
inputs by copying or reweighting it.

## Reproduce the digital validation

Install the repository test extra for NumPy, SciPy and pytest, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/core/test_sdr_upgrade_adapter.py -q

python tools/validate_sdr_upgrade_adapter.py \
  --capture-root ../results/sdr_reference_transport_2026-09-09 \
  --output ../results/sdr_adapter_validation_2026-09-09
```

The second command only reads the previously accepted antenna-attached,
RX-only record. It verifies that input and its receipt against their release
manifest and frozen accepted-record hash. The tool sets numerical library
thread limits to one and refuses a nonempty output directory. It saves
synthetic signed-tone and alias-rejection checks, exact independent scalar
integer projection checks for every replay row, per-frame powers and timing,
filter coefficients, weights, source snapshots, and a hash manifest.
The synthetic-tone comparison divides desired-term power by the larger of
one squared projection-code unit and the largest other-term power. The
receipt labels it as a lower bound with that denominator floor and records
the actual other-term power separately. When both other powers are zero,
the finite comparison is not the exact desired-to-other power ratio.

The ambient replay uses a declared mixer of +100 kHz, target bin zero, and
sample scale 500. Its references are therefore +/-6103.515625 Hz at the
output, corresponding to input **digital** frequencies 93896.484375 and
106103.515625 Hz. These differ from the 2 MHz, K=128 engineering project's
references at 68750 and 131250 Hz. The two analyses are not interchangeable.
The replay does not subtract fitted tones, optimize its scale, or label the
antenna record as thermal noise. Its distribution is only a descriptive
engineering observation, with a single-input denominator.

## Next controlled-capture acceptance

Before using this path to claim physical calibration or to compare qualified
noise and stable-signal distributions:

1. Establish the physical frequency sign and offset using independently known
   positive and negative offsets. Record LO and sample-clock readbacks and
   their uncertainties, IQ ordering, gain, antenna or termination, bandwidth,
   and the complete capture interval and transport receipt.
2. Measure the combined analog and digital transfer across the target and
   reference supports, including leakage, image response and edge behavior.
   Freeze the filter coefficients, projector weights, gain and quantization
   scale from a separate calibration record. Check clipping and branch
   covariance on independent evaluation records.
3. Collect terminated or independently bounded noise, then stable signal
   plus that noise at predeclared levels, with settled hardware state and
   enough independent frames for the chosen confidence interval. An
   antenna-attached transmitter-off record alone is not a thermal null.
4. Evaluate these single-input data without claiming the array degrees of
   freedom. Keep analog gain calibration and physical signal-to-residual
   transfer measurements distinct from integer arithmetic agreement.
5. Validate stateful streaming and the production fine-detector handoff.
   CPU row-projection parity does not certify the fine policy or Pathfinder.

The current digital receipt closes preparation and arithmetic checks only.
It does not close the physical-calibration or live-Pathfinder tasks.
