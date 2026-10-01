# CPU streaming SDR upgrade adapter

`pilot_proxy.integration.sdr_upgrade_streaming.StreamingDigitalAdapter` is a
stateful companion to `sdr_upgrade_adapter.py`. It consumes one contiguous
2 MHz complex digital stream and emits the same single-input detector geometry:
390625 Hz after 25/128 resampling, 16384 samples per frame, 128 rows of 128 samples,
three packed projectors and integer complex row projections. No hardware API,
physical RF mapping, array replication or thermal-null claim is supplied.

## Use and ownership

```python
from pilot_proxy.integration.sdr_upgrade_adapter import DigitalAdapterConfig
from pilot_proxy.integration.sdr_upgrade_streaming import StreamingDigitalAdapter

adapter = StreamingDigitalAdapter(
    DigitalAdapterConfig(2_000_000, mixer_hz=100_000., target_bin=0, sample_scale=5.),
    first_input_index=0,
)
for index, iq in contiguous_chunks:
    for frame in adapter.push(iq, input_start_index=index):
        consume(frame)
final = adapter.finish()
for frame in final["frames"]:
    consume(frame)
save_receipt(final["receipt"])
```

The index is mandatory and must equal `next_input_index`. Input chunks can have
any length, including a single sample or an empty complex no-op. Nonfinite,
non-complex, multidimensional or incorrectly indexed inputs fail before any
sample is consumed. One owner must serialize calls; the adapter is not thread
safe. Returned arrays belong to the caller and never alias retained buffers.

The fixed internal input block is 8192 samples (4.096 ms); this adds bounded
batching latency before an otherwise complete frame can be emitted. The retained
state consists of an 8192-input buffer, no more than 455 mixed history samples,
and one 16384-output frame buffer (at most 16383 filled between calls), plus
fixed FIR/weight arrays and counters. Work arrays are bounded by the fixed block.
Caller-owned input chunks and the returned list of complete frames use memory
proportional to their own sizes; callers seeking low peak memory should submit
bounded chunks and promptly consume frames. Processing throughput is unqualified.

## Continuity, boundaries and time

`finish()` drains the final short input block. Only complete FIR support is
accepted; no zero-padding ring-down enters the detector. Complete frames are
emitted and the incomplete frame is counted then discarded. A finished segment
rejects subsequent `push` or `finish` calls until an explicit reset.

`gap(missing_input_samples)` requires a positive known count. It first finishes
the old segment, then starts a new segment at the expected source index plus
that count. The caller must consume the boundary's returned frames and retain
its receipt. `reset(first_input_index=..., reason=...)` performs the same boundary
drain but declares a new source-index origin without inferring a missing count;
it can be used for an unknown transport loss, clock restart or new recording.
Reset after finish returns `None` because that old boundary was already emitted.

Every gap/reset restarts mixer phase, resampler lattice and frame origin. Old
FIR history and an incomplete frame are discarded; there is no artificial sample
or gap-spanning frame. Phase coherence across segments is explicitly unavailable.
Ordinary chunks within a segment preserve all of those states.

For segment-local untrimmed output index `j`, the sample center has exact rational
time `(source_origin*25 + j*128 - 4096) / 50000000` seconds relative to digital
source index zero. The first full-support index is 64, corresponding to 81.92 us
after the new segment start. Integer numerator/denominator fields are authoritative;
the floating seconds field is convenient. This is a digital sample-index label,
not proof of SDR clock continuity or a hardware timestamp. Segment IDs distinguish
explicit source-index resets that reuse earlier numerical indices.

## Arithmetic contract

Each segment uses the frozen batch FIR, mixer expression, scale, symmetric
four-bit ties-to-even quantizer, packed weights and CPU integer projection. Local
polyphase calls begin at an input index divisible by 128, preserving the original
resampling lattice. Every ordinary chunk partition feeds the same fixed internal
blocks, yielding deterministic, bitwise-equal streaming outputs for a given
segment on a fixed numerical runtime.

Batch and streaming floating outputs can differ at roundoff level because local
numerical kernels see different array extents. The qualification compares their
complex absolute error against `2e-12 * max(1, max(abs(input)))` and independently
checks selected scalar FIR convolutions. This is a tested engineering tolerance,
not a proof for every floating input. Phase uses the batch's absolute
segment-local mixer-index expression, with no chunkwise accumulated phase drift;
it inherits floating phase-evaluation precision limits for extremely long records.
Indices beyond `2**53 - 1` within a segment are refused.

Identical packed samples and integer projections are verified for each qualified
fixture. Each frame exposes `minimum_unclipped_quantizer_half_step_distance`,
the smallest distance of an unsaturated scaled component to a half-integer.
The release compares this margin with the largest measured scaled batch error.
Near an exact half-step, an ulp can change ties-to-even rounding between batch
and streaming; universal packed equality is not asserted. An adversarial half-step
test separately verifies that streaming remains invariant to chunk boundaries.

Zero target and reference powers give `undefined_both_zero` with NaN; a positive
target with zero references gives `infinite`. Neither is silently accepted as
noise, a finite statistic or a qualified detection. Saturated components are
counted. The ratio remains `2*target/(lower+upper)` with no weight-norm adjustment.

## Qualification

Run `PYTHONPATH=src python -m pytest tests/core/test_sdr_upgrade_streaming.py -q`
for continuity and boundary checks. Run
`PYTHONPATH=src python tools/validate_sdr_upgrade_streaming.py --output NEW_DIR`
for a non-overwriting release with six deterministic noise/tone/zero/saturation
records, five chunk partitions each (including all single-sample chunks),
independent scalar FIR and packed-integer projection comparisons, per-record
rounding-margin evidence, unit-test receipt, source snapshots and SHA256 manifest.

Real SDR transport, qualified radio noise/steady-signal references, clock/RF
mapping, physical filter transfer and live Pathfinder integration remain separate
measurements. This CPU companion establishes digital state handling only.
