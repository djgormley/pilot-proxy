Historical: written before the kotekan pull requests 1669 to 1681 merged.

# Kotekan Interface Preparation

This note separates the tested PilotProxy runtime bundle from the proposed
Kotekan integration. The repository exports and validates the bundle described
below, and the CUDA library exposes a device mask API. It does not contain a
Kotekan stage; the CHORD stage lives in the kotekan repository (see Section 7).
For CHIME, the state machine, metadata binding, and frame alignment remain
interface requirements rather than implemented behavior. For CHORD, Section 7
records the values that resolve this note's open questions against the kotekan
`chord` branch, and the companion kotekan patch series implements the stage
(`cudaPilotProxyDetector`) against this bundle contract.

## 1. Assumptions

The proposed stage uses these assumptions:

- We deploy the same detector code on each participating node.
- The Kotekan frame metadata provides an integer CHIME channel identifier.
- We bind the node to a channel profile after reading its first frame.
- A channel absent from the runtime bundle disables pilot detection on that
  node.
- A matching pilot channel selects exactly one weight profile.
- Detector samples and weights use packed complex int4 values.
- The detector produces one binary rejection decision for each aligned detector
  frame.

Only the packed detector, runtime bundle, and bundle validator are tested in
this repository. In particular, the exact Kotekan channel field and its
stability during a run have not been verified.

The Python CHIME path currently reads exact `uint64` target and reference powers
back to the host and forms the norm-corrected positive-excess decision there.
The CUDA library also exposes the rational-half-threshold mask API needed to
form the same comparison on the device.

## 2. Runtime State Machine

The first accepted frame determines whether this detector instance runs:

```text
INIT
  -> WAIT_FOR_FIRST_FRAME
       -> DISABLED   (channel identifier has no bundle profile)
       -> RUNNING    (channel identifier selects one bundle profile)
```

In `WAIT_FOR_FIRST_FRAME`, the stage reads the integer channel identifier and
looks it up in `pilot_profiles.json`. A miss enters `DISABLED`. A match enters
`RUNNING` with the selected byte offset, weight-profile pointer, and detector
contract.

This state machine assumes that the identifier does not change. If Kotekan can
change it within one stage lifetime, the integration must define whether to
reject that transition, drain and reinitialize, or select a new profile at a
frame boundary.

## 2a. Result Handoff (proposed)

The detector produces one binary rejection decision per aligned frame, well
ahead of the correlator's processing of the same frame: on the v1 kernel, GPU
resource measurements showed the detector completing with a large margin
relative to the correlator's per-frame cadence. The added fine-power stage
of kernel core 2.1.0 measured 1.4 ms per 2048-stream frame on an A100
(x30 margin against the 41.9 ms cadence; `test_fine_powers_gpu.py`
report). The combined measurement is now direct: the fused kernel (core
2.2.0) runs the complete samples-to-fine-and-coarse-powers datapath in
0.77 ms per 2048-stream frame (x55 margin; 2.8x faster than the composed
three-launch chain, and binding the row-sum debug tap adds ~5 us), and
the fused fine-mask candidate (core 2.3.0, decision epilogue included)
measures 0.90 ms (x47 margin; the epilogue itself costs +0.13 ms) ---
A100, `test_fused_fine_gpu.py` / `test_fused_mask_gpu.py` rate reports,
2026-08-05. The implemented candidate therefore occupies ~2% of the frame
cadence end to end; that timing result does not make an uncalibrated decision
an active observing policy.

The intended post-calibration enqueued decision is the fine designated-set
order-statistic CFAR computed on the device after the exact int32 row sums:
padded window-axis FFT, incoherent feed sum, null-bulk order statistic,
designated-set compare, and one mask bit per aligned frame. The queue design
below is unchanged. The current Python CHIME/archive path instead records the
norm-corrected coarse positive-excess decision as `reject_mask`; its stored fine
spectrum and threshold exceedances are diagnostics, not the active rejection
decision. The gating engineering
requirement --- a deterministic verification FFT implemented identically
in CUDA and the reference --- is frozen as fxfft256 v1; section 5 lists
the frozen artifacts and the bit-for-bit rule. The bundle
grows
per-channel measured-line anchors, designated-set widths, and CFAR
threshold multipliers, with their calibration provenance (epoch lists,
quantiles, source product hashes); anchors need on-epochs only, while
multiplier depth follows the verified-null program. The survey's recorded
null_power_ratio flag is unchanged mid-survey; any active-decision change lands at a
survey epoch boundary.

The proposed handoff exploits that asymmetry with a per-node FIFO queue:

- the detector stage enqueues each frame's decision as soon as the kernel
  completes (producer, fast);
- the correlator pops the queue when it finishes processing the same frame
  (consumer, slow), so the decision is synchronized to exactly the
  integration it masks;
- because the producer strictly leads the consumer, queue depth stays
  shallow and the decision adds no latency to the correlator path --- the
  detector's compute time is hidden entirely inside the correlator's.

The integration must still define the queue's bound and overflow policy (a
stalled consumer should drop toward `DISABLED` rather than block the
correlator), the frame-identity key used to match a popped decision to its
integration, and where the synced decision is recorded in the correlator
output. One named commissioning measurable falls out of the queue for
free: a per-10-s any-fired counter over the enqueued mask bits yields the
post-integration flagging exceedance (`exceedance_10s`) that archived
baseband snapshots cannot measure (see the calibration export's
`file3_substitution` provenance note) --- it should be recorded from the
first hours of deployment. These are interface requirements, not implemented behavior; see the
system-context figure shared by the Data Sheet and User Guide
(`docs/figures/system_context.tikz`).

## 3. Runtime Bundle Format

PilotProxy exports these five files:

```text
detector_contract.json
pilot_profiles.json
weights.bin
weights.manifest.json
sha256sums.txt
```

`detector_contract.json` records the detector geometry, coordinate convention,
mask rule, and input preprocessing. `weights.bin` concatenates one int8-packed
`(3, K)` weight profile per selected physical channel.
`weights.manifest.json` records the profile shape, source geometry, selected
channels, coordinate convention, reference placement, and hashes.

`pilot_profiles.json` maps each physical DTV channel to a profile index, byte
offset, byte count, calibration fields, and positive-excess rational threshold.
When the receiver profile declares `metadata.channel_id_map` (namespace plus
integer offset from `coarse_channel_index`), the exporter also populates
per-row receiver channel identifiers: `receiver_channel_id`,
`receiver_channel_id_namespace`, and -- for the `chord_freq_id` namespace --
the `chord_channel_id` alias consumed by the kotekan stage. The bundle
validator range-checks these fields, enforces uniqueness, and rejects
alias/receiver-id mismatches.
Each profile also carries a `fine_calibration` block (decision version,
anchor bin, designated half width, 256-bit bulk mask as four hex uint64
words, CFAR rank, Q16 multiplier, provenance) for the kernel core 2.3.0
mask entry; the exporter writes it with `status = "pending_campaign"`
and the validator range- and consistency-checks a `calibrated` block
(designated and guard bins excluded from the bulk mask, rank below the
bulk population).
The current exporter writes `chime_channel_id = null`; a CHIME/Kotekan metadata
mapping must populate and validate that field before first-frame selection can
be implemented. Thus, the current bundle supports physical-channel identity but
does not yet constitute a live Kotekan channel map.

Export and validate a candidate CHIME bundle with:

```bash
pilot-proxy export-runtime-weight-bundle \
  --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
  --detector-core-profile configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json \
  --weight-coordinate-system post_spectral_sense_normalization \
  --physical-channel-range 14:36 \
  --output-dir generated/runtime_bundle

pilot-proxy validate-runtime-weight-bundle \
  --bundle-dir generated/runtime_bundle
```

The validator checks file hashes, profile bounds and alignment, channel
uniqueness, detector-contract hashes, weight hashes, and coordinate consistency.
These checks validate the exported files; they do not validate the future
Kotekan metadata binding.

At startup, a stage can upload the complete `weights.bin` bank and select a
device offset, or copy only the selected profile after channel identification.
In either design, the CUDA kernel receives a device pointer. A host pointer is
not a valid substitute.

## 4. Weight Coordinate Convention

The CUDA kernel consumes detector-coordinate windows. For inverted-spectrum
CHIME input, the stage must choose one complete convention:

- Reverse each detector window and use
  `post_spectral_sense_normalization` weights.
- Preserve the raw time order and use `raw_input_frequency_coordinate`
  weights.

The three bundle metadata files must agree on `weight_coordinate_system`,
`input_coordinate_system`, and the preprocessing flag. For the current Python
CHIME path, the contract is:

```json
{
  "input_coordinate_system": "post_spectral_sense_normalized",
  "input_preprocessing": {
    "time_reverse_detector_windows_before_kernel": true
  }
}
```

Therefore the current path uses `post_spectral_sense_normalization` weights.
Raw input-coordinate weights remain a supported bundle-generation option for a
future stage, but they cannot be combined with detector-window reversal.

## 5. Kernel ABI

The proposed stage must bind the following CUDA inputs:

- packed complex int4 detector rows;
- a device pointer to the selected packed int4 target, lower-reference, and
  upper-reference weights;
- the rational half-threshold numerator and denominator stored in the selected
  profile;
- device storage for the target power, summed reference power, and rejection
  mask.

For the norm-corrected positive-excess rule, the bundle stores:

```text
positive_excess_half_threshold_num = target_norm_sq
positive_excess_half_threshold_den = reference_norm_sum_sq
```

These values are data, not architecture: a future deployment can move the
operating point (for example to a null-calibrated false-alarm quantile) by
regenerating the bundle with a different rational pair, with no kernel
change.

The fine candidate decision extends this ABI: the stage additionally binds
the row-sum entry (kernel core 2.0.0) and the on-device fine power stage
(kernel core 2.1.0, `FStat_Compute_FinePowers_U64`: the frozen fxfft256
v1 FFT plus exact uint64 feed sums, handle-free over the row-sum buffer).
The remaining decision arithmetic --- F2 ratio, CFAR estimate,
designated-set compare with the per-channel fine calibration fields
described in Section 2a --- forms downstream of those exact sums and
moves on-device once the calibration campaign fixes its inputs. The
rational half-threshold path remains bound for the recorded coarse flag
and debug comparison. The target deployed form is a single fused kernel
--- packed samples, packed weights, and bundle constants in; one mask
bit per aligned frame out; row sums and exact powers as optional debug
taps --- per the candidate-decision section of `docs/DESIGN_DECISIONS.md`.
Kernel core 2.2.0 lands the fused datapath
(`FStat_Compute_FusedFine_U64`: packed samples and weights in, exact
fine and coarse power sums out in one launch, row sums global only via
the optional debug tap, bit-identical to the two-entry composition ---
`tests/kernel/test_fused_fine_gpu.py`). Kernel core 2.3.0 completes
the form: `FStat_Compute_FusedFineMask_U64` binds the per-channel
`fine_calibration` bundle fields (anchor, designated width, 256-bit
bulk mask, CFAR rank, Q16 multiplier) as arguments and emits the mask
bit from the frozen fine decision v1 epilogue (bit-identical to
`src/pilot_proxy/fine_decision.py`;
`tests/kernel/test_fused_mask_gpu.py`). The stage binds this entry once
the campaign flips the channel's `fine_calibration.status` to
`calibrated`; a `pending_campaign` channel must not enable the fine
mask path.

The deployed CUDA comparison for that coarse flag is:

```text
mask = (p_target * reference_norm_sum_sq > target_norm_sq * p_ref_sum)
```

When `p_ref_sum == 0`, the CUDA API forces `mask = 0`. A consumer can therefore
derive `valid` as `p_ref_sum != 0`. Production may retain only the aligned mask
after this decision, while debug and validation builds should expose the exact
`uint64` powers and overflow counter long enough to compare the Kotekan path
with the Python reference.

The 16,384-sample framing used in the software profile is associated with the
planned CHIME engine upgrade. It should remain a configurable and recorded
interface value until the upgrade's Kotekan frame contract is published. A
12,288-sample current-frame value has been recalled but is still provisional;
the integration must obtain the active value from authoritative metadata rather
than infer it from this repository.

## 6. Open Questions (CHIME)

- Which Kotekan metadata field carries the integer CHIME channel identifier?
- How does that identifier map to `chime_channel_id` in the runtime bundle?
- Can the channel identifier change during one stage lifetime?
- Should a non-pilot node emit an all-zero mask frame or emit no mask product?
- Which frame or alignment key identifies each emitted mask?
- What frame length will the active CHIME engine expose, and where is that value
  carried in the Kotekan configuration or metadata?
- Which debug deployment will compare device mask decisions and `uint64` powers
  against the Python path before the powers are removed from production output?

For CHORD, Section 7 resolves each of these against the kotekan `chord`
branch; the CHIME answers must still be confirmed against the CHIME/Kotekan
deployment when that integration is scheduled.

## 7. CHORD / kotekan `chord` branch resolution

The CHORD integration (kotekan stage `cudaPilotProxyDetector`, vendored
kernel under `lib/cuda/pilotproxy/`, packer `cudaPilotProxyPacker.cu`,
kernel-level verification stage `testPilotProxyDetector`) binds this
repository's runtime bundle with the following resolved interface values.

### 7.1 Receiver parameters (kotekan `chord` branch source)

| quantity | CHORD value | source |
|---|---|---|
| ADC sample rate | 3.2 GHz, first Nyquist zone | `config/fengine/include/fengine_chord.j2` |
| PFB | 16384-point, 4 taps | same |
| coarse channels | 8192 x 195312.5 Hz exactly | `sampling_rate / fft_length` |
| channel RF center | `freq_id * 195312.5 Hz` (upright, ascending; channel 1536 = 300 MHz, 7680 = 1500 MHz) | `CHORDTelescope::FreqParams` (`freq0 = 0`, `df = +195312.5 Hz`) |
| channelized cadence | 5.12 us/sample | 16384 / 3.2 GHz |
| streams | 1024 dish-pol (512 dishes) full CHORD; 128 (64 dishes) pathfinder | `setup_chord.jl` / `setup_pathfinder.jl` |
| voltage buffer | `[T, F, P, D]`, `int4x2_swapped_withoffset` (offset-8; imag low nibble, real high nibble) | `DataType.hpp`, `fengine_chord.j2` |
| kotekan GPU frame | 8192 samples = 41.94304 ms | `num_times` |
| detector block | 8192 samples = 1 GPU frame = 41.94304 ms; K=64 gives `windows_per_stream = 128` (the frozen fine transform length) | stage config `samples_per_detector_frame` |
| fine bin width | 3051.7578125 Hz | 195312.5 / 64 |

The corresponding receiver profiles in this repository are
`configs/receiver_profiles/chord_dtv_fengine.json` and
`chord_pathfinder_dtv_fengine.json`; the ATSC 14-36 pilots land in CHORD
channels 2408-3084 (channels 14 and 21 are the adaptive-reference cases:
ch14's upper reference wraps the frame edge; ch21's lower reference is
DC-shifted, wrapping the frame edge).

### 7.2 Answers to the Section 6 questions, for CHORD

- **Channel identifier field**: `chordMetadata::coarse_freq` (a `vector<int>`
  with one entry per local frequency), populated by the receive path from
  `CHORDTelescope::to_freq_id`. Constant per run.
- **Identifier mapping**: `freq_id == coarse_channel_index` in the CHORD
  profiles (declared as `metadata.channel_id_map` with offset 0), so the
  bundle's `chord_channel_id` is directly comparable to `coarse_freq`
  entries. First-frame selection: bind every local frequency whose
  `coarse_freq` entry appears in the bundle.
- **Identifier stability**: treated as fixed for one run; the stage
  FATAL-errors on a changed `coarse_freq` vector rather than rebinding.
- **Non-pilot node behavior**: emit all-zero mask and power frames (uniform
  product shape across nodes; DISABLED nodes stay cheap).
- **Frame identity key**: the voltage ring-buffer read offset in channelized
  samples (recorded as the output `fpga_seq_num`), plus the input metadata's
  FPGA sequence base; one mask/power set per detector block.
- **Frame length**: carried in kotekan config (`num_times` for the ring
  advance; `samples_per_detector_frame` for the detector block). Not
  inferred from this repository.
- **Debug comparison**: the stage always emits the exact `uint64` coarse
  power marginals (`dtv_powers`, `[F][3]`) alongside the mask byte
  (`dtv_mask`, `[F]`), so Kotekan-vs-Python parity can be checked on real
  data before any production trimming. At the kernel level,
  `testPilotProxyDetector` bit-compares the full GPU datapath (packer, row
  sums, coarse powers, fine powers, coarse rational mask, fine
  designated-set CFAR mask) against CPU references, including a bit-exact
  C++ port of `fine_decision.py` cross-validated against the Python
  implementation.

### 7.3 Input conversion

`int4x2_swapped_withoffset` already stores the real component in the high
nibble and the imaginary component in the low nibble, so the kotekan packer's
value conversion is a per-byte `XOR 0x88` (subtract 8 per nibble:
offset-binary to two's-complement), identical in effect to
`repack_chime_offset_binary_i4_to_twos_complement`. In addition the packer
must time-reverse each K-sample detector window
(`time_reverse_detector_windows_before_kernel = true` in CHORD bundles):
the post-spectral-sense weight synthesis emits `exp(-2j*pi*f*k)` templates in
the true-sense raw frame and assumes the adapter flip for every explicit-
baseband-frame profile, upright receivers included. CHORD's upright sense
only means the raw-frame pilot offsets need no sense conversion -- it does
not remove the flip. (An earlier revision of this document claimed the
opposite; the CHORD tone-injection test, which synthesizes pilots at the
first-principles ATSC frequencies and runs the deployed bundle contract,
is the guard: without the reversal the detector is blind to upright pilot
tones.) The post-spectral-sense and raw-input weight coordinate systems
still generate bit-identical banks for CHORD; only the preprocessing flag
differs between those two contracts.

### 7.4 Remaining CHORD verification items

- Verify the baseband frame convention (`channel_center_normalized = 0.0`,
  upright sense, no per-parity twiddle) against CRS F-engine data; the
  profiles are marked `example_requires_data_product_verification` until
  then.
- The vendored kernel API executes on the legacy default CUDA stream; the
  kotekan stage brackets it with events. A stream-taking API variant is a
  candidate upstream change if profiling ever shows the implicit
  default-stream synchronization mattering at the 84 ms cadence.
- The fine mask path stays disabled per channel until the calibration
  campaign flips `fine_calibration.status` to `calibrated` (the stage's
  `decision_mode: auto` rule); until then the coarse rational mask runs.
- The proposed correlator-side FIFO handoff of Section 2a (decision consumed
  by the correlator at integration boundaries, `exceedance_10s` counter) is
  not yet wired; the current stage emits per-block products into ordinary
  kotekan buffers.
