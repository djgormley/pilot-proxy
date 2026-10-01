# Archive re-run parameter register

This is the scientific and product authority for the CHIME archive re-run. It
governs two qualified execution paths: the approved local WSL run and the
qualified CANFAR sharded session. The operator launches exactly one of them.
The pass/fail gates are in [`VALIDATION_GATES.md`](VALIDATION_GATES.md), the
local production command is in [`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md), the
CANFAR production commands are in [`CANFAR_RUNBOOK.md`](CANFAR_RUNBOOK.md), and
the actual launch values belong in the run ledger for the launched path. Neither
venue guide can override this chain. Rows below marked local or CANFAR are
venue-specific; every unmarked row applies identically to both paths.

The run is historical estimation and sufficient-statistic reprocessing. The
coarse positive-excess flag is retained as a bootstrap diagnostic; the fine
rank/eta decision is inactive and no calibrated detection policy is applied.
The checked-in profiles, weight manifest, and product contract remain
authoritative; this register ties them together and makes unresolved choices
visible.

Scientific and design choices use one of four justification classes:

- **fixed by instrument**: imposed by the receiver or recorded data format;
- **provable**: follows exactly from locked values or an arithmetic bound;
- **literature**: comes from a cited standard or statistical method;
- **shown reasonably optimal**: selected by a recorded comparison or measurement.

`Open` means the repository does not yet support one of those four claims. A
compiled default is not evidence that a choice is optimal. Implementation pins
and operational controls are listed separately and do not receive a scientific
justification label.

## CHIME scientific and receiver parameters

| Parameter | Production value | Status | Justification class | Basis |
|---|---:|---|---|---|
| RF band | 400--800 MHz | Locked | fixed by instrument | Receiver profile |
| ADC sample rate | 800 MHz | Locked | fixed by instrument | Receiver profile |
| PFB transform length | 2048 | Locked | fixed by instrument | Receiver profile |
| PFB taps | 4 | Locked | fixed by instrument | Receiver profile |
| Coarse channels | 1024 | Locked | fixed by instrument | Receiver profile |
| Coarse-channel width and output rate | 390625 Hz | Locked | fixed by instrument | Receiver profile |
| Frequency-axis order and sense | descending RF, inverted | Locked | fixed by instrument | Verified receiver profile |
| Baseband channel center | DFT DC | Locked | fixed by instrument | Verified receiver profile |
| Archive-adapter detector-window reversal | on | Locked | fixed by instrument | Verified archive frame convention |
| Input streams | 2048 feed-polarization streams | Locked | fixed by instrument | Archive format |
| Native sample encoding | offset-binary 4+4-bit complex in one byte | Locked | fixed by instrument | Packed archive reader |
| Detector repack | lossless offset-binary to signed int4 | Locked | provable | Encoding conversion; no scale estimate or requantization |
| Detector window, `K` | 128 samples | Locked | provable | Exact tiling, measured span/reference bracket, fixed-point closure, and kernel geometry |
| Expected frame size, `nfft` | 16384 channelized samples | Verify at launch | fixed by instrument | Profile value and measured pre-flight value; the profile permits an override |
| Detector-cell width | 3051.7578125 Hz | Locked with `K = 128` | provable | 390625 / 128 |
| Expected frame time | 41.94304 ms | Verify at launch | provable | 16384 / 390625 |
| Expected windows per stream | 128 | Verify at launch | provable | 16384 / 128 |
| Expected detector rows per frame | 262144 | Verify at launch | provable | 2048 streams multiplied by 128 windows |
| Weight terms | target, lower reference, upper reference | Locked | shown reasonably optimal | A symmetric pair cancels odd background terms; added references cost 33% each for little idealized narrowing in the measured regime |
| Skipped guard cells | 1 | Locked | provable | A two-cell reference offset leaves exactly one intervening cell |
| Reference offset | 2 detector cells | Locked | shown reasonably optimal | A one-cell offset admits 40.5% worst-case pilot leakage; two cells reduce it to 4.5% while retaining locality |
| Coarse statistic | `F = 2 P_target / (P_ref_lower + P_ref_upper)` | Locked | provable | Two-reference statistic definition |
| Null scale | `mu0 = 2 target_norm_sq / reference_norm_sum_sq` | Locked per weight row | provable | Exact packed-weight norms |
| Recorded coarse bootstrap flag | valid and `F > mu0` | Locked diagnostic recording policy | Open | Exact and calibration-free, but not a tuned or final detection policy |
| Television channel width | 6000000 Hz | Locked | literature | ATSC 1.0 channel plan |
| Standard pilot offset from lower channel edge | `177/572` MHz, about 309440.559 Hz | Locked | literature | ATSC 8-VSB pilot placement |
| Nominal pilot below data shelf | 11.3 dB | Locked only as the standard conversion | literature | ATSC synthetic convention |
| Pilot capture efficiency | 1.0 | Conditional | Open | Replace when a receiver measurement or model supplies a correction |
| Fine zero-padding factor | 2 | Locked measurement method | Open | Nonblocking; the exact terms are retained while the fine decision is inactive |
| Diagnostic fine false-alarm probability | 0.001 per bin | Not used by this acquisition | Open | Nonblocking; selection belongs to later detection calibration |
| Diagnostic fine guard | 1 independent bin | Not used by this acquisition | Open | Nonblocking; selection belongs to later detection calibration |
| Predicted-line half-width | 30 padded bins, about 357.63 Hz | Locked diagnostic default | shown reasonably optimal | Covers the measured station offsets with margin |
| Ordinary null location and scale | median; median minus P15.87 | Locked diagnostic method | literature | Robust center and one-standard-deviation left spread |
| Fallback null location and scale | P25; `(P25 - P2.275) / 2` | Locked diagnostic method | literature | Lower-quantile robust estimator |
| Fallback trigger fraction | 0.2 | Not used by this acquisition | Open | Nonblocking; selection belongs to later detection calibration |
| Science-budget factor, `zeta` | Not used by this acquisition | Science side | literature | A science entry (RFIsher `science.systematic_budget.primary_zeta`); it does not block sufficient-statistic acquisition |

The production pre-flight must resolve `nfft = 16384`. Any other value changes
frame time, windows, rows, fine-bin count, and fine-bin width. It should fail the
current production gate instead of silently inheriting the expected values above.

Sources: [`chime_dtv_fengine.json`](../configs/receiver_profiles/chime_dtv_fengine.json),
[`pilotproxy_cuda_local_reference_power_ratio.json`](../configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json),
[`detector.py`](../src/pilot_proxy/archive/detector.py),
[`METHOD_SPEC.md`](METHOD_SPEC.md), and
[`fine_reduction.py`](../src/pilot_proxy/fine_reduction.py).

## Cross-instrument design invariant

These are profile values, not settings for the CHIME archive launch. The CHORD
profiles still require verification against an operational data product.

| Parameter | CHIME | CHORD | CHORD Pathfinder | Basis |
|---|---:|---:|---:|---|
| Channelized sample rate | 390625 Hz | 195312.5 Hz | 195312.5 Hz | fixed by instrument |
| Profile frame samples | 16384 | 8192 | 8192 | fixed by instrument |
| Detector window, `K` | 128 | 64 | 64 | common detector-cell design |
| Profile frame time | 41.94304 ms | 41.94304 ms | 41.94304 ms | provable |
| Profile windows per stream | 128 | 128 | 128 | provable |
| Input streams | 2048 | 1024 | 128 | fixed by instrument |
| Profile detector rows | 262144 | 131072 | 16384 | provable |

The design invariant is `frame_size_samples / K = 128`. The supported compiled
values of `K` are 64 and 128. `K = 256` is not supported.

Sources: the CHIME profile above,
[`chord_dtv_fengine.json`](../configs/receiver_profiles/chord_dtv_fengine.json), and
[`chord_pathfinder_dtv_fengine.json`](../configs/receiver_profiles/chord_pathfinder_dtv_fengine.json).

## Implementation and product pins

These values reproduce the current method. They are implementation or provenance
pins, not claims that a scientific choice is optimal.

| Pin | Required value or rule | Authority |
|---|---|---|
| Product schema | `pilotproxy_per_pilot_product_v5` | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Source-event key schema | `pilotproxy_namespaced_source_event_key_v1` | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Per-unit source scope | nonempty archive scope; local inputs declare `local` | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Per-unit receiver build | raw HDF5 `git_version_tag`; archive gates require nonempty values | [`packed_reader.py`](../src/pilot_proxy/archive/packed_reader.py) |
| Per-unit input configuration | SHA-256 of typed `index_map/input` content; archive gates require valid nonempty digests | [`packed_reader.py`](../src/pilot_proxy/archive/packed_reader.py) |
| Per-unit collection host | raw HDF5 `collection_server`, empty only when absent | [`packed_reader.py`](../src/pilot_proxy/archive/packed_reader.py) |
| Recorded coarse bootstrap flag | coarse norm-corrected positive excess; diagnostic only | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Scan-time fine decision | inactive; calibration status `pending_campaign` | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Shipped detector channel range | physical channels 14 through 36 | [`chime_dtv_weights_k128.bin.manifest.json`](../weights/chime_dtv_weights_k128.bin.manifest.json) |
| Implemented pilot offset | 309441 Hz above the lower channel edge | [`atsc_channels.py`](../src/pilot_proxy/atsc_channels.py) |
| Fine transform input | 128 complex row sums, padded to 256 | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Fine transform arithmetic | radix-2 DIT, signed Q15 twiddles, natural-order output | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Fine multiplier encoding | Q16 uint64, deployable values 1 through `2^64 - 1`; `2^64` is an internal non-deployable boundary marker | [`fine_decision.py`](../src/pilot_proxy/fine_decision.py) |
| Butterfly rounding | add 16384, then arithmetic shift by 15 | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Stage scaling | none | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Input component bound | absolute value at most `2^20` | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Master twiddle length | 2048 | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Master twiddle SHA-256 | `fec7f22309f4689a1a4a26258dc562487a02e33698a5e0341e2db1928f58d197` | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Fine terms | `[frames, 3, 256]` unsigned 64-bit integers | [`PER_PILOT_PRODUCT_FIELDS.md`](PER_PILOT_PRODUCT_FIELDS.md) |
| Stored diagnostic fine designated bins | predicted pilot residual with 30-bin half-width; exact array stored per product | [`detector.py`](../src/pilot_proxy/archive/detector.py) |
| Fine census exclusions | empty unless explicitly supplied; exact array stored per product | [`fine_reduction.py`](../src/pilot_proxy/fine_reduction.py) |
| Per-frame spectrum | `[frames, 16384]` signed 16-bit codes when the required `nfft` passes | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Spectrum encoding | 0.01 dB per code about the per-frame reference; invalid code -32768 | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Rail and fill diagnostics | separate per-frame counts; total components `2 * nfft * streams` | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Per-channel target, references, norms, and adaptation | exact rows in the shipped manifest; channel 14 is the sole adaptive wrap case | [`chime_dtv_weights_k128.bin.manifest.json`](../weights/chime_dtv_weights_k128.bin.manifest.json) |
| Receiver-profile SHA-256 | `bc59e77442a4c15f74c716d14eaeea4f10a69517d3bfb8c88ce10a7a42ea1e15` | Checked-in file digest |
| Weight-bank SHA-256 | `1383c6d0ca521a26b317d008feb6e09eb41427155bda9a320f70bca62e0e6259` | Shipped manifest and checked-in file digest |
| Weight-manifest SHA-256 | `d0ccc8162a350e9d3266e6acf3b38d2fe5982c474b73ef0715b8b838954e81a7` | Checked-in file digest |

Do not duplicate the 23 per-channel weight rows here. Pin the manifest path and
digest in the run ledger, then check the resolved first product against it.

## Archive launch controls

| Control | Production setting | State |
|---|---|---|
| Instrument | `chime` | Locked |
| Source | `cadc-datatrail` | Locked |
| Reader | `chime-baseband-packed` | Locked |
| Analyzer | `pilot-proxy-detector` | Locked |
| Pilot selection | 506, 521, 537, 552, 568, 583, 598, 614, 629, 644, 660, 675, 690, 706, 721, 736, 752, 767, 783, 798, 813, 829, 844 | Locked |
| Frozen inventory path and SHA-256 | `/home/djg/rail/archive_inputs/chime-pilots-v5/inventory.jsonl`; `b2cfef752a6f2cf88317141a974161439299668a37a5464d710b3800b9a872d8` | Prepared; verify before launch |
| Source inventory accounting | 170,377 units and 9,214 events | Verified against the source survey and supplemental resolution |
| Frozen inventory accounting | 165,682 usable units and 8,983 events; 4,695 exclusions | Approved and generated |
| Exclusion ledger SHA-256 | `d0ab78db9dec847fb5270fb94e3fc45efd6bbbc71adc376830fb71467693079a` | Locked |
| Pending-event resolution | 3 events, 69 selected objects absent, no errors or sub-floor objects | Verified by authenticated metadata checks on 2026-08-25 |
| Pending-event resolution SHA-256 | `077915779d234ad9cca1e9b52171e28025a60474f20fad7aff5684f55ce0c4c7` | Locked |
| Inventory manifest SHA-256 | `5695e1cc9c007cb2c79ad39535cf9ed2fb20848ad00d3cf0215933670d0707e5` | Locked |
| Resolved `nfft` | 16384 | Must pass pre-flight |
| Execution host (local) | Local WSL workstation: 24-core host, 32 WSL virtual CPUs, 62 GiB WSL memory, one 16 GiB RTX 5000 Ada | Locked for the local path; verify current free space at launch |
| Execution host (CANFAR) | CANFAR H100 session: cgroup caps of 16 CPUs and 96 GiB RAM, 11,008 MiB visible device memory; stage on `/scratch` at 1.2 GB/s and never on `/arc` at 241 MB/s | Locked for the CANFAR path; verify free space and the staging filesystem at launch |
| Topology (local) | One detector process; no other detector process against the single GPU | Locked |
| Topology (CANFAR) | 5 concurrent single-channel shards, each holding exactly one distinct physical channel; the shard count is VRAM-bound at 5, since 6 shards leaves 4 of 6 running and 8 exhausts device memory; repeat shard waves until all 23 channels are complete | Locked; qualified by the concurrent sharded-topology gate |
| Source revision | Exact pushed commit recorded in the run ledger | Freeze after all code and documentation changes |
| Source state | Clean and equal to the pushed revision | Mandatory at every launch and resume |
| Package-source SHA-256 | Exact digest recorded in the run ledger and embedded products | Freeze with the final source |
| Download workers | 4 | Best tested starting value; 13.053 MiB/s aggregate across forward and reverse sequences, though two workers won the reverse sequence |
| Maximum staged files | 8 | Ordered prefetch; frozen-inventory bound about 8.35 GiB |
| Staging permissions | `umask 077` in every launch and resume shell | Locked |
| File cap | none | Locked |
| Chunk cap | none | Locked |
| Partial-run acknowledgement | off | Locked; any new quarantine or failure stops acceptance |
| Fine products | on | Locked; retains exact fine powers and does not enable fine detection |
| Checkpoint interval | 250 units | Reduces cumulative product rewrites; a stop can repeat 249 analyzed files plus uncheckpointed prefetched downloads |
| Real-data GPU gate | One 2048-stream file, one full chunk, peak VRAM below 13,900 MiB, product and receiver-state validation pass | Repeat after the receiver-state source change and record new evidence |
| Production-profile rehearsal | Archive source, four workers, eight slots, full chunks, forced checkpoint, receiver-state validation, and identical-command resume in a separate capped output | Repeat after the receiver-state source change and record new evidence |
| Resume invariant | Same source revision, inventory, output, staging, selection, execution settings, and preserved library | Locked |
| Output directory | fresh path outside the source checkout | Fill at launch |
| Receiver profile path and SHA-256 | Checked-in CHIME profile and locked digest above | Verify at launch |
| Weight bank and manifest SHA-256 | Checked-in bank, manifest, and locked digests above | Verify at launch |
| Detector core version | 2.3.0 | Locked; identical in both paths |
| Detector library (local) | `/home/djg/rail/kernels/pilotproxy-detector-core-2.3.0-sm89-f6cd8529ca4b.so`; SM89; `f6cd8529ca4b4581aaa37a6007a372d5afb4afa8c730d8a4372a8eaf25e807f2` | Preserved digest; repeat final-source gates after the source change |
| Detector library (CANFAR) | SM90 artifact built once on the session node from the same frozen CUDA source, then preserved outside the checkout; its SHA-256 is recorded in the run ledger, not pinned here | The preserved SM89 artifact cannot run on this device. `nvcc` is not byte-reproducible, so a rebuild yields a different digest and is a new artifact: preserve the built file, gate that exact file, record its digest, and reuse it for every resume |
| Terminal product | all 23 per-pilot v5 products; channel subsets are derived only | Locked |

The 23 pilot identifiers are the ATSC 14--36 pilot locations mapped onto the
verified CHIME grid. No production file or chunk cap is allowed.

Kernel core 2.3.0 is selected for the fresh re-run in both paths. On the local
path, use the preserved SM89 artifact named above. On the CANFAR path, build the
SM90 artifact on the session node from the same frozen CUDA source, because the
preserved SM89 artifact cannot run on that device. In either case complete the
source-build GPU and parity gates and the capability and real-file gates against
the exact artifact that will run, record its digest before launch, and use that
recorded digest for every restart. Rebuilding within an approved run changes the
pinned byte identity and requires a new approval; a CANFAR rebuild in a later
session requires re-running the SM90 gates and recording the new digest in the
run ledger before the run continues. The preserved 2.1.0 binary remains
historical cohort evidence and is not the production selection.

Source: [`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md).

## Approved launch decisions

- The frozen inventory excludes the exact 4,695 known-unusable source units.
  The original inventory and historical quarantine remain unchanged beside it.
- Three pending survey events were resolved by authenticated metadata checks.
  All 69 selected objects were absent, so no inventory rows were added.
- All 23 per-pilot v5 products are the authoritative terminal deliverable.
  Any common-event channel subset is derived after the scan.
- The historical reprocessing is unblinded. A future frozen epoch is reserved
  for blinded validation.
- Combined rail and fill fields do not block launch because combined products
  are not terminal products.

## Deferred calibration decisions

| Parameter or decision | Current state | Required evidence or decision |
|---|---|---|
| Fine decision anchor and width | Implemented as runtime data but not calibrated for activation | Measure per-channel and per-era pilot locations |
| Fine decision testbench rank, `rho` | 64 is the recorded candidate, corresponding to zero-based `cfar_rank = 63`; the runtime-bundle exporter's pending block remains unset | Check in the supporting measurement and recalibrate before activation |
| Fine decision rank and multiplier | No active production values | Calibrate on verified null epochs and record product hashes and quantiles |

These decisions do not block archive acquisition. The scan records the exact
fine terms while the fine decision remains inactive.

## Threshold preparation

These values are the detector side's: eras, the residual floor, the rank and
multiplier families and the drift margins are entries of this repository's
detector register (`projects/chime_atsc/detector_register.json`, every entry
tagged `side: detector`), read through `pilot_proxy.config.register`. The
science side owns the science and selector entries (RFIsher
`projects/chime_bao/`, its `docs/decision-register.md`). The detector archive
remains a threshold-independent input: this side writes every evaluable
candidate and does not choose one.

| Decision | Current value | State | Basis |
|---|---:|---|---|
| Era summary | monthly median per-acquisition level | Provisional | Robust reduction before station-state segmentation |
| Minimum era support | 6 observed months and a 270-day elapsed span per side | Provisional | Archive guardrail; first and last retained timestamps determine the span and gaps remain visible separately |
| Minimum station step | 2 dB | Provisional | Separates large state changes from ordinary propagation scatter |
| Rank threshold | `abs(z) >= 4` | Provisional | Conservative single-comparison threshold; the date scan is not globally calibrated |
| Maximum eras | 5 | Provisional | Over-segmentation cap rather than a formal penalty |
| Thresholding era | latest | Locked | Only the current station state is relevant to a new operating point |
| Quiet-era eligibility | median level at most 1 dB | Provisional | Archive-specific boundary for a transmitter-off floor |
| Quiet-floor summary | p90 of at least 30 finite frames | Provisional | One-sided screening bound; exact coverage is not calibrated |
| Within-era split | calendar midpoint of the latest era | Locked method | Avoids an equal-frame split when cadence changes |
| Cost drift margin | unset | Open | Must follow downstream science materiality |
| Systematic-residual drift margin | unset | Open | Must follow transfer uncertainty or a residual-budget allocation |
| Per-half retained-frame floor | unset | Open | Must come from estimator precision rather than reuse the pooled selector floor |
| Stability uncertainty | unset | Open | Point-estimate ratios need an acquisition- or day-blocked interval before an operational stability claim |
| Shelf-to-science systematic and variance gains | unity | Conditional only | Screening closure; no visibility-domain transfer measurement exists |
| Candidate rank family | every one-based `rho` supported by every accepted frame | Derived | Exhausts valid implemented order statistics |
| Rank index mapping | `rho = cfar_rank + 1` | Derived | The detector field is a zero-based array index; downstream `rho` is the corresponding one-based order-statistic rank |
| Candidate multiplier family | integer Q16 value 1 and every unique deployable required Q16 change point | Derived | Exact deployable empirical staircase; Q16 value 1 is eta = 1 / 65536, not eta = 1 |
| Designated-set anchor and width | unset | Open | Requires a held-out latest-era calibration |

The science side's selector refuses an operational label while required
choices remain provisional, open, or conditional. A unity transfer may be
used only for an explicitly labeled screening calculation. The stable
identifiers, literature and sensitivity values of each entry are in the
detector register; the science side's entries and refusal behavior are in
RFIsher's `docs/decision-register.md`.

The local calibration suite below is a historical report. Its keep/excise
labels and fallback threshold are never operational exports. Threshold input
uses the `--thresholds` option or `PP_THRESHOLD_TABLE` environment variable;
the default table is `<calibration>/tables/thresholds.csv`.

## Existing report-only choices

These values reproduce the calibration report. They do not define the new
prepared threshold family.

| Decision | Current value | State | Basis |
|---|---:|---|---|
| Report multiplier ladder | 1.0, 1.1, 1.2, 1.4, 2.0, 5.0 | Historical | Sparse display and comparison grid |
| Null scale probes | lower-half p32, p5, and p0.3 with normal deviates 1, 1.96, and 2.9677 | Historical | Robust report diagnostic inherited from the released residual calculation |
| Null-centre agreement | 0.20 dB | Provisional | Archive-specific diagnostic band |
| Lower-tail support | 20 frames | Provisional | Minimal report guard, not a precision calculation |
| Detection-floor marker | one-sided Gaussian probability 0.001 | Historical | Produces the derived normal deviate 3.090232 |
| Kept-tail summary | p99 | Historical | Report diagnostic only |
| Excision label | 50% masked | Historical | Replaced by continuous masking cost in threshold selection |
| Light-masking label | 10% masked | Historical | Report label only |
| Carrier-dominated split | 3 dB | Provisional | Lies in the archive's empty 1.0--5.3 dB population gap |
| Missing-threshold fallback | eta = 1.4 | Historical | Reproduces report rows and cannot support an operational disposition |
| Threshold-bracket identification | ratio below 1.10 | Provisional | Ten-percent materiality choice |
| Era upper-level summary | p90 | Historical | Report summary only |
| Wide-spectrum centre exclusion | 60 Hz half-width | Historical | Labels the channel-centre instrumental feature rather than a transmitter |
| Wide-spectrum pilot search | 5 kHz half-width | Historical | Report refinement window around the synthesized pilot |
| Wide-spectrum peak census | at least 3 dB, at least 3 kHz separation, at most 8 peaks | Historical | Sparse report census rather than a detection policy |
| Fine-spectrum upper summary | p90 | Historical | Shows intermittent power beside the median trace |
| Fine-line census | at least 1.5 dB level and prominence, at least 48 Hz separation, at most 6 lines | Historical | Four padded-bin separation avoids counting correlated neighbours twice |

## Operational setup before archive launch

| Item | Current state | Required action |
|---|---|---|
| Archive certificate | Present locally, owner-only, expires 2026-09-03 | Renew before launch; at least 72 hours before each expiry, stop after a checkpoint, renew, and resume with the identical command |
| Local archive environment and client | Environment prepared for the prior source revision | Refresh it and rerun the full suite on the final source; never record secrets in the ledger |
| Real-data GPU capacity | Prior source passed below the 13,900 MiB limit | Repeat on the final source and require receiver-state fields for every unit |
| Production-profile resume | Prior source passed with eight unique units, 22 unique frames, and empty staging | Repeat on the final source and require receiver-state fields through resume |

Do not promote the recorded testbench rank to deployed runtime data. An exported
pending bundle leaves anchor, zero-based `cfar_rank`, and multiplier unset while
calibration is pending.

## Separate radio transfer study

This study is not an archive-launch gate. It validates estimator transfer
through the radio path and remains separate from the historical archive
acquisition.

Fill these values after a real receiver capture exists:

| Parameter | Required value |
|---|---|
| Capture | Fresh headerless complex64 file outside the source checkout |
| Capture SHA-256 | Record before audit |
| Sample rate | True receiver rate in Hz; greater than 6 MHz with edge margin |
| Tuning | Center the selected 6 MHz channel at DC |
| Physical channel | One shipped channel from 14 through 36 |
| Expected audit pilot | about -2690559.441 Hz in centered baseband |
| Minimum samples | `ceil(((((16384 + 4 - 1) * 2048) - 1) * rate / 800000000)) + 1` |
| Minimum at 10762237.762237763 Hz | 451485 complex samples, or 3611880 bytes |
| Audit command | `PYTHONPATH=src "$PYTHON_BIN" -m pilot_proxy.testbench.audit_atsc_signal --input-iq "$CAPTURE" --sample-rate-hz "$SAMPLE_RATE_HZ" --output-json "$AUDIT_JSON"` |
| Evaluator sample rate | Pass `--iq-sample-rate-hz "$SAMPLE_RATE_HZ"` with the same true receiver rate used by the audit |
| Evaluator audit attachment | Pass `--waveform-audit-json "$AUDIT_JSON"` so the report embeds the capture audit rather than the synthetic default |
| Pilot below data shelf | Pass `audit.measured_pilot_below_data_db` explicitly as `--pilot-below-data-db` |
| Evaluator frame samples | 16384 |
| Evaluator streams | 4 |
| SNR grid | -60 through 0 dB in 2 dB steps |
| Trials | 60 per point |
| Seed | 20260824 for direct comparison with the synthetic curve |
| Detector backend | CUDA |
| Synthesis backend | CPU for direct comparison with the synthetic curve |
| Noise source | GNU Radio through `/usr/bin/python3` |
| Quantization clip sigma | `--clip-sigma 3`; leave `--scale` unset so the evaluator estimates it |
| Channel gain and phase | `--channel-gain-db 0 --channel-phase-deg 0` |
| Threshold override | Leave `--threshold-data-shelf-snr-db` unset |
| Spectral sense | normal |
| Reference archive phase | off; pass `--no-reference-archive-phase` |
| Added frequency offset | 0 Hz |
| Output directory | fresh path outside the source checkout |

Four evaluator streams repeat the same captured waveform with independent added
noise; they are not four independent receiver feeds. The sweep treats receiver
noise and multipath already present in the capture as part of the clean signal.
The signal audit runs first. The evaluator startup guard then checks the clean
capture through the detector statistic and refuses a mistuned or reversed signal.

The archive adapter's detector-window reversal is part of reading the recorded
CHIME frames. The evaluator's reference archive phase is a separate testbench
transform and must remain off for this transfer curve.

Preserve the exact audit and evaluator commands. The evaluator report embeds the
audit JSON, but it does not independently bind that file to the capture or record
every command-line value. `--waveform-audit-json` is provenance-only and does not
replace the evaluator's nominal 11.3 dB conversion; pass the measured value
explicitly.

## Run ledger

Copy the run ledger template of the 2026-09 run
(`docs/LOCAL_ARCHIVE_RUN_LEDGER_TEMPLATE.md@30e50c2`) beside the run output
before launch and record:

- UTC start time and operator;
- complete command as executed;
- source revision and dirty-state result;
- resolved `nfft`, frame time, stream count, and fine geometry;
- detector library version, path, and SHA-256 digest;
- receiver-profile path and SHA-256 digest;
- weight-bank and manifest paths and SHA-256 digests;
- inventory path, SHA-256 digest, and selected pilot list;
- runtime image, Python, NumPy, CuPy, driver, and toolkit versions;
- host and WSL CPU counts, WSL memory and swap, ext4 free space, and measured
  peak VRAM for the 2048-stream smoke test;
- output directory, delivery-order controls, and checkpoint interval;
- production-profile rehearsal command, interruption point, resume result, and
  validation result;
- certificate expiry, each planned checkpoint stop and renewal, and each exact
  resume command;
- terminal-product decision;
- input capture and signal-audit paths and SHA-256 digests, when applicable;
- every quarantine-ledger entry and its disposition;
- first-product tripwire result, including the stored designated and census arrays;
- nonempty per-unit archive scopes, receiver build tags, and valid input-map
  SHA-256 values at smoke, rehearsal, first checkpoint, and closeout;
- final receiver-configuration counts from `final_inventory_audit.json`;
- final per-pilot audit results and canonical validation when terminal combine
  succeeds.

Do not store credentials or generated products in the source checkout.
