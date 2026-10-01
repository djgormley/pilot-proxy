# Completed-archive health and fine-diagnostic repair

This document defines the first retrospective repair of the completed
23-channel CHIME pilot survey. The source products remain immutable; the
repair writes a versioned inclusion mask, a reason-coded exclusion ledger,
recomputed scalar/fine summaries, and the one integrated-spectrum correction
that the retained data can prove exactly.

```bash
pilot-proxy audit-archive-health \
  --products-dir /path/to/_per_pilot \
  --product-archive /path/to/_per_pilot.zip \
  --inventory-archive /path/to/chime-pilots.zip \
  --kernel-library /path/to/libfstatistic-2.1.0-c85f50dd.so \
  --source-repository /path/to/pilot-proxy \
  --source-commit 94b1de0e07bdbabb7e544aff22b62ef866e9cf0c \
  --output-dir /path/to/archive-health-v1 \
  --dissertation-style \
  --expect-products 23 \
  --expect-excluded-frames 182 \
  --expect-invalid-frames 4 \
  --expect-ceiling-frames 178
```

Use `--no-plots` for a machine-readable-only run. The four `--expect-*`
arguments are snapshot assertions, not tunable health thresholds. They make a
different input set fail instead of quietly inheriting the completed survey's
known counts.

The optional source-history pair rehashes `src/pilot_proxy` at each supplied
clean Git commit with the same algorithm stamped into the products. In the
attached completed archive, commit `94b1de0e...` reproduces the source hash in
21 products. The other two products carry a second build hash that does not
map to the supplied clean revision and therefore remains explicitly
unmatched; the audit does not guess that it came from a clean commit.

## Versioned, fail-closed frame gate

The gate identity is
`pilotproxy_archive_frame_health_gate_v1`. It validates required fields,
shapes, binary flags, frame-to-unit indices, source-event keys, and the fine
array before returning an inclusion mask. Missing or malformed evidence is an
error, not an implicit healthy row.

The completed products contain 750,461 frames. The v1 union excludes 182 and
retains 750,279:

| Reason code | Rows | Interpretation |
| --- | ---: | --- |
| `detector_invalid` | 4 | The detector denominator was zero. The analyzer did not add these rows to either integrated spectrum. |
| `detector_powers_all_zero` | 4 | Both detector power terms are zero. These are the same four rows, so this count must not be added again. |
| `baseband_power_at_negative_full_scale_ceiling` | 178 | The decoded frame-mean sample power is exactly its complex-int4 maximum, 128. |

The gate also fails closed on non-finite or out-of-range baseband power,
non-finite valid-row coarse F, and non-finite or negative valid-row fine F. No
additional completed-survey row triggers those guards.

Every excluded row appears once in `archive_exclusion_ledger.jsonl`. A record
contains all triggered reason codes, product hash, physical channel, `freq_id`,
frame index, frame-within-source index, event ID, source-unit key, and derived
UTC. Its stable `ledger_key` includes the gate version, source event, frame,
and reasons. Reason counts can overlap; `excluded_unique_frames` is always the
union count.

## Why power 128 identifies the native byte

Each native CHIME byte stores two offset-binary four-bit components. Decoding
maps each component to the integer range -8 through +7. A complex sample has
power at most

```text
(-8)^2 + (-8)^2 = 128.
```

`baseband_power_linear` is the arithmetic mean of those non-negative sample
powers over every time sample and input stream in one frame. A mean equals the
upper bound only if every term equals the upper bound. Thus a mean of exactly
128 proves that every decoded sample is `(-8, -8)`.

The encoding coordinate matters:

- the native CHIME offset-binary raw byte is **`0x00`**;
- the losslessly repacked detector-input two's-complement byte for the same
  sample is **`0x88`**.

Calling the native byte `0x88` is incorrect. `0x88` is correct only after the
detector-input repack.

## Exact correction of the integrated spectra

Ordinarily, the NPZ's two accumulated spectra cannot be remasked: they contain
no per-frame complex FFTs. The 178 ceiling rows are a special case because the
retained scalar proves their entire sample array.

For a constant `(-8,-8)` frame, the unnormalised rectangular FFT is zero away
from bin 0. At bin 0 the analyzer's production expression is
`abs(complex64 FFT)**2`, followed by a float64 sum over streams. The
complex64 absolute value rounds before it is squared. For the recorded
`nfft=16384` and 2,048 input streams, the exact production contribution is

```text
34,359,736,320 per stream
70,368,739,983,360 per frame.
```

An RTX/CuPy reproduction of the analyzer operation gives that value at bin 0
and exact zeros in every other bin. The infinite-precision expression
`2048 * 16384^2 * 128` is slightly larger and must not be substituted for the
recorded complex64 arithmetic.

The repair subtracts the production value:

- from the before-mask spectrum once for each of the 178 ceiling rows;
- from the after-mask spectrum once for each of the 118 ceiling rows whose
  stored coarse decision kept the frame.

The remaining 60 ceiling rows were already rejected by the stored mask and
therefore never entered the after spectrum. The four detector-invalid rows
entered neither spectrum, so their exact spectral correction is zero. The
result is published as `health_corrected_integrated_spectra.npz`, with the
counts and DC correction value beside every channel. Publication refuses any
product with an excluded valid row whose samples are not reconstructible.

As a second invariant, each corrected spectrum sum is compared with
`nfft^2 * num_input_streams * sum(baseband_power_linear)` over the same
health-included frames. The completed products agree to at worst
`2.1e-7` relative error, inside the recorded `5e-7` bound for the production
complex64 FFT/absolute-square roundoff. Every corrected spectral bin is
non-negative.

This exception does not restore general per-frame spectra. A new arbitrary
threshold, mask, FFT window, or dilation still requires the source HDF5 files.

## Retrospective fine repair

The archived fine-F arrays are retained, but the run's stored designated set
was fine bin 0. That historical ancillary field is not reused as evidence. A
pilot need not lie at envelope bin 0: the geometry-predicted centre is the
residual between its RF frequency and the nearest coarse detector grid point,
after applying the recorded spectral sense.

The repair deliberately publishes two different objects rather than calling
one broad window a final designation:

1. `predicted_acquisition_neighborhood` is the geometry-predicted anchor
   +/-30 bins, modulo the 256-bin grid. It is a broad acquisition and diagnosis
   prior, not an empirical line localization.
2. `measured_epoch_line_anchor` is estimated independently in retrospective,
   provisional UTC-calendar quarters. These quarters are a reproducible
   partition of the archive, not authoritative transmitter/station epochs.
   Within each quarter, every health-included fine-F row is divided by its row
   median. Candidate pilot bins are restricted to the predicted acquisition
   neighborhood. A candidate is accepted only with at least 30 usable frames,
   a normalized-ratio-at-least-1.5 persistence fraction of at least 0.10, and
   a persistence margin of at least 0.02 over the strongest competitor inside
   the acquisition neighborhood but outside the candidate +/-5-bin clearance.
   A candidate within two bins of the acquisition boundary is also refused when
   a stronger persistent sentinel lies outside that boundary; this prevents the
   shoulder of an out-of-domain line from being mislabelled as the pilot (the
   important channel-23 edge case). An accepted line receives the narrow
   candidate +/-2-bin window used by the calibration tools.

The strongest line outside the acquisition neighborhood is exported as a
sentinel. It may identify an instrument or unrelated spectral line, but it can
never redefine the target pilot anchor. Each rejected quarter records its
evidence and refusal reasons, then falls back explicitly to the broad predicted
acquisition diagnostic. Circular measured-minus-predicted deltas use the
interval `[-N/2,N/2)`; positive follows increasing fine-array index.

The recomputed independent CFAR null bulk excludes the broad predicted
acquisition neighborhood, every accepted measured-line window, the stored
guard bins, and stored census exclusions. The machine summary exports the
predicted-neighborhood rate and the selected epoch-window/fallback rate under
different names. These are float retrospective diagnostics derived from the
stored fine-F array. They do not rewrite `reject_mask`, and they are not a
deployed or prospectively calibrated fine decision.

## Inventory and binary evidence

The optional evidence arguments bind independent attachments to the result:

- every NPZ member in `_per_pilot.zip` must hash-identically to the audited
  product;
- the supplied kernel library SHA-256 must equal the `kernel_sha256` token in
  every product;
- every processed and excluded source-unit key must join `inventory.jsonl`;
- inventory-minus-processed units must equal the surviving quarantine ledger
  when both archives are supplied.

For the completed evidence set, `_per_pilot.zip` reproduces all 23 NPZs. The
supplied kernel hash
`c85f50ddf898517bc0101d1882c854c3df70b09f0ab0b58803dc32f59e3c6d12`
matches every product. The inventory contains 170,377 unique target-frequency
objects across 9,214 events; products consume 170,374 units, and the three-unit
difference is exactly the three-row quarantine. All 9,214 inventory events
join processed products because each quarantined event has other usable
frequency units.

The per-pilot unit axes contain 4,692 units with no complete stored frame.
Across channels, 8,983 distinct events have at least one stored frame and
8,980 have at least one v1 health-included frame. The latter is the event count
appropriate to health-filtered frame-level analyses; it must not be replaced
by the broader 9,214-event inventory coverage count.

The inventory catalogs 28,237,618,443,352 bytes of discovered objects. This is
not a transfer measurement. The attachments contain no transfer/performance
log, so network bytes, peak storage, GPU time, wall time, and throughput are
unavailable. Those operational metrics do not block the frame-health science.

The surviving enumeration accounts for 16,327 scope events: 6,140
outrigger-labelled events were excluded, leaving 10,187 in the target survey
scope. Of those, 10,184 appear in the completed-event list and three remain in
the attempts record, each with recorded attempt count 2. The completed set
partitions into 9,214 events with
target-frequency inventory rows, 107 explicitly recorded aged-out accepted
empty events, and 863 events derived by set subtraction as having no common
target path. The last category is explicitly marked *derived*: no direct
per-event status ledger for those 863 survived.

Neither ZIP contains raw HDF5 data or a run/performance log. Inventory CADC
keys locate the original units, including all 71 unique source files implicated
by the 182 exclusions, but retrieving those files remains a separate root-cause
exercise.

## Health-filtered observing exposure

Each channel summary also exports frame counts and rectangular-frame exposure
seconds after the v1 gate in all of these reproducible coordinates:

- UTC calendar month and UTC hour of day;
- `America/Vancouver` civil calendar month and hour, with daylight-saving
  conversion supplied by the IANA time-zone database;
- local meteorological season, defined as DJF, MAM, JJA, and SON from the local
  civil month; and
- DRAO local mean sidereal hour (LMST), distinct from local civil time, local
  solar time, and local apparent sidereal time (LAST).

Frame time is
`unit_time0_ctime[u] + frame_in_unit * nfft * unit_delta_time[u]`, and frame
duration is `nfft * unit_delta_time[u]`. A frame's complete rectangular
duration is credited to the bin containing its start rather than split at a
boundary. Coverage counters make any missing timing metadata explicit.

For LMST, longitude is -119.6175 degrees in the east-positive convention. The
dependency-free transform uses

```text
JD = unix_utc_seconds / 86400 + 2440587.5
T = (JD - 2451545.0) / 36525
GMST_deg = 280.46061837 + 360.98564736629 (JD - 2451545.0)
           + 0.000387933 T^2 - T^3 / 38710000
LMST_hours = ((GMST_deg - 119.6175) mod 360) / 15.
```

UTC is used as the UT1 proxy because DUT1/Earth-orientation values were not
retained. This is adequate for one-hour exposure bins, not precision
astrometry. Hard-coded tests include the J2000 anchor and modern values from
the USNO Sidereal Time API; the declared implementation stays within 0.01
sidereal second of those vectors.

In the completed archive, 2,489 stored frames (all otherwise health-included)
belong to units without a finite positive `unit_delta_time`. The temporal
profile is therefore explicitly `partial`: 747,972 stored and 747,790
health-included frames have reproducible time/duration coordinates. The 2,489
unresolved frames remain in scalar science denominators but do not silently
enter temporal exposure seconds.

Trigger-versus-scheduled class is absent from the NPZs, so it is never guessed.
When the inventory attachment is supplied, `unit_order` joins exactly to its
`scope` field. The attached inventory maps `chime.event.baseband.raw` to
`triggered_event` and `chime.scheduled.baseband.raw` to `scheduled`. The
machine summary exports inventory/product unit counts, stored and
health-included frame counts, and exposure seconds for both classes. It labels
these exposure seconds partial for the same missing-duration units; discovered
catalogued bytes remain separate from unavailable transferred bytes.

## Outputs and their scientific names

`archive_health_summary.json` contains per-channel scalar distributions,
histograms, Wilson 95% binomial intervals, predicted/measured fine-anchor
evidence and refusals, health-filtered temporal exposure, spectrum correction
provenance, and evidence joins. `archive_exclusion_ledger.jsonl`
contains the excluded rows. `health_corrected_integrated_spectra.npz` contains
the exactly corrected spectral sums and denominators.

When plotting is enabled, every manifest path is a portable POSIX path relative
to the release root. Each deterministic
`figures/channel_XX_fid_YYYY/` directory receives:

- `health_filtered_histograms.pdf`, a vector PDF containing four
  health-filtered scalar histograms;
- `relative_time_averaged_spectra.pdf`, a vector PDF containing the exactly
  v1-corrected relative before/after time averages;
- `fine_f_utc_monthly_heatmap.png`, a 300-DPI raster heatmap with the broad
  predicted anchor and accepted provisional-quarter measured anchors labelled;
  and
- `channel_XX_diagnostic_atlas.pdf`, a one-page dissertation atlas containing
  all three diagnostic sections. Its text, histograms, and spectra are vector
  objects; the heatmap is an embedded raster image.

`diagnostic_manifest.json` gives the release-relative path, SHA-256, format, rendering
mode, channel, and `freq_id` for every asset. The 23 atlas PDFs are intended as
the direct appendix inputs; the 69 stand-alone files support alternative
layouts without rerunning the science analysis.

For dissertation release builds, pass `--dissertation-style`. This option
fails closed unless LaTeX can render embedded Latin Modern/T1 text; it never
silently substitutes DejaVu. The PDF writer suppresses `CreationDate` and
`ModDate`, and the manifest records the style contract, so repeated builds from
the same products have deterministic content and metadata. Set the standard
`SOURCE_DATE_EPOCH` environment variable to pin the machine summary's
`created_utc`; the summary records whether that timestamp came from the
environment or the wall clock. Attachment provenance is portable
basename/byte-count/SHA-256 identity rather than an operator-local absolute
path.

The spectrum is relative power, not an absolutely calibrated PSD. The fine-F
heatmap is a detector-statistic diagnostic, not a raw-voltage spectrogram. The
NPZs contain neither a flux/temperature calibration nor per-frame voltage FFTs,
so those stronger product names would be scientifically false.

## Dissertation policy and residual consumers

The released `rfisher` APIs used by the dissertation accept product paths
and load their own frame arrays. Filtering only the surrounding report code
would therefore be insufficient: the residual floor, correlation time,
threshold sweep, and mask-fraction routines would silently read the excluded
rows again.

`temporary_residual_health_views` closes that boundary. It applies the same v1
gate and writes short-lived, deliberately incomplete NPZ views containing only
the coarse-policy and residual columns used by RFIsher, plus the source
product hash and health counts. It omits the superseded fine-decision fields
and is not a replacement canonical survey product.

The dissertation-facing generators of the time
(`analysis/make_report_data.py@30e50c2`, `analysis/make_policy_data.py@30e50c2`,
`analysis/plot_channel_histograms.py@30e50c2` and
`tools/make_chain_table.py@bdfcdce`) routed every science-side call
through those views; their per-frame policy denominators also applied
`evaluate_frame_health`. The census spectra of
`tools/make_dissertation_tables.py@bdfcdce` used the exact v1 spectral
correction, and its worked-example search refused excluded rows. This is the minimum boundary
required before removing a manuscript qualifier that policy and chain values
predate the archive health repair.

The rebuild exposes one substantive consequence, rather than merely changing
small denominators. On channels 22 and 24, every frame kept by the stored
coarse rule is one of the full-scale rows excluded by v1. No health-included
stored-mask kept/null frame remains, so the empirical residual floor and
threshold sweep are unavailable. The policy generator now records that status
and takes the conservative excision action; it does not describe the two
coherence brackets as agreeing. Reopening threshold optimization for either
channel requires new clean or transmitter-off calibration data (or an
independently justified synthetic floor), not more manipulation of the NPZs.

The audit does not rewrite `dissertation_summary_v3.json`,
`channel_status.csv`, or `epoch_operating_points.csv`. Their status/epoch
figures remain provisional legacy products and must not be called
health-corrected by implication. The provisional v4 integration of those files
(`tools/make_dissertation_status_v4.py@30e50c2`) is kept in the history; it was
not a new blinded BAO verdict.

In contrast, rerunning the repaired `tools/make_dissertation_tables.py@bdfcdce`
produces final health-filtered `census_psd.csv` and
`bao_time_vs_masking.csv` inputs; its optional worked example also refuses
excluded frames.
