# Dissertation data-export boundary

The dissertation and the PilotProxy repository have deliberately different
responsibilities:

- **PilotProxy owns scientific computation, validation, and provenance.**
- **The dissertation owns typography, layout, and explanatory graphics.**

The dissertation must not import PilotProxy or depend on a mutable checkout at
build time. Instead, PilotProxy emits a small, versioned data export that the
dissertation freezes, hashes, and renders in its own visual style.

## Create an export

From the repository root:

```bash
PYTHONPATH=src python tools/export_dissertation_data.py \
  --output-dir exports/dissertation/v1
```

The exporter must run from a clean PilotProxy Git checkout. It refuses staged
or unstaged changes to tracked files, and it checks again after reading the
inputs so a manifest cannot present a dirty worktree as a clean `HEAD`.
`--source-commit` is an optional assertion for automation: the supplied
revision must resolve to the current `HEAD`; it cannot override the checkout's
actual provenance. Generated optional CSV inputs may remain outside Git because
their normalized exported bytes are recorded by SHA-256 in the manifest.

The default export is intentionally **partial**. It always includes:

- the inclusive 500-mile transmitter-census envelope derived from
  `data/census/census.csv`, preserving its `schema_version` and per-row
  `evidence_status` so licence-only candidates are not presented as observed
  carriers;
- the inner-120-mile subset used for the detailed map panel;
- the current epoch-operating-point summary;
- the current 23-channel evidence-status matrix; and
- the current channel-33 policy comparison.

The default summary is the explicit, versioned
`data/provenance/dissertation_summary_v3.json`; no mutable duplicate or
`current` copy is maintained. Pass `--summary` only when deliberately
reproducing an older frozen draft.

Large archive products and Fisher-forecast curves are included only when their
authoritative CSV exports are supplied explicitly:

```bash
PYTHONPATH=src python tools/export_dissertation_data.py \
  --output-dir exports/dissertation/v1 \
  --census-psd /path/to/census_psd.csv \
  --worked-example-spectra /path/to/worked_example_spectra.csv \
  --bao-time-vs-masking /path/to/bao_time_vs_masking.csv \
  --bao-convergence /path/to/bao_convergence.csv \
  --bao-two-walls /path/to/bao_two_walls.csv \
  --require-complete
```

A missing table is recorded as `pending` in `export_manifest.json`. The exporter
never substitutes digitized artwork, inferred values, or synthetic data.

## Optional tables

The optional tables are supplied to the exporter through `--census-psd`,
`--worked-example-spectra` and `--bao-time-vs-masking`. The generators that
used to build them here (`tools/make_dissertation_tables.py` and
`tools/make_chain_table.py`) imported the released `rfisher` package and were
removed when the detector characterization moved into this repository; the
science-side tables are built in RFIsher. A table that is not supplied stays
`pending` in `export_manifest.json`.

## Verify before import

```bash
PYTHONPATH=src python tools/export_dissertation_data.py \
  --verify exports/dissertation/v1
```

Use `--require-complete` for a final archival snapshot. Verification checks the
schema, completion flag, file paths, row counts, and SHA-256 hashes.

## Export contents

Each export contains:

- schema-normalized CSV tables;
- `export_manifest.json`, with source repository/commit, ownership, authority,
  row counts, and file hashes;
- `SHA256SUMS`; and
- a generated `README.md` explaining available and pending tables.

The export contains **data only**. Plotting code remains in the dissertation so
that dissertation figures share one font, canvas, palette, and annotation
grammar. Article and software-documentation figures remain in this repository
because they serve different documents and may use different layouts.

## Curated summary snapshot

`data/provenance/dissertation_summary_v3.json` (snapshot
`dissertation-draft-2026-08-19`) is the current snapshot and the exporter
default. It covers all 23 ATSC allocations, including channels 14 and 15, and
carries the corrected channel-33 policy comparison on the common
`acquisitions >= 8` population. It is intentionally labeled
`curated-dissertation-snapshot` in the export manifest. Updating the default to
a later numbered snapshot is a scientific change and must be accompanied by
the analysis evidence that justifies the new values or classifications.

`dissertation_summary_v1.json` and `dissertation_summary_v2.json` remain as
immutable historical inputs for reproducing the 2026-08-12 and 2026-08-18
drafts. Select one explicitly with `--summary` only for that purpose; neither is
the current export source.

The archive-health command also does **not** rewrite
`data/provenance/dissertation_summary_v3.json`,
`analysis/dissertation/data/channel_status.csv`, or
`analysis/dissertation/data/epoch_operating_points.csv`. Figures built from
those files remain legacy/provisional epoch-status products. The dedicated
`tools/make_dissertation_status_v4.py` integration consumes the immutable v3
snapshot, health summary, rebuilt policy data, and rebuilt chain table and
writes portable `dissertation_summary_v4.json`, `channel_status_v4.csv`, and
`epoch_operating_points_v4.csv` files. It corrects invalidated prose claims
without pretending that the legacy numeric epoch operating points or their
figures were health-recomputed. Do not describe the old rows or figures as
health-corrected by implication.

## Large products

The exporter does not move raw captures, result bundles, NPZ dumps, or Fisher
workspaces into git. Those remain in the archive channel. Their small tabular
exports can be supplied to the exporter and are then fingerprinted in the
resulting dissertation export.

Before generating dissertation-facing archive tables or figures, run the
versioned health repair in [`ARCHIVE_HEALTH_REPAIR.md`](ARCHIVE_HEALTH_REPAIR.md).
Its scalar summaries and fine-F heatmaps exclude the reason-coded unhealthy
rows. Its relative spectra use the exactly reconstructible v1 DC subtraction.
The portable summary also carries per-channel health-filtered exposure by UTC
month/hour, Vancouver civil month/hour, meteorological season, and DRAO LMST
hour. Inventory-backed triggered-versus-scheduled counts and exposure are
reported separately; missing sample periods make the completed temporal
exposure explicitly partial rather than silently shrinking its denominator.
The broad predicted +/-30-bin fine neighborhood is labelled only as an
acquisition diagnostic. Narrow measured +/-2-bin anchors are provisional
UTC-quarter estimates with persistence/uniqueness evidence and refusal
reasons; an outside-neighborhood line is only a sentinel and cannot move the
pilot anchor.
The report, policy, residual-chain, and histogram generators create transient
minimal v1-filtered views for all path-only RFIsher calls; the census-PSD
table uses the same exact spectrum correction.
Do not call the fine-F heatmaps raw-voltage spectrograms, and do not call the
relative spectra absolute PSDs.

## Frozen dissertation inputs still requiring replacement

The dissertation bundle can render every current figure reproducibly, but some
plots still consume audited frozen intermediate tables rather than direct
exports from the owning scientific pipeline. The census power spectra and the
observing-time-versus-masking curves now have direct generators (above). Before
the archival dissertation release, replace the remaining frozen inputs: the
worked detector example, the BAO convergence and two-walls curves, and any
digitized introductory cosmology curves. The manifest must continue to label
these as frozen bridges until their upstream generators emit hash-pinned tables
directly.
