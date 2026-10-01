# Single-input distribution references

`tools/run_sdr_distribution_reference.py` freezes and runs actual GNU Radio
noise and steady-tone-plus-noise records through the unchanged batch adapter.
The completed study is documented in
`../../results/sdr_distribution_references_2026-09-09/README.md`.

Each case uses 24 independent seeds, two-second 2 MHz records, and 1,128 complete
16,384-sample frames. Floating natural weights, quantized weights in floating
arithmetic, and packed sample/integer processing are paired on identical data.
The ideal single-input F(256,512) and noncentral F(256,512;2304) references have
declared levels; no empirical recentering, whitening or scale fitting occurs.

The study retains raw generated streams, per-record source identities, all
projected statistics, clipping and dependence diagnostics. It verifies exact
packed arithmetic and independent source regeneration. Histogram discrepancies
are descriptive effects, not a physical qualification or a formal IID F-law
test. Physical SDR reference captures and the archive's 2048-input independence
assumptions remain separate requirements.

Use the runner's `freeze`, `run` and `report` stages with a new output directory.
It invokes GNU Radio using isolated distro Python and limits numerical-library
threads to one. The summary was plotted by RFIsher's
`scripts/plot_sdr_distribution_references_v1.py@f56916a` (kept in that
repository's history). Focused tests are in
`tests/core/test_sdr_distribution_reference.py`.
