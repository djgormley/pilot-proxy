# CANFAR shard tooling for the 2026-09 archive run

These are the scripts that ran the CHIME pilot-tone archive scan on CANFAR in
September 2026, committed as provenance rather than as a general deployment
kit. Paths under `/arc/home/$CANFAR_USER/`, the session hostnames, the channel
partition, and every digest they assert are specific to that run and are meant
to be read alongside the run ledger.

`CANFAR_USER` is the CANFAR account name. It is not committed: export it
before running, and every shell script here stops if it is unset.

**Session image must be `images.canfar.net/skaha/astroml-cuda:latest`.** The
plain `astroml` image has a GPU and driver but no CUDA toolkit: every gate
passes and the scan then dies on its first file inside cupy's JIT. The
bootstrap refuses such a session up front.

## The flow that ran

| step | script | role |
|---|---|---|
| 1 | `canfar_probe_bootstrap.sh` | Per-node setup: refuse a toolkit-less image, clone at the frozen revision, restore the runtime offline from the freeze bundle's wheelhouse, reuse or build the node's kernel, measure fetch throughput. Once per session. |
| 2 | `canfar_smoke_844.sh` | Cross-arch qualification: rerun the local rehearsal (channel 844, first 8 files) and compare the product unit-by-unit against the local sm89 reference. Integer fields must be bit-identical. |
| 3 | `canfar_shard.sh` | Shard controller: `update`, `gate`, `launch`, `resume`, `run`, `tripwire`, `status`, `stop`. Every launch and resume re-runs the full gate chain, ending in an md5-verified byte fetch over the route the scan will use, and a cupy JIT compile. |
| 4 | `canfar_supervise.sh` | Keeps a shard running across archive outages and transient fetch failures, launching it if it has never run and resuming it otherwise. Distinguishes a refused gate (nothing ran) from a scan that ran and exited. |
| 5 | `canfar_merge_channel.sh` | Hands a channel finished by a helper shard to the shard that owns it, so the owner skips it instead of re-fetching. Checks run-wide identity and that the product is the channel it claims. |
| — | `canfar_compare_duplicates.py` | Field-level, NaN-aware comparison of the channels two shards both built; the basis of the byte-identical reproduction result in the ledger. |
| 6 | `canfar_closeout.sh` | Assembles the 23 canonical products, split across two non-overlapping shard directories, into one run directory. |

The CANFAR home these scripts name (`/arc/home/$CANFAR_USER/`) holds nothing
unique any more: after the closeout its copy of the products was verified
against the workstation copy and the home is being cleared. The qualified
sm90 kernel, the `pp_switch` kit (inventory, launcher scripts) and the runtime
freeze tar are preserved under the campaign's `kit/` directory,
`~/rail/products/chime_pilots_rebuild_20260829/kit/` on the workstation and
`Datasets/pilot-tone-pipeline/products/chime_pilots_rebuild_20260829/kit/` on
the WVU RAIL OneDrive, beside the products and the run ledger.

Incident and handoff artifacts from the run -- the one-off ledger repair
script, the morning handoff note, the retired autoresume watcher -- are not
tooling and do not live here. They are in the campaign archive's `kit/`
alongside the ledger that explains them (git history has them too).

## After the closeout

`canfar_closeout.sh assemble` only assembles. Everything downstream is
`scripts/generate_results.py`, which runs CPU-only:

    source "$HOME/pp-venv-$(hostname)/bin/activate"
    cd ~/pilot-proxy
    python scripts/generate_results.py --run-dir <the assembled run dir>

The venv must be active: the session's base python has no `pilot_proxy`.

It runs anywhere the products are: the September 2026 closeout ran on the
local WSL copy, with `--repo-dir` pointing at the checkout (the default is
`~/pilot-proxy`) so the integrity script and the census table are found.

It runs the per-product integrity checks, chooses the stacked combine subset
by the **pre-registered rule** (`docs/PAPER_PLAN.md@30e50c2` decision 1), combines,
validates, plots, builds the H0 zero-point tables and the cleaning tradeoffs,
runs the census, and bundles the small outputs to carry off CANFAR.

Do not hand-pick the combine subset. No event in this archive was captured on
all 23 channels, so a 23-way event-keyed stack is empty -- a fact about
coverage, not a fault. Choosing which channels to drop by reading the
drop-curve would select the analysis subset from the results, which is what
the registered rule exists to prevent.

## What this run taught

Each of these is in the run ledger, and where it is a code change, in
`docs/POST_RUN_DEFERRED.md@30e50c2`:

* a bash async job (`nohup cmd &`) inherits SIGINT ignored, so scans are
  started with `setsid -f` and stopped with INT then TERM;
* `/minoc/capabilities` stays HTTP 200 through outages -- only a real byte
  fetch, or `/raven/availability`, proves the archive is serving, and even
  `/raven/availability` reported healthy during two later interruptions;
* a pinned replica must be throughput-probed before use: UVic measured
  0.18-1.7 MiB/s against raven's 15-187 when both were healthy;
* removing a staging directory under a live scan permanently quarantines the
  units in flight -- never remove staging until `stop` confirms exit;
* one dead download worker ends the scan, and not every transient is retried
  inside `fetch()`, which is why the supervisor exists;
* launching a session rewrites the shared 30-day certificate with a 7-day
  one, so re-mint after every launch;
* `/arc` in a running session serves a cached copy: a freshly uploaded script
  is not immediately visible there. To run the current version now, use
  `git show origin/main:<path> > /tmp/<name>`.

See `docs/CANFAR_RUNBOOK.md` for the general procedure and
`docs/CADC_OUTAGE_2026-09-01.md@30e50c2` for the outage report filed during the run.
