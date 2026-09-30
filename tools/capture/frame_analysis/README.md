# Capture ruling scripts of record

The capture measurements moved into the package (`pilot-proxy capture ...`, see `docs/capture/README.md`). What
remains here are the science scripts of the capture ruling, kept runnable until their port to the science side
lands, so that neither repository is ever without a runnable table of record:

| script | what it writes |
|---|---|
| `ruling_baseline.py` | the baseline-resolved ruling (the table of record per baseline range) |
| `table_of_record.py` | the table of record's upstream table from the frame residuals and the detector runs |
| `robustness_v8.py`, `credit_ledger.py`, `blanket_credit.py`, `concession_stack.py` | the credit and robustness readings of the ruling |
| `forecast_baseline_domain.py` | the forecast's baseline domain |
| `ch24_2020_basis.py` | channel 24's 2020 basis |

They are byte copies of the frozen ruling tree (`table_of_record.py` differs only in how it finds and records its
inputs), read the frozen frame-analysis tree by path, and are not linted (`pyproject.toml`). The capture
measurements they read (class excess, control bins, cadence levels, lags, frame residuals) are written by the
package now: the same bytes, except that the frame residual no longer carries its two tolerance cells, which these
scripts do not read.

## Inputs of `table_of_record.py`

Apart from the per-dump products, `table_of_record.py` reads four inputs. They are CHIME collaboration products
and are not published here, so a reviewer has to get them through the collaboration.

| key | variable | name under `TABLE_OF_RECORD_INPUTS` | content |
|---|---|---|---|
| `channels=` | `TOR_CHANNELS_CSV` | `channels.csv` | tolerance without credit per channel (forecast-convergence release, 2026-09-09) |
| `worlds=` | `TOR_WORLDS_CSV` | `coarse-world-sensitivity.csv` | deployed-world tolerance and suppression (CANFAR reanalysis, 2026-09-09) |
| `board=` | `TOR_BOARD_CSV` | `board_final.csv` | archive gain bound and tau bound per channel (channel ruling, 2026-09-14) |
| `archive=` | `TOR_ARCHIVE_DIR` | `_per_pilot/` | archive per-pilot detector products `<freq_id>.npz` (archive rebuild, 2026-08-29) |

Each input is taken from its `key=` argument first, then from its variable, then from the directory named by
`TABLE_OF_RECORD_INPUTS`. For example:

```sh
TABLE_OF_RECORD_INPUTS=/path/to/inputs python table_of_record.py table_of_record.csv \
  <label>=<frame_residual csv>:<chime-run dir> ... gain=<cadence_tau csv> lowbound=<csv>,... bao=<gain_bao csv>
(cd /path/to/inputs && sha256sum -c table_of_record_inputs.sha256)
```

`table_of_record_inputs.sha256` lists the copies of record. Every run writes `<out csv without extension>.inputs.json`
next to the table. It records the sha256, size, modification time, resolved path and provenance of every file read,
covering these inputs, the dump inputs and the script itself. The script differs from the frozen copy
(sha256 `f2372bf677095b32e840ad5b075dedb0e186a5198620bb304b4ba99475ba709c`) only in how it finds and records its
inputs. On 2026-09-25, a three-dump run gave the same inputs to both versions, and they wrote the same table byte for
byte.
