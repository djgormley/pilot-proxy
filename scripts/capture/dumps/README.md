# Dump scripts of the 2026-09 matched capture

These scripts queued, fired and reduced the manual CHIME baseband dumps of the matched capture in September 2026.
They are published so that the capture ruling's rerun path can be reviewed. They are copies of the scripts as run,
with every site-specific value replaced by an environment variable. Apart from those substitutions, the
configuration comments and the checks for unset variables, the logic is unchanged.

**Authorization.** Each dump writes every live frequency to CHIME's shared baseband buffer, and the reduction reads
CHIME collaboration data. Run these scripts only with CHIME/FRB authorization for that specific dump, and never
without it. The site values below are internal to CHIME/FRB and are not published. Anyone who needs them should get
them from the collaboration, not from this repository.

## Scripts

| script | what it did |
|---|---|
| `dump_at.sh <label> <unix start> <length s> <min TB> <max TB> [--dry-run]` | Fired one approved science dump. It pre-flights three gates: our resident volume on the buffer, the free space, and the volume implied by the live host count falling inside the approved band. If a gate fails, it postpones 30 min, up to 8 tries. When the gates pass, it calls `trigger_dump.py` and starts `remote_wait_reduce.sh` for the accepted event. It never proceeds on a measurement it could not make. |
| `cadence_fire.sh <unix start> <length s>` | The cadence campaign: ten short dumps at offsets 0, 15, 30, 60, 120, 240, 480, 720, 960 and 1200 s from a common start. It pre-flights once for free space, live count and a 2 TB campaign total, then posts every dump ahead of time and starts one `remote_wait_reduce.sh` per accepted event. |
| `trigger_dump.py` | Posts one dump to coco's `/baseband` endpoint and deposits the baseband-converter Work, with the volume guards the shared buffer needs: a 2 TB hard cap, at most a quarter of the measured free space, a lead window of 60 to 2400 s, and a total the operator must restate in TB. Each event_id goes into a local ledger, `dumps.csv`, before the POST. Without `--dump` it only prints what it would send. |
| `remote_wait_reduce.sh <event_id>` | On the analysis host. Waits up to 20 h until the converted event has at least 630 files and the count holds for two checks, then runs `reduce_driver.sh`. |
| `reduce_driver.sh <event dir> <out dir> [containers] [threads] [max frames]` | Reduces the DTV-band files (freq_id 477 to 844, channels 14 to 37) of one event in parallel containers, on a common FPGA frame grid. |
| `reduce_dump.py` | The per-frequency reducer the driver runs: `src/pilot_proxy/instruments/chime/capture/reduce_dump.py`, copied to `$DUMP_DIR/reduce/`. |

When they were used:

- `trigger_dump.py` sent the 0.5 s pilot dump, event 20260916162300.
- `dump_at.sh`, queued with `at` on the analysis host, fired the three science dumps approved on 2026-09-16: events
  20260917040230, 20260917090230 and 20260917140230. Each asked for 3 s, and CHIME capped each at 1.4 s.
- `cadence_fire.sh` posted the ten 0.2 s cadence dumps of 2026-09-17, events 20260917160208 to 20260917162208.

These are the local copies as last run, and the copies on the analysis host were not re-read for this publication.
`trigger_dump.py` is the revision the cadence campaign ran, after its lead window was widened to 2400 s. The pilot and
science dumps ran an earlier revision, which was not kept.

`../process_dump.sh` brought each reduced event home and ran the kernel and the frame analysis on it (now
`pilot-proxy capture frame-residual` and `capture ladder-place`). The campaign's method is in
`docs/capture/FRAME_ANALYSIS_PREDECLARATION.md` and the commands of record in `docs/capture/README.md`.

## Layout on the analysis host

```
$DUMP_DIR/trigger_dump.py            (dumps.csv, its ledger, is written next to it)
$DUMP_DIR/reduce/dump_at.sh
$DUMP_DIR/reduce/cadence_fire.sh
$DUMP_DIR/reduce/remote_wait_reduce.sh
$DUMP_DIR/reduce/reduce_driver.sh
$DUMP_DIR/reduce/reduce_dump.py
```

Logs go to `$DUMP_DIR/reduce/`: `schedule_<label>.log`, `cadence_<time>.log` and `wait_<event_id>.log`.

## Variables

Export these before running. Each script checks the ones it needs and stops if any is unset.

| variable | meaning | read by |
|---|---|---|
| `DUMP_DIR` | working directory on the analysis host (see the layout above) | all shell scripts |
| `REMOTE_HOME` | the operator's home directory on the analysis host, mounted into the container | dump_at, cadence_fire, reduce_driver |
| `BUFFER_HOST` | ssh alias of the baseband buffer host | dump_at, cadence_fire |
| `BUFFER_KEY` | path to an ssh key restricted to one read-only disk report on `BUFFER_HOST` | dump_at, cadence_fire |
| `COCO_URL` | coco base URL on the CHIME network, `http://<host>:<port>`. The scripts read `$COCO_URL/baseband-status` and post to `$COCO_URL/baseband`. | dump_at, cadence_fire, trigger_dump |
| `CHIME_FRB_USER` | the operator's CHIME/FRB workflow account, recorded on the converter Work | trigger_dump (passed into the container) |
| `BASEBAND_IMAGE` | the CHIME/FRB baseband-analysis container image | dump_at, cadence_fire, reduce_driver |
| `BASEBAND_RAW_ROOT` | root of the converted raw baseband archive on the analysis host, `<root>/YYYY/MM/DD/baseband_<event_id>` | remote_wait_reduce, reduce_driver |
| `USER_DATA_DIR` | the operator's user-data directory on the analysis host. Products are written to `pilot_reduce/<event_id>/`. | remote_wait_reduce, reduce_driver |

`trigger_dump.py` also accepts `--master` and `--user` in place of `COCO_URL` and `CHIME_FRB_USER`.

The key on `BUFFER_KEY` is authorized on the buffer host with a forced command, which is not in this repository. That
command prints one line `FREE_GB <n>` and one line `OURS_MB <n>` per event directory of ours, and nothing else. The
pre-flight parses exactly those lines.

## Limits learned during the campaign

- CHIME caps a dump at 546,875 samples (1.4 s), so a longer request returns 1.4 s. Above 1.4 s, the trigger's
  estimate of 509 GB per requested second is an upper bound.
- A dump is not scoped to hosts. Every dump writes every live frequency, at 0.8 GB/s per frequency.
