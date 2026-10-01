#!/usr/bin/env python3
"""Trigger one manual CHIME baseband dump, with the volume guards the shared buffer needs.

A dump is NOT scoped to hosts: coco's /baseband endpoint is declared `group: cluster` and its worker
forwards without a host list, so every dump writes every live frequency. The cost is therefore

    total = duration_s x 0.8 GB/s x (live frequencies)          (2048 inputs x 1 B per 2.56 us sample)

which at 636 live frequencies is 509 GB for every second of dump. The per-frequency figure is 636 times
smaller and must never be the number anyone checks.

Nothing is sent unless --dump is given AND --confirm-total-tb matches the computed total. The operator has
to state the total in terabytes, so a wrong duration cannot pass as a wrong-but-plausible per-file size.

  python3 trigger_dump.py --start now+600 --length 0.5 --live-freq 636 --free-gb 9700
  python3 trigger_dump.py --start now+600 --length 0.5 --live-freq 636 --free-gb 9700 \
      --dump --confirm-total-tb 0.25

--live-freq and --free-gb are the two numbers the runbook makes you measure first (coco baseband-status
for the live frequency count, df -h on the buffer host's data volume for the free space). --work-only re-deposits
the converter Work for a dump that was already accepted, without POSTing again.
"""
import argparse, csv, datetime, json, math, os, sys, time

# Site configuration: CHIME/FRB internal values, never committed; set them in the environment (see README.md).
#   COCO_URL        coco base URL on the CHIME network, http://<host>:<port>; the dump is POSTed to $COCO_URL/baseband
#   CHIME_FRB_USER  the operator's CHIME/FRB workflow account, recorded on the converter Work
# --dump sends a real dump to CHIME's shared buffer: every dump needs CHIME/FRB authorization; never send without it.
COCO_URL = os.environ.get("COCO_URL", "").rstrip("/")

GB_PER_FREQ_PER_S = 0.8          # 2048 inputs x 390,625 samples/s x 1 byte
HARD_CAP_GB = 2000.0             # no single dump above 2 TB without editing this file
MAX_FRACTION_OF_FREE = 0.25      # and never more than a quarter of the measured free space
MIN_LEAD_S, MAX_LEAD_S = 60.0, 2400.0
USABLE_RING_S = 20.0
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps.csv")

p = argparse.ArgumentParser()
p.add_argument("--start", required=True, help="naive ISO UTC, a unix time, or now+<seconds>")
p.add_argument("--length", type=float, required=True, help="dump duration in seconds")
p.add_argument("--live-freq", type=int, required=True, help="live frequencies now (runbook step 2)")
p.add_argument("--free-gb", type=float, required=True, help="free GB on the buffer host's data volume (runbook step 1)")
p.add_argument("--dm", type=float, default=0.0)
p.add_argument("--user", default=os.environ.get("CHIME_FRB_USER"))
p.add_argument("--master", default=f"{COCO_URL}/baseband" if COCO_URL else None)
p.add_argument("--dump", action="store_true", help="actually send")
p.add_argument("--confirm-total-tb", type=float, help="required with --dump: the total you expect, in TB")
p.add_argument("--work-only", action="store_true", help="re-deposit the converter Work for an accepted dump")
a = p.parse_args()
if (a.dump or a.work_only) and not a.user:
    sys.exit("set CHIME_FRB_USER (or pass --user) before sending")
if a.dump and not a.master:
    sys.exit("set COCO_URL (or pass --master) before sending")

# ---- resolve the start time (naive means UTC; an explicit offset is converted, not overwritten)
if a.start.startswith("now+"):
    t = time.time() + float(a.start[4:])
elif a.start.replace(".", "", 1).isdigit():
    t = float(a.start)
else:
    d = datetime.datetime.fromisoformat(a.start)
    d = d.replace(tzinfo=datetime.timezone.utc) if d.tzinfo is None else d.astimezone(datetime.timezone.utc)
    t = d.timestamp()
lead = t - time.time()

# ---- guards, in the order that matters
if not math.isfinite(a.length) or not (0.0 < a.length <= USABLE_RING_S):
    sys.exit(f"length must be between 0 and {USABLE_RING_S:g} s (the usable ring), got {a.length}")
if a.live_freq <= 0 or a.live_freq > 1024:
    sys.exit(f"live frequencies must be 1..1024, got {a.live_freq}")
if a.free_gb <= 0:
    sys.exit("free space must be positive")
total_gb = a.length * GB_PER_FREQ_PER_S * a.live_freq
if not a.work_only:
    if total_gb > HARD_CAP_GB:
        sys.exit(f"this dump is {total_gb/1000:.2f} TB, over the {HARD_CAP_GB/1000:.1f} TB cap in this script")
    if total_gb > MAX_FRACTION_OF_FREE * a.free_gb:
        sys.exit(f"this dump is {total_gb/1000:.2f} TB, over {MAX_FRACTION_OF_FREE:.0%} of the {a.free_gb/1000:.2f} TB free")
    if not (MIN_LEAD_S <= lead <= MAX_LEAD_S):
        sys.exit(f"start must be {MIN_LEAD_S:g} to {MAX_LEAD_S:g} s ahead; this is {lead:+.1f} s from now")

when = datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
local = datetime.datetime.fromtimestamp(t - 7 * 3600, datetime.timezone.utc)   # CHIME site, PDT
frac, whole = math.modf(t)
event_id = int(when.strftime("%Y%m%d%H%M%S"))
dump = {"event_id": event_id, "file_path": str(event_id), "start_unix_seconds": int(whole),
        "start_unix_nano": int(round(frac * 1e9)), "duration_nano": int(round(a.length * 1e9)),
        "dm": float(a.dm), "dm_error": 0.0}
conv = {"timestamp_utc": when.strftime("%Y-%m-%d %H:%M:%S.%f"), "scheduled_dump_no": event_id}

print(f"TOTAL {total_gb/1000:.2f} TB  ({total_gb:.0f} GB across {a.live_freq} live frequencies, "
      f"{100*total_gb/a.free_gb:.0f}% of the {a.free_gb/1000:.2f} TB free)")
print(f"  per frequency {a.length*GB_PER_FREQ_PER_S:.2f} GB; duration {a.length:g} s; lead {lead:+.1f} s")
print(f"  start {when:%Y-%m-%d %H:%M:%S} UTC = {local:%H:%M:%S} at the telescope; event_id {event_id}")
print(f"POST {a.master} {json.dumps(dump)}")
print(f"Work baseband-converter site=chime user={a.user} parameters={json.dumps(conv)}")

# ---- the converter path is resolved and built BEFORE anything irreversible happens
def work_class():
    try:
        from chime_frb_api.workflow import Work
    except ModuleNotFoundError:
        from workflow.definitions.work import Work
    return Work
if a.dump or a.work_only:
    import requests
    Work = work_class()
    w = Work(pipeline="baseband-converter", site="chime", user=a.user, parameters=conv)
    print(f"converter Work built ok ({Work.__module__})")
if a.work_only:
    print("work-only:", w.deposit()); sys.exit(0)
if not a.dump:
    print("(nothing sent; add --dump and --confirm-total-tb to send)"); sys.exit(0)

# ---- the operator must state the total, so a duration slip cannot pass as a plausible per-file size
if a.confirm_total_tb is None:
    sys.exit("--dump requires --confirm-total-tb <TB>")
# accept the value as printed (two decimals) or anything within 1% of the exact total
if round(a.confirm_total_tb, 2) != round(total_gb / 1000.0, 2) and abs(a.confirm_total_tb - total_gb / 1000.0) > 0.01 * max(total_gb / 1000.0, 0.05):
    sys.exit(f"--confirm-total-tb {a.confirm_total_tb} does not match the computed {total_gb/1000:.2f} TB; refusing")

# ---- one event_id per dump, recorded before the POST so a crash still leaves a trail
seen = set()
if os.path.exists(LEDGER):
    with open(LEDGER) as fh:
        seen = {r["event_id"] for r in csv.DictReader(fh)}
if str(event_id) in seen:
    sys.exit(f"event_id {event_id} is already in {LEDGER}; pick another start time")
new = not os.path.exists(LEDGER)
with open(LEDGER, "a", newline="") as fh:
    wr = csv.writer(fh)
    if new: wr.writerow(["event_id", "start_utc", "length_s", "live_freq", "total_gb", "posted_utc", "coco_status"])
    wr.writerow([event_id, f"{when:%Y-%m-%dT%H:%M:%S}", a.length, a.live_freq, f"{total_gb:.0f}",
                 datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), "sending"])

try:
    r = requests.post(url=a.master, json=dump, timeout=300)
except Exception as e:
    print(f"POST raised: {e}\n*** THE DUMP MAY HAVE BEEN ACCEPTED. DO NOT RE-RUN. ***")
    print(f"    check:   curl -s '{COCO_URL}/baseband-status?event_id={event_id}'")
    print(f"    if it ran, deposit the Work with: --work-only --start {int(t)} --length {a.length} "
          f"--live-freq {a.live_freq} --free-gb {a.free_gb}")
    sys.exit(1)
print(f"coco {r.status_code}: {r.text}")
ok = r.ok
try:
    ok = ok and json.loads(r.text).get("success", False)
except Exception:
    ok = False
if not ok:
    sys.exit("coco did not report success; the Work was NOT deposited")
print("work deposited:", w.deposit())
print(f"done. Watch: curl -s '{COCO_URL}/baseband-status?event_id={event_id}'")
