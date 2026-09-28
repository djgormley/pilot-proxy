"""The archive's inventory-side denominators (chapter 7, ``sec:survey:data``,
the stub that closes the release-ledger paragraph).

The archive analysis counts frames, units and channels inside the products.
It never sees the archive *outside* them: how many events the survey
enumerated, how many the recorded outrigger-label rule removed, how many were
targeted and completed, how many units quarantined, and how many unique
event--channel objects the inventory held. Those live in the ``pilot-proxy``
inventory exports and in the scanning campaign's own ledgers, and this table
is the only place the dissertation states them.

Two inventories are involved and the table keeps them apart, because they are
different snapshots of the same archive and the chapter must not average them:

- the **2026-08-03 freeze** ``chime-pilots-v5`` (``--inventory``, or
  ``$PILOT_PROXY_ARCHIVE_INVENTORY``), which is the only export carrying the
  *event-level* survey ledger; and
- the **2026-08-29 rebuild** campaign that actually produced the per-pilot
  products the run of record reads, whose inventory, scan summary and
  quarantine ledger sit beside those products (found from the run's own
  ``products_dir``).

Columns of ``archive_accounting.tex``, one row per denominator:

``quantity``
    the denominator, indented under the total it partitions.
``count``
    the value, recounted from the row-level file named in ``record`` rather
    than copied from a summary; ``--`` where these files do not carry it.
``record``
    the file the count was taken from, relative to its inventory root.

Rows and their inputs. Freeze block: ``events enumerated`` /
``carrying the outrigger label`` / ``targeted`` from ``enum_cache.source.json``
(one entry per enumerated event, its value the event's dataset labels; the
recorded exclusion rule, reproduced here, is :data:`OUTRIGGER_RULE`);
``completed`` from ``surveyed_events.source.txt``; ``pending`` from
``attempts.source.json``; ``incomplete`` from ``incomplete_events.source.txt``;
``no target-channel object`` from ``no_files_events.source.jsonl``;
``objects before exclusion`` from ``inventory.meta.json``
(``derived_inventory.source_units``); ``excluded at the freeze`` from
``exclusions.jsonl``; ``retained`` and ``events represented`` from
``inventory.jsonl`` (unique ``(scope, event, freq_id)`` and ``(scope, event)``).
Campaign block: objects, events, per-channel spread and catalogued bytes from
``kit/pp_switch/inventory.jsonl``; completed, failed and unprocessed units from
``logs/channels/summary.csv``; quarantined units from
``products/_per_pilot/quarantine.jsonl``; the campaign's ``*_RUN_LEDGER.md``
FINAL ACCOUNTING block is read only as a cross-check and disagreements are
reported in the notes. Last row: the run of record's own unit total, summed
over ``ledger/channels/ch*.json`` (``product.n_units``), which counts units
contributing at least one health-gated frame and is therefore *not* the
campaign's completed-unit count.

Every count is recounted from the rows; where a manifest asserts the same
quantity the two are compared and any disagreement becomes a note. Numbers:
``ch07.accounting.*``.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from .core import DASH, Fragment, Run, booktabs, fmt, fmt_int, tex

NAME = "archive_accounting"
LABEL = "tab:survey:accounting"

FREEZE_ENV = "PILOT_PROXY_ARCHIVE_INVENTORY"

# The survey source's own exclusion rule, recorded in pilot-proxy
# (archive/sources/cadc.py `_OUTRIGGER_RE`, and archive_health.py's
# "outrigger_label_filter"). It is reproduced, not invented, here.
OUTRIGGER_SUBSTRING = "outrigger"
OUTRIGGER_RULE = ("case-insensitive substring 'outrigger' in an event's enumeration-cache "
                  "labels, matching the archive survey source's exclusion rule")
# the campaign's own predeclared quarantine class, as its ledger words it
SUBFRAME_PHRASE = "shorter than one transform"
TIB = 1024 ** 4

FREEZE_FILES = {
    "enum": "enum_cache.source.json",
    "surveyed": "surveyed_events.source.txt",
    "attempts": "attempts.source.json",
    "incomplete": "incomplete_events.source.txt",
    "no_files": "no_files_events.source.jsonl",
    "meta": "inventory.meta.json",
    "inventory": "inventory.jsonl",
    "exclusions": "exclusions.jsonl",
    "manifest": "inventory_manifest.json",
}
CAMPAIGN_FILES = {
    "inventory": "kit/pp_switch/inventory.jsonl",
    "summary": "logs/channels/summary.csv",
    "quarantine": "products/_per_pilot/quarantine.jsonl",
}
LEDGER_GLOB = "*RUN_LEDGER.md"

INDENT = ("", r"\quad ", r"\qquad ")
# (block, indent, key, quantity, record)
ROWS = (
    (0, 0, "events_enumerated", "events enumerated", "enum_cache.source.json"),
    (0, 1, "events_outrigger_excluded", "carrying the outrigger label (excluded)", "enum_cache.source.json"),
    (0, 1, "events_targeted", "targeted", "enum_cache.source.json"),
    (0, 2, "events_completed", "completed", "surveyed_events.source.txt"),
    (0, 2, "events_pending", "pending at the freeze", "attempts.source.json"),
    (0, 2, "events_incomplete", "incomplete", "incomplete_events.source.txt"),
    (0, 0, "events_without_target_object", "completed with no target-channel object", "no_files_events.source.jsonl"),
    (1, 0, "freeze_source_objects", "inventory objects before exclusion", "inventory.meta.json"),
    (1, 1, "freeze_excluded_objects", "excluded at the freeze", "exclusions.jsonl"),
    (1, 1, "freeze_objects", "retained: unique event--channel objects", "inventory.jsonl"),
    (1, 0, "freeze_events", "events represented in the retained inventory", "inventory.jsonl"),
    (2, 0, "campaign_objects", "unique event--channel inventory objects", "kit/pp_switch/inventory.jsonl"),
    (2, 1, "campaign_units_completed", "units completed", "logs/channels/summary.csv"),
    (2, 1, "campaign_units_quarantined", "units quarantined", "_per_pilot/quarantine.jsonl"),
    (2, 1, "campaign_units_failed_or_unprocessed", "units failed or unprocessed", "logs/channels/summary.csv"),
    (2, 0, "campaign_events", "events represented in this inventory", "kit/pp_switch/inventory.jsonl"),
    (2, 1, "campaign_events_per_channel", "distinct events per channel (min--max)", "kit/pp_switch/inventory.jsonl"),
    (2, 0, "campaign_bytes_tib", "catalogued source bytes (TiB)", "kit/pp_switch/inventory.jsonl"),
    (2, 0, "campaign_events_enumerated", "events enumerated, outrigger-excluded, targeted", ""),
    (3, 0, "run_product_units", "product units with a health-gated frame", "ledger/channels/ch*.json"),
)
# the italic heading that opens each block: which snapshot the rows below it count
BLOCK_TITLES = ("survey enumeration (freeze, 2026-08-03)", "unit inventory (freeze, 2026-08-03)",
                "scanning campaign (rebuild, 2026-08-29)", "the run of record")


# ------------------------------------------------------------------ locations
def freeze_dir(path: Path | str | None = None) -> Path:
    """Where the ``chime-pilots-v5`` inventory export is (the argument, else ``$PILOT_PROXY_ARCHIVE_INVENTORY``)."""
    if path is not None:
        return Path(path)
    if not os.environ.get(FREEZE_ENV):
        raise ValueError(f"the accounting table needs the inventory export: pass --inventory or set ${FREEZE_ENV}")
    return Path(os.environ[FREEZE_ENV]).expanduser()


def campaign_dir(run: Run, path: Path | str | None = None) -> Path | None:
    """The scanning campaign's root, from the run's own ``products_dir``."""
    if path is not None:
        return Path(path)
    products = run.run.get("products_dir")
    if not isinstance(products, str) or not products:
        return None
    p = Path(products)
    return p.parents[1] if p.name == "_per_pilot" and len(p.parents) > 1 else p.parent


# ------------------------------------------------------------------- readers
def _rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _labels(value) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _compare(values: Mapping, asserted: Mapping, checks: list, where: str) -> None:
    """Record every recount that disagrees with a summary's own assertion."""
    for key, claimed in asserted.items():
        got = values.get(key)
        if got is not None and claimed is not None and int(got) != int(claimed):
            checks.append(f"{where} asserts {key} = {int(claimed):,}; the rows count {int(got):,}")


def read_freeze(directory: Path | str) -> dict:
    """Recount every freeze-side denominator from its own row file."""
    directory = Path(directory)
    values: dict = {}
    inputs: list[Path] = []
    missing: list[str] = []
    checks: list[str] = []

    def take(key: str) -> Path | None:
        path = directory / FREEZE_FILES[key]
        if path.is_file():
            inputs.append(path)
            return path
        missing.append(FREEZE_FILES[key])
        return None

    path = take("enum")
    if path is not None:
        cache = json.loads(path.read_text(encoding="utf-8"))
        blocked = [k for k, labels in cache.items()
                   if any(OUTRIGGER_SUBSTRING in str(x).lower() for x in _labels(labels))]
        values["events_enumerated"] = len(cache)
        values["events_outrigger_excluded"] = len(blocked)
        values["events_targeted"] = len(cache) - len(blocked)
    path = take("surveyed")
    if path is not None:
        lines = _lines(path)
        values["events_completed"] = len(set(lines))
        if len(set(lines)) != len(lines):
            checks.append(f"{FREEZE_FILES['surveyed']} repeats {len(lines) - len(set(lines))} event keys")
    path = take("attempts")
    if path is not None:
        attempts = json.loads(path.read_text(encoding="utf-8"))
        values["events_pending"] = len(attempts)
        values["attempts_total"] = sum(int(v) for v in attempts.values())
    path = take("incomplete")
    if path is not None:
        values["events_incomplete"] = len(_lines(path))
    path = take("no_files")
    if path is not None:
        values["events_without_target_object"] = sum(1 for _ in _rows(path))
    path = take("meta")
    if path is not None:
        meta = json.loads(path.read_text(encoding="utf-8"))
        derived = meta.get("derived_inventory") or {}
        if derived.get("source_units") is not None:
            values["freeze_source_objects"] = int(derived["source_units"])
    path = take("exclusions")
    if path is not None:
        reasons: Counter = Counter()
        count = 0
        for row in _rows(path):
            count += 1
            reasons.update(str(r) for r in _labels(row.get("reasons")))
        values["freeze_excluded_objects"] = count
        values["freeze_excluded_zero_frame"] = reasons.get("prior_product_zero_frames", 0)
        values["freeze_excluded_historical_quarantine"] = reasons.get("historical_quarantine", 0)
    path = take("inventory")
    if path is not None:
        objects, events = set(), set()
        rows = 0
        size = 0
        for row in _rows(path):
            rows += 1
            objects.add((row.get("scope"), row.get("event"), row.get("freq_id")))
            events.add((row.get("scope"), row.get("event")))
            size += int(row.get("size_bytes") or 0)
        values["freeze_objects"] = len(objects)
        values["freeze_events"] = len(events)
        values["freeze_bytes"] = size
        if rows != len(objects):
            checks.append(f"{FREEZE_FILES['inventory']} holds {rows - len(objects)} duplicate event-channel keys")
    path = take("manifest")
    if path is not None:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        survey = ((manifest.get("source") or {}).get("survey")) or {}
        accounting = manifest.get("accounting") or {}
        _compare(values, {"events_enumerated": survey.get("enum_cache_entries"),
                          "events_outrigger_excluded": survey.get("outrigger_excluded_events"),
                          "events_targeted": survey.get("eligible_events"),
                          "events_completed": survey.get("surveyed_events"),
                          "events_pending": survey.get("pending_events"),
                          "events_incomplete": survey.get("incomplete_events"),
                          "events_without_target_object": survey.get("no_files_events"),
                          "freeze_source_objects": accounting.get("source_units"),
                          "freeze_excluded_objects": accounting.get("excluded_units"),
                          "freeze_objects": accounting.get("frozen_units"),
                          "freeze_events": accounting.get("frozen_events")},
                 checks, FREEZE_FILES["manifest"])
    return {"values": values, "inputs": inputs, "missing": missing, "checks": checks}


def ledger_accounting(text: str) -> dict:
    """The FINAL ACCOUNTING totals of a campaign run ledger, for cross-checking.

    Each line names a quantity and ends in its total, optionally as a sum
    (``completed 84,167 + 83,561 = 167,728``) and optionally followed by a
    ``==`` commentary the parser drops.
    """
    out: dict = {}
    body = text.split("FINAL ACCOUNTING", 1)
    if len(body) < 2:
        return out
    for line in body[1].splitlines():
        match = re.match(r"\s+(enumerated|completed|quarantined|failed|unprocessed)\b(.*)$", line)
        if not match:
            continue
        rest = match.group(2).split("==")[0]
        rest = rest.rsplit("=", 1)[-1]
        numbers = re.findall(r"\d[\d,]*", rest)
        if numbers:
            out[match.group(1)] = int(numbers[-1].replace(",", ""))
    return out


def read_campaign(directory: Path | str | None) -> dict:
    """Recount the scanning campaign's unit-side denominators."""
    values: dict = {}
    inputs: list[Path] = []
    missing: list[str] = []
    checks: list[str] = []
    if directory is None:
        return {"values": values, "inputs": inputs, "missing": ["the campaign directory"], "checks": checks}
    directory = Path(directory)

    def take(key: str) -> Path | None:
        path = directory / CAMPAIGN_FILES[key]
        if path.is_file():
            inputs.append(path)
            return path
        missing.append(CAMPAIGN_FILES[key])
        return None

    path = take("inventory")
    if path is not None:
        objects, events = set(), set()
        per_channel: Counter = Counter()
        rows = 0
        size = 0
        for row in _rows(path):
            rows += 1
            objects.add((row.get("scope"), row.get("event"), row.get("freq_id")))
            events.add((row.get("scope"), row.get("event")))
            per_channel[row.get("freq_id")] += 1
            size += int(row.get("size_bytes") or 0)
        values["campaign_objects"] = len(objects)
        values["campaign_events"] = len(events)
        values["campaign_bytes"] = size
        values["campaign_bytes_tib"] = size / TIB
        if per_channel:
            # one object per (event, freq_id), so a channel's object count is its distinct-event count
            values["campaign_events_per_channel_min"] = min(per_channel.values())
            values["campaign_events_per_channel_max"] = max(per_channel.values())
            values["campaign_inventory_channels"] = len(per_channel)
        if rows != len(objects):
            checks.append(f"{CAMPAIGN_FILES['inventory']} holds {rows - len(objects)} duplicate event-channel keys")
    path = take("summary")
    if path is not None:
        with path.open(encoding="utf-8", newline="") as fh:
            summary = list(csv.DictReader(fh))
        totals = {name: 0 for name in ("enumerated", "completed", "quarantined", "failed", "unprocessed")}
        for row in summary:
            for name in totals:
                totals[name] += int(row.get(name) or 0)
        values["campaign_units_enumerated"] = totals["enumerated"]
        values["campaign_units_completed"] = totals["completed"]
        values["campaign_units_failed"] = totals["failed"]
        values["campaign_units_unprocessed"] = totals["unprocessed"]
        values["campaign_units_failed_or_unprocessed"] = totals["failed"] + totals["unprocessed"]
        values["campaign_summary_quarantined"] = totals["quarantined"]
        values["campaign_summary_channels"] = len(summary)
    path = take("quarantine")
    if path is not None:
        keys, events = set(), set()
        rows = 0
        subframe = 0
        for row in _rows(path):
            rows += 1
            key = str(row.get("quarantine_key") or row.get("unit_key") or rows)
            keys.add(key)
            events.add(_quarantine_event(key))
            if SUBFRAME_PHRASE in str(row.get("reason") or ""):
                subframe += 1
        values["campaign_units_quarantined"] = rows
        values["campaign_quarantine_keys"] = len(keys)
        values["campaign_quarantine_events"] = len(events - {None})
        values["campaign_quarantine_subframe"] = subframe
        values["campaign_quarantine_other"] = rows - subframe
        if rows != len(keys):
            checks.append(f"{CAMPAIGN_FILES['quarantine']} repeats {rows - len(keys)} quarantine keys")
    # the scan summary counts the same quarantines; the ledger of rows is what the table prints
    _compare(values, {"campaign_units_quarantined": values.pop("campaign_summary_quarantined", None)},
             checks, CAMPAIGN_FILES["summary"])
    ledgers = sorted(directory.glob(LEDGER_GLOB))
    if ledgers:
        inputs.append(ledgers[0])
        stated = ledger_accounting(ledgers[0].read_text(encoding="utf-8", errors="replace"))
        if stated:
            _compare(values, {"campaign_units_enumerated": stated.get("enumerated"),
                              "campaign_units_completed": stated.get("completed"),
                              "campaign_units_quarantined": stated.get("quarantined"),
                              "campaign_units_failed": stated.get("failed"),
                              "campaign_units_unprocessed": stated.get("unprocessed")},
                     checks, ledgers[0].name)
        else:
            checks.append(f"{ledgers[0].name} carries no FINAL ACCOUNTING block to cross-check against")
    else:
        missing.append(LEDGER_GLOB)
    return {"values": values, "inputs": inputs, "missing": missing, "checks": checks}


def _quarantine_event(key: str) -> str | None:
    """The event a quarantine key names, in either recorded key form.

    The campaign writes ``source:["scope","event","name"]``; the freeze's own
    historical ledger writes ``event:freq_id``.
    """
    head, _, tail = key.partition(":")
    try:
        parsed = json.loads(tail)
    except (ValueError, TypeError):
        return head or None
    if isinstance(parsed, list) and len(parsed) >= 2:
        return str(parsed[1])
    return head or None


# -------------------------------------------------------------------- table
def _run_units(run: Run) -> dict:
    """The run of record's own unit total: units with at least one health-gated frame."""
    present = [c for c in run.channels if c.has("product")]
    if not present:
        return {}
    return {"run_product_units": sum(int(c.get("product", "n_units", 0) or 0) for c in present),
            "run_channels": len(present)}


def collect(run: Run, *, freeze: Path | str | None = None, campaign: Path | str | None = None) -> dict:
    """Every denominator, with the files read, the files missing and the cross-checks."""
    a = read_freeze(freeze_dir(freeze))
    b = read_campaign(campaign_dir(run, campaign))
    values = dict(a["values"])
    values.update(b["values"])
    values.update(_run_units(run))
    return {"values": values, "inputs": a["inputs"] + b["inputs"],
            "missing": a["missing"] + b["missing"], "checks": a["checks"] + b["checks"]}


def _cell(key: str, values: Mapping) -> str:
    if key == "campaign_events_per_channel":
        lo, hi = values.get("campaign_events_per_channel_min"), values.get("campaign_events_per_channel_max")
        return DASH if lo is None or hi is None else f"{fmt_int(lo)}--{fmt_int(hi)}"
    if key == "campaign_bytes_tib":
        return fmt(values.get("campaign_bytes_tib"), 2)
    if key == "campaign_events_enumerated":
        return DASH                       # the campaign shipped no event-level survey ledger
    return fmt_int(values.get(key))


def build(run: Run) -> Fragment:
    frag = Fragment(NAME, LABEL, "")
    found = collect(run)
    values = found["values"]
    cells, midrules = [], []
    block = None
    for this_block, indent, key, label, record in ROWS:
        if this_block != block:              # each block opens with its own heading row
            if block is not None:
                midrules.append(len(cells))
            cells.append([r"\emph{" + tex(BLOCK_TITLES[this_block]) + "}", "", ""])
        block = this_block
        source = tex(record) if record else r"\emph{not exported}"
        cells.append([INDENT[indent] + tex(label), _cell(key, values), source])
    frag.tex = booktabs(["quantity", "count", "record"], cells, "lrl", midrules=midrules)

    printed = set()
    for _, _, key, label, _ in ROWS:
        row = {"quantity": label}
        if key == "campaign_events_per_channel":
            for side in ("min", "max"):
                name = f"{key}_{side}"
                if values.get(name) is not None:
                    frag.add(f"ch07.accounting.{name}", int(values[name]), kind="int", row=row, column="count")
                    printed.add(name)
            continue
        if key == "campaign_bytes_tib" and values.get(key) is not None:
            frag.add(f"ch07.accounting.{key}", float(values[key]), precision=2, row=row, column="count")
            printed.add(key)
            continue
        if values.get(key) is not None:
            frag.add(f"ch07.accounting.{key}", int(values[key]), kind="int", row=row, column="count")
            printed.add(key)
    for key in ("attempts_total", "freeze_bytes", "freeze_excluded_zero_frame",
                "freeze_excluded_historical_quarantine", "campaign_bytes", "campaign_units_enumerated",
                "campaign_units_failed", "campaign_units_unprocessed", "campaign_quarantine_subframe",
                "campaign_quarantine_other", "campaign_quarantine_events", "campaign_quarantine_keys",
                "campaign_inventory_channels", "campaign_summary_channels", "run_channels"):
        if key not in printed and values.get(key) is not None:
            frag.add(f"ch07.accounting.{key}", int(values[key]), kind="int", column="")
    frag.add("ch07.accounting.outrigger_rule", OUTRIGGER_RULE, kind="text", renderings=(OUTRIGGER_RULE,),
             column="quantity")

    frag.inputs = list(found["inputs"]) + run.inputs()
    _notes(frag, values, found)
    return frag


def _closure(frag: Fragment, values: Mapping, total: str, parts: tuple[str, ...], sentence: str) -> None:
    """Note a partition that closes, or the residual when it does not."""
    if any(values.get(k) is None for k in (total,) + parts):
        return
    left, right = int(values[total]), sum(int(values[k]) for k in parts)
    shown = " + ".join(f"{int(values[k]):,}" for k in parts)
    verdict = "closes" if left == right else f"leaves {left - right:,} unaccounted"
    frag.notes.append(f"{sentence}: {left:,} = {shown} {verdict}")


def _notes(frag: Fragment, values: Mapping, found: Mapping) -> None:
    frag.notes.append(f"the outrigger exclusion is the survey source's recorded rule -- {OUTRIGGER_RULE} -- "
                      "reproduced here from the enumeration cache, not re-derived")
    _closure(frag, values, "events_enumerated", ("events_outrigger_excluded", "events_targeted"),
             "freeze enumeration")
    _closure(frag, values, "events_targeted", ("events_completed", "events_pending"), "freeze targeting")
    _closure(frag, values, "freeze_source_objects", ("freeze_excluded_objects", "freeze_objects"),
             "freeze inventory")
    _closure(frag, values, "campaign_objects", ("campaign_units_completed", "campaign_units_quarantined",
                                                "campaign_units_failed_or_unprocessed"), "campaign scan")
    if values.get("freeze_excluded_zero_frame") is not None:
        frag.notes.append(f"the {int(values['freeze_excluded_objects']):,} freeze exclusions are "
                          f"{int(values['freeze_excluded_zero_frame']):,} units a prior product had already found to "
                          f"yield zero frames and {int(values['freeze_excluded_historical_quarantine']):,} historical "
                          "quarantines; the rebuild ships no exclusions file and quarantines that class at scan time instead")
    if values.get("campaign_quarantine_subframe") is not None:
        frag.notes.append(f"the {int(values['campaign_units_quarantined']):,} quarantined units are "
                          f"{int(values['campaign_quarantine_subframe']):,} shorter than one transform and "
                          f"{int(values['campaign_quarantine_other']):,} unreadable in the archive itself, over "
                          f"{int(values['campaign_quarantine_events']):,} events")
    if values.get("run_product_units") is not None and values.get("campaign_units_completed") is not None:
        gap = int(values["campaign_units_completed"]) - int(values["run_product_units"])
        frag.notes.append(f"the products of record carry {int(values['run_product_units']):,} units with at least one "
                          f"health-gated frame, {gap:,} fewer than the {int(values['campaign_units_completed']):,} units "
                          "the campaign completed: the ledger's unit count is the health-gated population, so it is a "
                          "product denominator and not an inventory one")
    frag.notes.append("the campaign shipped no event-level survey ledger, so events enumerated, outrigger-excluded and "
                      "targeted are stated for the 2026-08-03 freeze only and are dashed for the inventory the products "
                      "were actually built from; the two inventories are separate enumerations of the same archive and "
                      "the same two scopes, a month apart, and are never added")
    if values.get("campaign_events_per_channel_min") is not None:
        frag.notes.append("an inventory object is one (event, channel) pair -- the counts are recounted as unique "
                          "(scope, event, freq_id) keys -- so a channel's object count is its distinct-event count, "
                          "which is what the per-channel row states")
    frag.notes.append("catalogued source bytes are the sum of inventory object sizes; no transfer log is present, so "
                      "they are not transferred bytes, peak storage or bytes read")
    for name in found["missing"]:
        frag.notes.append(f"absent input: {name}; every quantity it carries prints the dash")
    for check in found["checks"]:
        frag.notes.append(f"cross-check: {check}")


BUILDERS = (build,)
