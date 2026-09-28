"""The archive's inventory-side denominators: chapter 7's accounting table.

The synthetic fixture is a miniature of both inventory exports, with every
partition closing, so the tests can assert the recount rather than a copy of a
summary; separate tests remove the inputs, break the summaries and duplicate
the keys, because a table whose whole job is a denominator has to say when it
cannot state one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot_proxy.characterization.report import accounting, core

REAL_RUN = Path("/home/djg/rail/results/archive_v5_2026-09-07")
REAL_FREEZE = Path("/home/djg/rail/archive_inputs/chime-pilots-v5")


# ------------------------------------------------------------------ fixtures
def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _jsonl(path: Path, rows) -> Path:
    return _write(path, "".join(json.dumps(r) + "\n" for r in rows))


def freeze_tree(root: Path, *, surveyed_duplicate: bool = False, manifest_events: int | None = None) -> Path:
    """A miniature ``chime-pilots-v5`` export whose accounting closes.

    Six enumerated events, two of them outrigger-labelled (one capitalised,
    one behind a second label, so the case-insensitive substring rule is what
    is being tested); four targeted, three completed and one still pending.
    Eight source objects, two excluded, six retained over three events.
    """
    root.mkdir(parents=True, exist_ok=True)
    scope = "chime.event.baseband.raw"
    enum = {f"{scope}|{n}": [] for n in ("1", "2", "3", "4")}
    enum[f"{scope}|5"] = ["Outrigger.commissioning.B0643"]
    enum[f"{scope}|6"] = ["backlog.pulsar.B0355+54", "chime/OUTRIGGER-KKO"]
    _write(root / "enum_cache.source.json", json.dumps(enum))
    surveyed = [f"{scope}|1", f"{scope}|2", f"{scope}|3"]
    if surveyed_duplicate:
        surveyed.append(f"{scope}|3")
    _write(root / "surveyed_events.source.txt", "\n".join(surveyed) + "\n")
    _write(root / "attempts.source.json", json.dumps({f"{scope}|4": 2}))
    _write(root / "incomplete_events.source.txt", "")
    _jsonl(root / "no_files_events.source.jsonl", [{"scope": scope, "event": "3", "reason": "aged-out"}])
    _write(root / "inventory.meta.json",
           json.dumps({"derived_inventory": {"source_units": 8, "retained_units": 6, "excluded_units": 2}}))
    _jsonl(root / "inventory.jsonl",
           [{"scope": scope, "event": e, "freq_id": f, "size_bytes": 1024 ** 4}
            for e in ("1", "2", "3") for f in (506, 521)])
    _jsonl(root / "exclusions.jsonl",
           [{"scope": scope, "event": "1", "freq_id": 537, "reasons": ["prior_product_zero_frames"]},
            {"scope": scope, "event": "2", "freq_id": 537, "reasons": ["historical_quarantine"]}])
    _write(root / "inventory_manifest.json", json.dumps({
        "source": {"survey": {"enum_cache_entries": 6, "outrigger_excluded_events": 2, "eligible_events": 4,
                              "surveyed_events": manifest_events if manifest_events is not None else 3,
                              "pending_events": 1, "incomplete_events": 0, "no_files_events": 1}},
        "accounting": {"source_units": 8, "excluded_units": 2, "frozen_units": 6, "frozen_events": 3}}))
    return root


LEDGER = """# a campaign ledger

## RUN COMPLETE

FINAL ACCOUNTING

  enumerated   4 + 4 = 8   == the frozen inventory's row count
  completed    3 + 3 = 6
  quarantined  1 + 1 =   2
  failed                 0
  unprocessed            0
  duplicate quarantine keys   0   (within and across shards)
"""


def campaign_tree(root: Path, *, quarantine_duplicate: bool = False, ledger_completed: int | None = None) -> Path:
    """A miniature campaign: eight enumerated objects, six completed, two quarantined."""
    scope = "chime.event.baseband.raw"
    _jsonl(root / "kit" / "pp_switch" / "inventory.jsonl",
           [{"scope": scope, "event": e, "freq_id": f, "name": f"baseband_{e}_{f}.h5", "size_bytes": 2 * 1024 ** 4}
            for e in ("1", "2", "3", "4") for f in (506, 521)])
    _write(root / "logs" / "channels" / "summary.csv",
           "channel,enumerated,completed,quarantined,failed,unprocessed\n"
           "506,4,3,1,0,0\n521,4,3,1,0,0\n")
    quarantine = [{"quarantine_key": f'cadc-datatrail:["{scope}","4","baseband_4_506.h5"]',
                   "reason": "probe/read: UnreadableUnitError: CHIME packed baseband is "
                             "shorter than one transform: 3906 time samples < nfft 16384"},
                  {"quarantine_key": f'cadc-datatrail:["{scope}","4","baseband_4_521.h5"]',
                   "reason": "probe/read: UnreadableUnitError: Unable to synchronously open file (truncated file)"}]
    if quarantine_duplicate:
        quarantine.append(dict(quarantine[0]))
    _jsonl(root / "products" / "_per_pilot" / "quarantine.jsonl", quarantine)
    ledger = LEDGER if ledger_completed is None else LEDGER.replace("= 6", f"= {ledger_completed}")
    _write(root / "campaign_RUN_LEDGER.md", ledger)
    return root


def make_run(tmp_path: Path, campaign: Path | None, units=(4, 2)) -> core.Run:
    ledger = tmp_path / "results" / "ledger"
    (ledger / "channels").mkdir(parents=True)
    names = []
    for i, n in enumerate(units):
        sections = {"product": {"n_units": n}} if n is not None else {}
        rec = {"channel": 14 + i, "freq_id": 844 - i, "product": f"{844 - i}.npz", "product_sha256": "b" * 64,
               "notes": [], "sections": sections}
        _write(ledger / "channels" / f"ch{14 + i}_fid{844 - i}.json", json.dumps(rec))
        names.append(f"channels/ch{14 + i}_fid{844 - i}.json")
    run = {"generated": "x", "producer": {"commit": "a" * 40}, "channels": names}
    if campaign is not None:
        run["products_dir"] = str(campaign / "products" / "_per_pilot")
    _write(ledger / "run.json", json.dumps(run))
    return core.load_run(tmp_path / "results")


def cells(frag) -> dict:
    """The table's body as ``{quantity: (count, record)}``, headings dropped."""
    out = {}
    body = frag.tex.split(r"\midrule", 1)[-1]
    for line in body.splitlines():
        if not line.endswith(r"\\") or "&" not in line:
            continue
        parts = [p.strip() for p in line[:-2].split("&")]
        if len(parts) == 3 and parts[1]:
            label = parts[0].replace(r"\quad", "").replace(r"\qquad", "").strip()
            out[label] = (parts[1], parts[2])
    return out


def numbers(frag) -> dict:
    return {n.key: n for n in frag.numbers}


# --------------------------------------------------------------------- tests
def test_every_denominator_is_recounted_from_its_own_rows(tmp_path, monkeypatch):
    monkeypatch.setenv(accounting.FREEZE_ENV, str(freeze_tree(tmp_path / "freeze")))
    run = make_run(tmp_path, campaign_tree(tmp_path / "campaign"))
    frag = accounting.build(run)
    n = numbers(frag)
    assert n["ch07.accounting.events_enumerated"].value == 6
    assert n["ch07.accounting.events_outrigger_excluded"].value == 2      # 'Outrigger' and 'OUTRIGGER-KKO'
    assert n["ch07.accounting.events_targeted"].value == 4
    assert n["ch07.accounting.events_completed"].value == 3
    assert n["ch07.accounting.events_pending"].value == 1 and n["ch07.accounting.attempts_total"].value == 2
    assert n["ch07.accounting.events_incomplete"].value == 0
    assert n["ch07.accounting.events_without_target_object"].value == 1
    assert n["ch07.accounting.freeze_source_objects"].value == 8
    assert n["ch07.accounting.freeze_excluded_objects"].value == 2
    assert n["ch07.accounting.freeze_excluded_zero_frame"].value == 1
    assert n["ch07.accounting.freeze_excluded_historical_quarantine"].value == 1
    assert n["ch07.accounting.freeze_objects"].value == 6 and n["ch07.accounting.freeze_events"].value == 3
    assert n["ch07.accounting.campaign_objects"].value == 8 and n["ch07.accounting.campaign_events"].value == 4
    assert n["ch07.accounting.campaign_units_completed"].value == 6
    assert n["ch07.accounting.campaign_units_quarantined"].value == 2
    assert n["ch07.accounting.campaign_units_failed_or_unprocessed"].value == 0
    assert n["ch07.accounting.campaign_quarantine_subframe"].value == 1        # the recorded sub-frame phrase
    assert n["ch07.accounting.campaign_quarantine_other"].value == 1
    assert n["ch07.accounting.campaign_quarantine_events"].value == 1
    assert n["ch07.accounting.campaign_events_per_channel_min"].value == 4
    assert n["ch07.accounting.campaign_events_per_channel_max"].value == 4
    assert n["ch07.accounting.campaign_bytes_tib"].value == pytest.approx(16.0)
    assert n["ch07.accounting.run_product_units"].value == 6 and n["ch07.accounting.run_channels"].value == 2
    assert n["ch07.accounting.outrigger_rule"].kind == "text"
    assert len({x.key for x in frag.numbers}) == len(frag.numbers)
    body = cells(frag)
    assert body["events enumerated"] == ("6", r"enum\_cache.source.json")
    assert body["catalogued source bytes (TiB)"][0] == "16.00"
    assert body["distinct events per channel (min--max)"][0] == "4--4"
    closing = [note for note in frag.notes if note.endswith("closes")]
    assert len(closing) == 4                       # enumeration, targeting, freeze inventory, campaign scan
    assert not [note for note in frag.notes if note.startswith(("absent input", "cross-check"))]


def test_the_quantities_the_records_do_not_carry_are_dashed_and_named(tmp_path, monkeypatch):
    """Every input absent: every cell dashes, every missing file is named, nothing is invented."""
    monkeypatch.setenv(accounting.FREEZE_ENV, str(tmp_path / "nowhere"))
    run = make_run(tmp_path, None, units=(None, None))
    frag = accounting.build(run)
    body = cells(frag)
    assert len(body) == len(accounting.ROWS)       # every row present, every count a dash
    assert {count for count, _ in body.values()} == {core.DASH}
    assert [x.key for x in frag.numbers] == ["ch07.accounting.outrigger_rule"]
    missing = [note for note in frag.notes if note.startswith("absent input")]
    assert len(missing) == len(accounting.FREEZE_FILES) + 1
    assert any("the campaign directory" in note for note in missing)
    assert not [note for note in frag.notes if note.endswith("closes")]


def test_the_campaigns_event_survey_is_always_dashed(tmp_path, monkeypatch):
    """The campaign ships no event-level ledger; the row says so rather than borrowing the freeze's."""
    monkeypatch.setenv(accounting.FREEZE_ENV, str(freeze_tree(tmp_path / "freeze")))
    run = make_run(tmp_path, campaign_tree(tmp_path / "campaign"))
    frag = accounting.build(run)
    row = [line for line in frag.tex.splitlines() if "outrigger-excluded, targeted" in line]
    assert len(row) == 1 and core.DASH in row[0] and r"\emph{not exported}" in row[0]
    assert "ch07.accounting.campaign_events_enumerated" not in numbers(frag)
    assert any("no event-level survey ledger" in note for note in frag.notes)


def test_a_partial_export_dashes_only_what_it_lost(tmp_path, monkeypatch):
    """Three files removed: their rows dash, every other row still counts, no partition is guessed."""
    freeze = freeze_tree(tmp_path / "freeze")
    (freeze / "inventory.meta.json").unlink()
    (freeze / "exclusions.jsonl").unlink()
    monkeypatch.setenv(accounting.FREEZE_ENV, str(freeze))
    campaign = campaign_tree(tmp_path / "campaign")
    (campaign / "logs" / "channels" / "summary.csv").unlink()
    frag = accounting.build(make_run(tmp_path, campaign))
    n = numbers(frag)
    assert "ch07.accounting.freeze_source_objects" not in n and "ch07.accounting.freeze_objects" in n
    assert "ch07.accounting.freeze_excluded_objects" not in n
    assert "ch07.accounting.campaign_units_completed" not in n and "ch07.accounting.campaign_objects" in n
    assert "ch07.accounting.campaign_units_quarantined" in n     # its own ledger, not the missing summary
    body = cells(frag)
    assert body["inventory objects before exclusion"][0] == core.DASH
    assert body["excluded at the freeze"][0] == core.DASH and body["units completed"][0] == core.DASH
    assert body["retained: unique event--channel objects"][0] == "6" and body["units quarantined"][0] == "2"
    assert {note.split(":")[1].strip() for note in frag.notes if note.startswith("absent input")} == {
        "inventory.meta.json; every quantity it carries prints the dash",
        "exclusions.jsonl; every quantity it carries prints the dash",
        "logs/channels/summary.csv; every quantity it carries prints the dash"}
    closing = [note.split(":")[0] for note in frag.notes if note.endswith("closes")]
    assert closing == ["freeze enumeration", "freeze targeting"]  # the two partitions still wholly recorded


def test_a_partition_that_does_not_close_is_reported_not_hidden(tmp_path, monkeypatch):
    freeze = freeze_tree(tmp_path / "freeze")
    _jsonl(freeze / "exclusions.jsonl", [{"scope": "s", "event": "1", "freq_id": 537, "reasons": ["x"]}])
    monkeypatch.setenv(accounting.FREEZE_ENV, str(freeze))
    frag = accounting.build(make_run(tmp_path, campaign_tree(tmp_path / "campaign")))
    assert any(note.startswith("freeze inventory") and "leaves 1 unaccounted" in note for note in frag.notes)


def test_summaries_that_disagree_with_the_rows_become_cross_checks(tmp_path, monkeypatch):
    monkeypatch.setenv(accounting.FREEZE_ENV, str(freeze_tree(tmp_path / "freeze", manifest_events=99)))
    campaign = campaign_tree(tmp_path / "campaign", ledger_completed=99)
    frag = accounting.build(make_run(tmp_path, campaign))
    checks = [note for note in frag.notes if note.startswith("cross-check")]
    assert any("inventory_manifest.json asserts events_completed = 99" in c and "the rows count 3" in c for c in checks)
    assert any("campaign_RUN_LEDGER.md asserts campaign_units_completed = 99" in c for c in checks)
    assert numbers(frag)["ch07.accounting.events_completed"].value == 3        # the recount is what is printed


def test_duplicate_keys_are_counted_once_and_reported(tmp_path, monkeypatch):
    freeze = freeze_tree(tmp_path / "freeze", surveyed_duplicate=True)
    rows = [json.loads(line) for line in (freeze / "inventory.jsonl").read_text().splitlines()]
    _jsonl(freeze / "inventory.jsonl", rows + [rows[0]])
    monkeypatch.setenv(accounting.FREEZE_ENV, str(freeze))
    frag = accounting.build(make_run(tmp_path, campaign_tree(tmp_path / "campaign", quarantine_duplicate=True)))
    n = numbers(frag)
    assert n["ch07.accounting.events_completed"].value == 3 and n["ch07.accounting.freeze_objects"].value == 6
    assert n["ch07.accounting.campaign_quarantine_keys"].value == 2
    checks = " | ".join(note for note in frag.notes if note.startswith("cross-check"))
    assert "repeats 1 event keys" in checks and "1 duplicate event-channel keys" in checks
    assert "repeats 1 quarantine keys" in checks


def test_a_run_without_a_ledger_to_cross_check_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv(accounting.FREEZE_ENV, str(freeze_tree(tmp_path / "freeze")))
    campaign = campaign_tree(tmp_path / "campaign")
    (campaign / "campaign_RUN_LEDGER.md").unlink()
    frag = accounting.build(make_run(tmp_path, campaign))
    assert any(note.startswith("absent input: *RUN_LEDGER.md") for note in frag.notes)
    _write(campaign / "campaign_RUN_LEDGER.md", "no accounting here\n")
    frag = accounting.build(make_run(tmp_path / "second", campaign))
    assert any("carries no FINAL ACCOUNTING block" in note for note in frag.notes)


def test_ledger_accounting_reads_the_sum_and_drops_the_commentary():
    stated = accounting.ledger_accounting(LEDGER)
    assert stated == {"enumerated": 8, "completed": 6, "quarantined": 2, "failed": 0, "unprocessed": 0}
    assert accounting.ledger_accounting("no such block") == {}
    assert accounting.ledger_accounting("FINAL ACCOUNTING\n  completed 167,728\n") == {"completed": 167728}


def test_quarantine_keys_are_read_in_both_recorded_forms():
    assert accounting._quarantine_event('cadc-datatrail:["scope","41615268","baseband.h5"]') == "41615268"
    assert accounting._quarantine_event("41615268:614") == "41615268"       # the freeze's historical ledger
    assert accounting._quarantine_event("plain") == "plain"


def test_the_campaign_is_found_from_the_runs_own_products_dir(tmp_path):
    run = make_run(tmp_path, tmp_path / "campaign")
    assert accounting.campaign_dir(run) == tmp_path / "campaign"
    assert accounting.campaign_dir(run, tmp_path / "elsewhere") == tmp_path / "elsewhere"
    assert accounting.campaign_dir(make_run(tmp_path / "b", None)) is None


def test_the_fragment_registers_its_inputs_and_its_label(tmp_path, monkeypatch):
    freeze = freeze_tree(tmp_path / "freeze")
    monkeypatch.setenv(accounting.FREEZE_ENV, str(freeze))
    campaign = campaign_tree(tmp_path / "campaign")
    frag = accounting.build(make_run(tmp_path, campaign))
    assert frag.name == "archive_accounting" and frag.label == accounting.LABEL
    paths = {Path(p) for p in frag.inputs}
    assert freeze / "inventory.jsonl" in paths and campaign / "campaign_RUN_LEDGER.md" in paths
    assert campaign / "products" / "_per_pilot" / "quarantine.jsonl" in paths
    assert (tmp_path / "results" / "ledger" / "run.json") in paths


def test_the_module_is_registered():
    from pilot_proxy.characterization.report import build as registry

    assert "accounting" in registry.TABLE_MODULES
    assert accounting.build in registry.table_builders(("accounting",))


# ------------------------------------------------------------------ real data
@pytest.mark.skipif(not (REAL_RUN / "ledger" / "run.json").is_file() or not REAL_FREEZE.is_dir(),
                    reason="the archive run or the chime-pilots-v5 inventory export is not on this machine")
def test_real_inventory_and_campaign_close():
    frag = accounting.build(core.load_run(REAL_RUN))
    n = numbers(frag)
    assert n["ch07.accounting.events_enumerated"].value == 16327
    assert n["ch07.accounting.events_outrigger_excluded"].value == 6140
    assert n["ch07.accounting.events_targeted"].value == 10187
    assert n["ch07.accounting.events_completed"].value == 10184
    assert n["ch07.accounting.freeze_objects"].value == 165682
    assert n["ch07.accounting.campaign_objects"].value == 172437
    assert n["ch07.accounting.campaign_units_quarantined"].value == 4709
    assert n["ch07.accounting.run_product_units"].value == 167675
    assert len([note for note in frag.notes if note.endswith("closes")]) == 4
    assert not [note for note in frag.notes if note.startswith(("absent input", "cross-check"))]
