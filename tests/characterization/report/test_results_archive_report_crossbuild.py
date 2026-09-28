"""``crossbuild``: the cross-build reproduction record, on a synthetic campaign then the real one."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization.report import core
from pilot_proxy.characterization.report import crossbuild as m
from pilot_proxy.characterization.report import numbers as nb

ROOT = Path(__file__).resolve().parents[3]
REAL_RUN = Path("/home/djg/rail/results/archive_v5_2026-09-07")

_SPEC = importlib.util.spec_from_file_location("v5_fixture", ROOT / "tests" / "products" / "test_pilotproxy_v5.py")
v5_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5_fixture)

CHANNEL = 14
FREQ_ID = 844


def _product(path: Path, *, frames: int = 12, units: int = 4, **changes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    v5_fixture._write_product(path, CHANNEL, frames=frames, units=units)
    with np.load(path, allow_pickle=False) as archive:
        n = int(np.asarray(archive["valid"]).shape[0])
    v5_fixture._replace(path, baseband_power_linear=np.full((n, 1), 4.0), **changes)
    return path


def _detector(source: str, kernel: str) -> np.ndarray:
    return np.asarray(f"pilot-proxy/fixture source={source} kernel=2.3.0 kernel_sha256={kernel} "
                      "pilotproxy_per_pilot_product_v5 K=128")


def _campaign(tmp_path: Path, *, reference_changes=None, campaign_changes=None, survey_frames: int = 12,
              survey_units: int = 4, summary: bool = True, sums: bool = True) -> Path:
    """A campaign tree: a qualification pair, the run of record's product, SHA256SUMS and summary.csv."""
    root = tmp_path / "campaign"
    (root / "products" / "_per_pilot").mkdir(parents=True)
    reference = _product(root / "qualification" / "ref_local.npz", **(reference_changes or {}))
    smoke = _product(root / "qualification" / "smoke_run" / "_per_pilot" / f"{FREQ_ID}.npz",
                     **(campaign_changes or {}))
    survey = _product(root / "products" / "_per_pilot" / f"{FREQ_ID}.npz", frames=survey_frames, units=survey_units)
    if sums:
        lines = []
        for path in (reference, smoke, survey):
            lines.append(f"{'a' * 64}  {path.relative_to(root)}")
        (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if summary:
        (root / "logs" / "channels").mkdir(parents=True)
        (root / "logs" / "channels" / "summary.csv").write_text(
            "channel,owner_session,built_by,enumerated,completed,quarantined,failed,unprocessed,status,"
            "shard3_rebuilt,shard3_completed,shard3_quarantined\n"
            "506,shard1,shard1,10,9,1,0,0,partial,,,\n"                     # not rebuilt: no row
            "767,shard1,shard3 (merged into shard1),20,18,2,0,0,partial,yes,18,2\n"   # one builder only: no row
            "844,shard2,shard2,100,90,10,0,0,partial,yes,90,10\n"           # a genuine duplicate, agreeing
            "813,shard1,shard1,50,44,6,0,0,partial,yes,43,6\n",             # a genuine duplicate, disagreeing
            encoding="utf-8")
    return root


def _run(tmp_path: Path, root: Path | None) -> core.Run:
    ledger = tmp_path / "ledger"
    (ledger / "channels").mkdir(parents=True, exist_ok=True)
    record = {"channel": CHANNEL, "freq_id": FREQ_ID, "product": f"{FREQ_ID}.npz", "product_sha256": "b" * 64,
              "notes": [], "sections": {"era": {"current_frames": 10}}}
    (ledger / "channels" / f"ch{CHANNEL}_fid{FREQ_ID}.json").write_text(json.dumps(record))
    run = {"generated": "2026-09-07T00:00:00+00:00", "producer": {"commit": "a" * 40},
           "channels": [f"channels/ch{CHANNEL}_fid{FREQ_ID}.json"]}
    if root is not None:
        run["products_dir"] = str(root / "products" / "_per_pilot")
    (ledger / "run.json").write_text(json.dumps(run))
    return core.load_run(tmp_path)


# ------------------------------------------------------------------ discovery
def test_discover_pairs_the_qualification_products_by_freq_id(tmp_path):
    root = _campaign(tmp_path)
    reference, campaign, reason = m.discover(root)
    assert reason == "" and reference.name == "ref_local.npz" and campaign.parent.name == "_per_pilot"


def test_discover_says_what_is_missing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert m.discover(empty)[2].startswith("no qualification/")
    root = _campaign(tmp_path)
    (root / "qualification" / "ref_local.npz").unlink()
    reference, campaign, reason = m.discover(root)
    assert reference is None and campaign is None and "no freq_id built both ways" in reason


def test_discover_refuses_an_ambiguous_pair(tmp_path):
    root = _campaign(tmp_path)
    other = root / "qualification" / "ref_other.npz"
    v5_fixture._write_product(other, 15, frames=4, units=2)
    v5_fixture._write_product(root / "qualification" / "smoke_run" / "_per_pilot" / "829.npz", 15, frames=4, units=2)
    assert "more than one cross-built freq_id" in m.discover(root)[2]


# ------------------------------------------------------------------ the record
def test_the_two_cohorts_agree_on_every_integer(tmp_path):
    root = _campaign(tmp_path, reference_changes={"detector_version": _detector("e" * 64, "f" * 64)},
                     campaign_changes={"detector_version": _detector("3" * 64, "9" * 64)})
    run = _run(tmp_path, root)
    rec = m.record(run)
    assert rec.reason == "" and len(rec.arms) == 2
    qualification, of_record = rec.arms
    assert qualification.arm == m.ARM_QUALIFICATION and of_record.arm == m.ARM_RUN
    for arm in rec.arms:
        assert arm.joined_events == 4 and arm.joined_frames == 12
        assert arm.missing_frames == 0 and arm.mismatches == 0
        assert arm.event_completeness == 1.0 and arm.frame_completeness == 1.0
        # every compared member is present in the fixture except the stored spectra, which it does not write
        absent = {c.name for c in arm.comparisons if not c.present}
        assert absent == {"psd_frame_db_i16"}
        assert arm.compared > 0
    # the manifests differ exactly where the builds differ
    manifests = {c: b.manifest for c, b in rec.builds.items()}
    assert manifests[m.REFERENCE]["analyzer source digest"] != manifests[m.CAMPAIGN]["analyzer source digest"]
    assert manifests[m.REFERENCE]["detector binary sha256"] != manifests[m.CAMPAIGN]["detector binary sha256"]
    assert manifests[m.REFERENCE]["weight bank sha256"] == manifests[m.CAMPAIGN]["weight bank sha256"]
    assert manifests[m.REFERENCE]["product file sha256"] == "a" * 64


def test_a_planted_integer_mismatch_is_counted_and_localised(tmp_path):
    root = _campaign(tmp_path)
    with np.load(root / "qualification" / "smoke_run" / "_per_pilot" / f"{FREQ_ID}.npz") as archive:
        target = np.array(archive["p_target_u64"], copy=True)
        fine = np.array(archive["fine_power_u64"], copy=True)
    target[3, 0] += 1
    fine[1, 0, 7] += 2
    fine[1, 0, 8] += 2
    v5_fixture._replace(root / "qualification" / "smoke_run" / "_per_pilot" / f"{FREQ_ID}.npz",
                        p_target_u64=target, fine_power_u64=fine)
    rec = m.record(_run(tmp_path, root))
    qualification = rec.arms[0]
    by_name = {c.name: c for c in qualification.comparisons}
    assert by_name["p_target_u64"].mismatches == 1 and by_name["fine_power_u64"].mismatches == 2
    assert by_name["p_ref_sum_u64"].mismatches == 0
    assert qualification.mismatches == 3
    assert rec.arms[1].mismatches == 0                 # the run of record still matches the reference


def test_a_disjoint_pair_reports_what_can_still_be_compared(tmp_path):
    """No shared archive key: the join is empty and the arm says so instead of comparing nothing quietly."""
    root = _campaign(tmp_path)
    smoke = root / "qualification" / "smoke_run" / "_per_pilot" / f"{FREQ_ID}.npz"
    with np.load(smoke) as archive:
        keys = np.array([str(k) + "-other" for k in np.asarray(archive["unit_keys"]).reshape(-1)])
    v5_fixture._replace(smoke, unit_keys=keys, unit_order=keys)
    rec = m.record(_run(tmp_path, root))
    qualification = rec.arms[0]
    assert qualification.joined_frames == 0 and qualification.compared == 0
    assert "share no frame" in qualification.disjoint_reason
    assert qualification.reference.events == 4 and qualification.other.events == 4
    frag = m.build(_run(tmp_path, root))
    assert "do not overlap and cannot be compared frame by frame" in " ".join(frag.notes)
    assert core.DASH in frag.tex
    assert core.DASH in m.build_fields(_run(tmp_path, root)).tex


def test_missing_and_invalid_records_are_counted(tmp_path):
    root = _campaign(tmp_path, survey_frames=8, survey_units=4)
    with np.load(root / "products" / "_per_pilot" / f"{FREQ_ID}.npz") as archive:
        valid = np.array(archive["valid"], copy=True)
    valid[0, 0] = 0
    v5_fixture._replace(root / "products" / "_per_pilot" / f"{FREQ_ID}.npz", valid=valid)
    rec = m.record(_run(tmp_path, root))
    of_record = rec.arms[1]
    assert of_record.reference.frames == 12 and of_record.other.frames == 8
    assert of_record.joined_frames == 8 and of_record.missing_frames == 4
    assert of_record.frame_completeness == pytest.approx(8 / 12)
    assert of_record.invalid_records == 1
    assert {c.name: c.mismatches for c in of_record.comparisons}["valid"] == 1


def test_derived_floats_are_reported_outside_the_claim(tmp_path):
    root = _campaign(tmp_path)
    smoke = root / "qualification" / "smoke_run" / "_per_pilot" / f"{FREQ_ID}.npz"
    with np.load(smoke) as archive:
        db = np.array(archive["normalized_coarse_power_ratio_db"], copy=True)
    db[2, 0] = np.nextafter(db[2, 0], np.inf)
    v5_fixture._replace(smoke, normalized_coarse_power_ratio_db=db)
    rec = m.record(_run(tmp_path, root))
    qualification = rec.arms[0]
    assert qualification.mismatches == 0                                  # the integers are untouched
    floats = {c.name: c.mismatches for c in qualification.float_comparisons}
    assert floats["normalized_coarse_power_ratio_db"] == 1
    assert qualification.float_max_ulps["normalized_coarse_power_ratio_db"] == pytest.approx(1.0, abs=0.5)
    frag = m.build(_run(tmp_path, root))
    assert "outside the claim" in " ".join(frag.notes)


def test_the_stored_spectra_can_be_dropped(tmp_path):
    root = _campaign(tmp_path)
    rec = m.record(_run(tmp_path, root), stored_spectra=False)
    assert any("stored spectra disabled" in reason for reason in rec.arms[0].absent_fields)


# --------------------------------------------------------- independent re-execution
def test_reexecution_keeps_only_the_channels_two_sessions_built(tmp_path):
    rows, reason = m.reexecutions(_campaign(tmp_path))
    assert reason == "" and [r.freq_id for r in rows] == [813, 844]
    by_id = {r.freq_id: r for r in rows}
    assert by_id[844].mismatches == 0 and by_id[844].units == 100
    assert by_id[813].mismatches == 1                    # completed 44 against 43
    assert by_id[813].second == "shard3" and by_id[813].owner == "shard1"


def test_reexecution_says_why_there_are_no_rows(tmp_path):
    root = _campaign(tmp_path, summary=False)
    rows, reason = m.reexecutions(root)
    assert rows == [] and "summary.csv" in reason


# ------------------------------------------------------------------ the fragment
def test_the_chapter_fragment_prints_its_panels_and_keys_every_number(tmp_path):
    root = _campaign(tmp_path, reference_changes={"detector_version": _detector("e" * 64, "f" * 64)},
                     campaign_changes={"detector_version": _detector("3" * 64, "9" * 64)})
    frag = m.build(_run(tmp_path, root))
    assert frag.name == m.NAME and frag.label == m.LABEL
    assert frag.tex.count(r"\begin{tabular}") == 4      # the stacking box and three panels
    assert frag.tex.count(r"\toprule") == 3
    for caption in ("what both compared builds record identically", "where they differ", "Joined denominator"):
        assert caption in frag.tex
    # the manifest splits by the two compared builds: the digests they differ in, and nothing else
    differing = frag.tex.split("where they differ")[1].split(r"\bottomrule")[0]
    assert "analyzer source" in differing and "detector binary sha256" in differing
    assert "weight bank" not in differing and "product schema" not in differing
    keys = {n.key for n in frag.numbers}
    assert f"{m.PREFIX}.mismatches_total" in keys and f"{m.PREFIX}.integer_values_compared" in keys
    assert f"{m.PREFIX}.joined_frames.{m.ARM_QUALIFICATION}" in keys
    assert f"{m.PREFIX}.joined_frames.{m.ARM_RUN}" in keys
    assert f"{m.PREFIX}.float_mismatches.{m.ARM_QUALIFICATION}" in keys
    assert f"{m.PREFIX}.manifest.detector_binary_sha256.{m.REFERENCE}" in keys
    assert len(keys) == len(frag.numbers)               # NumbersDocument refuses a duplicate key
    total = next(n for n in frag.numbers if n.key == f"{m.PREFIX}.mismatches_total")
    assert total.value == 0 and total.kind == "int"
    digest = next(n for n in frag.numbers if n.key.endswith(f"manifest.product_file_sha256.{m.REFERENCE}"))
    assert digest.value == "a" * 64
    doc = nb.NumbersDocument.new(frag.name, repository="x", commit="y", script="z", generated="w")
    for number in frag.numbers:
        doc.add(number)                                 # raises on a duplicate key


def test_the_companion_fragment_carries_the_members_and_the_reexecution(tmp_path):
    root = _campaign(tmp_path)
    frag = m.build_fields(_run(tmp_path, root))
    assert frag.name == m.LEDGER_NAME and frag.label == m.LEDGER_LABEL
    assert frag.tex.count(r"\begin{tabular}") == 3      # the stacking box and two panels
    for caption in ("Mismatches by compared integer member", "Independent re-execution"):
        assert caption in frag.tex
    keys = {n.key for n in frag.numbers}
    assert f"{m.PREFIX}.compared.fine_power_u64.{m.ARM_QUALIFICATION}" in keys
    assert f"{m.PREFIX}.mismatches.p_target_u64.{m.ARM_RUN}" in keys
    assert f"{m.PREFIX}.reexec.completed.fid844" in keys
    assert f"{m.PREFIX}.reexecution_units" in keys
    # the floats are keyed here but not printed: the chapter table counts them
    assert f"{m.PREFIX}.compared.pilot_excess_db.{m.ARM_QUALIFICATION}" in keys
    assert "pilot" not in frag.tex.split("Mismatches by compared")[1].split(r"\bottomrule")[0]
    assert len(keys) == len(frag.numbers)
    assert m.BUILDERS == (m.build, m.build_fields)


def test_the_two_builders_share_one_join(tmp_path):
    """The join reads gigabytes; the second builder must not repeat it."""
    root = _campaign(tmp_path)
    run = _run(tmp_path, root)
    m._CACHE.clear()
    m.build(run)
    assert len(m._CACHE) == 1
    calls = []
    original = m._record
    m._record = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        m.build_fields(run)
    finally:
        m._record = original
    assert calls == []


def test_absent_sha256sums_dashes_the_digest_row(tmp_path):
    root = _campaign(tmp_path, sums=False)
    frag = m.build(_run(tmp_path, root))
    digest = next(n for n in frag.numbers if n.key.endswith(f"manifest.product_file_sha256.{m.CAMPAIGN}"))
    assert digest.value is None and digest.status == "pending"
    assert "product file sha256 & --" in frag.tex


def test_a_run_without_a_campaign_directory_dashes_everything(tmp_path):
    run = _run(tmp_path, None)
    rec = m.record(run)
    assert rec.arms == [] and "names no readable campaign directory" in rec.reason
    frag = m.build(run)
    assert core.DASH in frag.tex
    assert any("cross-build pair not assembled" in note for note in frag.notes)
    assert next(n for n in frag.numbers if n.key == f"{m.PREFIX}.arms").value == 0


def test_a_run_of_record_without_that_channel_keeps_the_qualification_arm(tmp_path):
    root = _campaign(tmp_path)
    (root / "products" / "_per_pilot" / f"{FREQ_ID}.npz").unlink()
    rec = m.record(_run(tmp_path, root))
    assert [a.arm for a in rec.arms] == [m.ARM_QUALIFICATION]
    assert "carries no product for freq_id" in rec.reason


# ------------------------------------------------------------------ the real run
@pytest.mark.skipif(not (REAL_RUN / "ledger" / "run.json").exists(), reason="the archive run of record is not on this machine")
def test_real_run_reproduces_the_campaign_cross_build():
    run = core.load_run(REAL_RUN)
    m._CACHE.clear()
    frag = m.build(run)
    values = {n.key: n.value for n in frag.numbers}
    assert values[f"{m.PREFIX}.arms"] == 2
    assert values[f"{m.PREFIX}.mismatches_total"] == 0
    assert values[f"{m.PREFIX}.joined_events.{m.ARM_QUALIFICATION}"] == 8
    assert values[f"{m.PREFIX}.joined_frames.{m.ARM_QUALIFICATION}"] == 22
    assert values[f"{m.PREFIX}.event_completeness.{m.ARM_QUALIFICATION}"] == 1.0
    assert values[f"{m.PREFIX}.joined_frames.{m.ARM_RUN}"] == 22
    assert values[f"{m.PREFIX}.float_mismatches.{m.ARM_QUALIFICATION}"] == 8
    companion = {n.key: n.value for n in m.build_fields(run).numbers}
    assert companion[f"{m.PREFIX}.compared.psd_frame_db_i16.{m.ARM_QUALIFICATION}"] == 22 * 16384
    assert companion[f"{m.PREFIX}.mismatches.fine_power_u64.{m.ARM_RUN}"] == 0
    assert companion[f"{m.PREFIX}.reexecution_channels"] == 4
    assert companion[f"{m.PREFIX}.reexecution_units"] == 34307
    assert companion[f"{m.PREFIX}.reexecution_mismatches"] == 0
    # the builds differ in the two axes the products record, and agree on the weights
    reference = values[f"{m.PREFIX}.manifest.analyzer_source_digest.{m.REFERENCE}"]
    campaign = values[f"{m.PREFIX}.manifest.analyzer_source_digest.{m.CAMPAIGN}"]
    assert reference != campaign
    assert values[f"{m.PREFIX}.manifest.detector_binary_sha256.{m.REFERENCE}"] != \
        values[f"{m.PREFIX}.manifest.detector_binary_sha256.{m.CAMPAIGN}"]
    assert values[f"{m.PREFIX}.manifest.weight_bank_sha256.{m.REFERENCE}"] == \
        values[f"{m.PREFIX}.manifest.weight_bank_sha256.{m.CAMPAIGN}"]
    # the campaign build and the run of record's product are the same build
    assert values[f"{m.PREFIX}.manifest.analyzer_source_digest.{m.RUN_OF_RECORD}"] == campaign
