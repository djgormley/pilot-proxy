"""Corrupt one contract at a time in an otherwise complete, valid v5 cohort."""
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.products import acceptance as gate
from pilot_proxy.products.npzio import load_npz
from test_pilotproxy_v5 import _cohort


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    return _cohort(tmp_path_factory.mktemp("acceptance-cohort"))


def _altered(cohort, tmp_path, alter):
    values = load_npz(cohort[0])
    alter(values)
    path = tmp_path / "damaged-ch14.npz"
    np.savez(path, **values)
    return [path, *cohort[1:]]


@pytest.mark.parametrize("field", [
    "valid", "fine_power_u64", "detector_contract_json", "decision_contract_json",
    "weight_bank_sha256", "unit_scope", "source_event_keys", "frame_unit_index",
    "unit_delta_time", "railed_sample_total", "sample_rate_hz", "pilot_frequency_hz",
])
def test_missing_contract_field_refuses_the_complete_cohort(cohort, tmp_path, field):
    paths = _altered(cohort, tmp_path, lambda values: values.pop(field))
    with pytest.raises(gate.ArchiveAcceptanceError, match="damaged-ch14.npz"):
        gate.validate_archive_products(paths)


@pytest.mark.parametrize("field,value", [
    ("fine_status", "disabled"),
    ("fine_num_bins", np.int64(64)),
    ("fine_pad_factor", np.int64(3)),
    ("fine_guard_fine_bins", np.int64(2)),
    ("fine_p_fa", np.float64(0.01)),
    ("fine_designated_bins", np.array([], dtype=np.int64)),
    ("fine_designated_bins", np.array([1, 1], dtype=np.int64)),
    ("fine_census_excluded_bins", np.array([-1], dtype=np.int64)),
    ("fine_power_u64", np.ones((120, 3, 255), dtype=np.uint64)),
    ("nfft", np.int64(8192)),
    ("sense", np.int64(1)),
    ("sample_rate_hz", np.float64(1)),
    ("rational_overflow_count", np.uint64(1)),
    ("detector_version", "no kernel identity"),
    ("weight_bank_sha256", "A" * 64),
    ("weight_manifest_sha256", "bad digest"),
    ("detector_contract_json", "{"),
    ("decision_contract_json", "[]"),
    ("reference_placement_json", "null"),
    ("pilot_frequency_hz", np.array([np.nan])),
    ("pilot_capture_efficiency", np.float64(0)),
    ("frame_unit_index", np.zeros(120, dtype=np.int64)),
    ("frame_unit_index", np.full(120, -1, dtype=np.int32)),
    ("frame_in_unit", np.zeros(120, dtype=np.int32)),
    ("unit_scope", np.full(12, "")),
    ("unit_input_map_sha256", np.full(12, "malformed")),
    ("unit_delta_time", np.full(12, np.nan)),
    ("unit_delta_time", np.full(12, -1.0)),
    ("railed_sample_count", np.full((120, 1), 2**64 - 1, dtype=np.uint64)),
])
def test_corrupt_contract_is_rejected_as_a_named_archive_error(cohort, tmp_path, field, value):
    paths = _altered(cohort, tmp_path, lambda values: values.update({field: value}))
    with pytest.raises(gate.ArchiveAcceptanceError, match="damaged-ch14.npz"):
        gate.validate_archive_products(paths)


def test_valid_cohort_is_order_independent_and_accounting_closes(cohort):
    forward = gate.validate_archive_products(cohort)
    reverse = gate.validate_archive_products(cohort[::-1])
    assert forward == reverse
    assert [p.physical_channel for p in forward.products] == list(range(14, 37))
    assert forward.frame_count == forward.valid_frames + forward.invalid_frames
    assert forward.frame_count == sum(sum(p.frames_by_unit) for p in forward.products)
    assert forward.valid_frames == sum(sum(p.valid_by_unit) for p in forward.products)
    assert forward.unit_count == sum(len(p.frames_by_unit) for p in forward.products)
    assert forward.untimed_frames == forward.untimed_units == 0


def test_json_formatting_is_not_a_new_detector_identity(cohort, tmp_path):
    def reorder(values):
        for field in ("detector_contract_json", "decision_contract_json"):
            parsed = json.loads(str(values[field]))
            values[field] = json.dumps(dict(reversed(list(parsed.items()))), indent=4)

    altered = _altered(cohort, tmp_path, reorder)
    before, after = (gate.validate_archive_products(paths) for paths in (cohort, altered))
    assert after.detector_contract_sha256 == before.detector_contract_sha256
    assert after.decision_contract_sha256 == before.decision_contract_sha256


def test_archive_cli_accepts_directory_and_reports_cohort(cohort, capsys):
    assert gate.main([str(cohort[0].parent)]) == 0
    output = capsys.readouterr().out
    assert "archive cohort accepted" in output and "channels: 23" in output
    assert all(f"ch{channel}:" in output for channel in range(14, 37))


def test_archive_cli_rejects_mixed_paths_and_incomplete_cohorts(cohort, tmp_path, capsys):
    for paths in ([str(cohort[0].parent), str(cohort[0])], [str(tmp_path)]):
        with pytest.raises(SystemExit) as result:
            gate.main(paths)
        assert result.value.code == 2
        assert "error:" in capsys.readouterr().err


def test_non_npz_and_pickled_archives_cannot_enter_a_cohort(cohort, tmp_path):
    path = tmp_path / "unsafe.npz"
    for payload in (b"bad archive", None):
        if payload is None:
            np.savez(path, schema=np.array([{}], dtype=object))
        else:
            path.write_bytes(payload)
        with pytest.raises(gate.ArchiveAcceptanceError, match="unsafe.npz"):
            gate.validate_archive_products([path, *cohort[1:]])
    with path.open("wb") as stream:
        np.save(stream, np.ones(3))
    with pytest.raises(gate.ArchiveAcceptanceError, match="not an NPZ"):
        gate.validate_archive_products([path, *cohort[1:]])


def test_missing_cohort_files_report_paths_before_opening(cohort, tmp_path):
    with pytest.raises(gate.ArchiveAcceptanceError, match="do not exist"):
        gate.validate_archive_products([Path(tmp_path / "missing.npz"), *cohort[1:]])
