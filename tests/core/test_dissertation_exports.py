from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from pilot_proxy.records.chime_atsc_2026.dissertation_export_v1 import (
    ExportError,
    create_export,
    main,
    verify_export,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git(repo: Path, *arguments: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "snapshot_id": "test-snapshot",
        "epoch_operating_points": [
            {
                "channel": 32,
                "epoch_key": "pre",
                "epoch_group": 0,
                "epoch_label": "pre",
                "survey_mask_fraction": 0.9,
                "fine_mask_fraction": None,
                "residual_ratio": None,
                "retained_frames": 7,
                "status": "failure",
                "evidence_state": "measured",
                "note": "test",
            }
        ],
        "channel_status_groups": [
            {
                "channels": list(range(14, 37)),
                "status": "unmeasured",
                "secondary_status": "",
                "epoch_scope": "all",
                "evidence_state": "pending",
                "note": "test",
            }
        ],
        "bao_policy_case": {
            "channel": 33,
            "residual_tolerance": 0.0015,
            "correlation_time_limit_minutes": 5.0,
            "policies": [
                {
                    "policy_key": "pilot_proxy",
                    "label": "pilot proxy",
                    "residual_multiple": 24,
                    "time_multiple": 5.3,
                    "evidence_state": "measured+modeled",
                    "note": "test",
                }
            ],
        },
    }


def _mock_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    census = repo / "data/census/census.csv"
    fields = [
        "schema_version",
        "rf_channel",
        "callsign",
        "service_class",
        "detectability_db",
        "distance_km",
        "bearing_deg",
        "frequency_tolerance",
        "chime_ch_index",
        "nominal_pilot_mhz",
        "city",
        "state_prov",
        "evidence_status",
    ]
    _write_csv(
        census,
        fields,
        [
            {
                "schema_version": "dtv_transmitter_census_v1",
                "rf_channel": 30,
                "callsign": "NEAR",
                "service_class": "Relay",
                "detectability_db": 70,
                "distance_km": 100.0,
                "bearing_deg": 10.0,
                "frequency_tolerance": "±1 kHz",
                "chime_ch_index": 598,
                "nominal_pilot_mhz": 566.3,
                "city": "Near",
                "state_prov": "BC",
                "evidence_status": "reported_on_air_licensed",
            },
            {
                "schema_version": "dtv_transmitter_census_v1",
                "rf_channel": 31,
                "callsign": "FAR",
                "service_class": "Relay",
                "detectability_db": 60,
                "distance_km": 250.0,
                "bearing_deg": 20.0,
                "frequency_tolerance": "±1 kHz",
                "chime_ch_index": 583,
                "nominal_pilot_mhz": 572.3,
                "city": "Far",
                "state_prov": "BC",
                "evidence_status": "licensed_candidate",
            },
            {
                "schema_version": "dtv_transmitter_census_v1",
                "rf_channel": 29,
                "callsign": "NEAREST",
                "service_class": "Full-power",
                "detectability_db": 80,
                "distance_km": 50.0,
                "bearing_deg": 5.0,
                "frequency_tolerance": "±1 kHz",
                "chime_ch_index": 614,
                "nominal_pilot_mhz": 560.3,
                "city": "Nearest",
                "state_prov": "BC",
                "evidence_status": "reported_on_air_unverified",
            },
        ],
    )
    summary = repo / "data/provenance/test_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(_summary()), encoding="utf-8")
    producer = repo / "src/producer.py"
    producer.parent.mkdir(parents=True, exist_ok=True)
    producer.write_text('PRODUCER = "test"\n', encoding="utf-8")
    _git(repo, "init", "--quiet")
    _git(repo, "add", "data/census/census.csv")
    _git(repo, "add", "data/provenance/test_summary.json")
    _git(repo, "add", "src/producer.py")
    _git(
        repo,
        "-c",
        "user.name=PilotProxy tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    return repo, summary


def test_partial_export_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    repo, summary = _mock_repo(tmp_path)
    output = tmp_path / "out"
    manifest = create_export(
        repo_root=repo,
        output_dir=output,
        summary_path=summary,
        source_commit=_git(repo, "rev-parse", "HEAD"),
    )

    assert manifest["complete"] is False
    assert manifest["source"]["commit"] == _git(repo, "rev-parse", "HEAD")
    assert verify_export(output)["schema"]["version"] == 1

    with (output / "census_inner_120mi.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["callsign"] for row in rows] == ["NEAREST", "NEAR"]

    with (output / "census_full_500mi.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames is not None
    assert reader.fieldnames[0] == "schema_version"
    assert reader.fieldnames[-1] == "evidence_status"
    assert [row["callsign"] for row in rows] == ["NEAREST", "NEAR", "FAR"]
    assert [row["evidence_status"] for row in rows] == [
        "reported_on_air_unverified",
        "reported_on_air_licensed",
        "licensed_candidate",
    ]

    statuses = {
        record["path"]: record["status"] for record in manifest["artifacts"]
    }
    assert statuses["census_psd.csv"] == "pending"
    assert statuses["channel_status.csv"] == "available"


def test_default_export_uses_current_23_channel_summary(tmp_path: Path) -> None:
    source_repo = Path(__file__).resolve().parents[2]
    repo = tmp_path / "repo"
    census = repo / "data/census/census.csv"
    summary = repo / "data/provenance/dissertation_summary_v3.json"
    census.parent.mkdir(parents=True)
    summary.parent.mkdir(parents=True)
    shutil.copy2(source_repo / "data/census/census.csv", census)
    shutil.copy2(
        source_repo / "data/provenance/dissertation_summary_v3.json",
        summary,
    )
    _git(repo, "init", "--quiet")
    _git(repo, "add", "data/census/census.csv")
    _git(repo, "add", "data/provenance/dissertation_summary_v3.json")
    _git(
        repo,
        "-c",
        "user.name=PilotProxy tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    output = tmp_path / "default-export"

    assert main(
        [
            "--repo-root",
            str(repo),
            "--output-dir",
            str(output),
        ]
    ) == 0

    with (output / "channel_status.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        status_rows = list(csv.DictReader(handle))
    assert [int(row["channel"]) for row in status_rows] == list(range(14, 37))

    with (output / "bao_policy_case.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        policy_rows = list(csv.DictReader(handle))
    keep = next(row for row in policy_rows if row["policy_key"] == "keep_everything")
    assert keep["residual_multiple"] == "1566"
    assert keep["time_multiple"] == "3.35"

    with (output / "census_full_500mi.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        census_rows = list(csv.DictReader(handle))
    assert {row["schema_version"] for row in census_rows} == {
        "dtv_transmitter_census_v1"
    }
    assert Counter(row["evidence_status"] for row in census_rows) == {
        "reported_on_air_unverified": 421,
        "reported_on_air_licensed": 67,
        "licensed_candidate": 11,
    }


def test_optional_table_is_normalised_and_hashed(tmp_path: Path) -> None:
    repo, summary = _mock_repo(tmp_path)
    supplied = tmp_path / "census_psd_input.csv"
    _write_csv(
        supplied,
        ["channel", "offset_khz", "db_rel_median", "ignored"],
        [
            {"channel": 28, "offset_khz": 1, "db_rel_median": 2, "ignored": "x"},
            {"channel": 27, "offset_khz": -1, "db_rel_median": 3, "ignored": "y"},
        ],
    )
    output = tmp_path / "out"
    manifest = create_export(
        repo_root=repo,
        output_dir=output,
        summary_path=summary,
        source_commit=_git(repo, "rev-parse", "HEAD"),
        optional_inputs={"census_psd": supplied},
    )
    record = next(r for r in manifest["artifacts"] if r["path"] == "census_psd.csv")
    assert record["status"] == "available"
    with (output / "census_psd.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == ["channel", "offset_khz", "db_rel_median"]
    assert [row["channel"] for row in rows] == ["27", "28"]


def test_verify_detects_tampering(tmp_path: Path) -> None:
    repo, summary = _mock_repo(tmp_path)
    output = tmp_path / "out"
    create_export(
        repo_root=repo,
        output_dir=output,
        summary_path=summary,
        source_commit=_git(repo, "rev-parse", "HEAD"),
    )
    with (output / "channel_status.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ExportError, match="SHA-256 mismatch"):
        verify_export(output)


def test_require_complete_refuses_pending_tables(tmp_path: Path) -> None:
    repo, summary = _mock_repo(tmp_path)
    with pytest.raises(ExportError, match="complete export requested"):
        create_export(
            repo_root=repo,
            output_dir=tmp_path / "out",
            summary_path=summary,
            source_commit=_git(repo, "rev-parse", "HEAD"),
            require_complete=True,
        )


@pytest.mark.parametrize("staged", [False, True])
def test_export_refuses_tracked_worktree_changes(
    tmp_path: Path,
    staged: bool,
) -> None:
    repo, summary = _mock_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    producer = repo / "src/producer.py"
    producer.write_text('PRODUCER = "locally-modified"\n', encoding="utf-8")
    if staged:
        _git(repo, "add", "src/producer.py")

    output = tmp_path / "out"
    with pytest.raises(ExportError, match="dirty PilotProxy worktree"):
        create_export(
            repo_root=repo,
            output_dir=output,
            summary_path=summary,
            source_commit=head,
        )
    assert not output.exists()


def test_source_commit_is_an_assertion_not_an_override(tmp_path: Path) -> None:
    repo, summary = _mock_repo(tmp_path)
    previous = _git(repo, "rev-parse", "HEAD")
    note = repo / "README.md"
    note.write_text("second clean commit\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(
        repo,
        "-c",
        "user.name=PilotProxy tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "advance fixture",
    )

    with pytest.raises(ExportError, match="but PilotProxy HEAD is"):
        create_export(
            repo_root=repo,
            output_dir=tmp_path / "out",
            summary_path=summary,
            source_commit=previous,
        )
