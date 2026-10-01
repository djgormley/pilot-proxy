"""Versioned data exports for dissertation figures.

This module defines the boundary between PilotProxy's scientific analysis and
an independently buildable dissertation.  It writes small, schema-checked CSV
products plus a manifest; it never writes figure artwork and the dissertation
never imports :mod:`pilot_proxy` at build time.

The default export is intentionally allowed to be partial.  Tables that require
large archived products or an external Fisher forecast are recorded as
``pending`` instead of being replaced with inferred, digitized, or synthetic
values.  Use ``--require-complete`` when preparing a final archival snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_NAME = "pilot-proxy-dissertation-export"
SCHEMA_VERSION = 1
# The module path when the v1 format was frozen; exports record it as their producer.
PRODUCER = "pilot_proxy.dissertation_exports"
PRODUCER_VERSION = 1
DEFAULT_REPOSITORY = "WVURAIL/pilot-proxy"
DEFAULT_CENSUS_RADIUS_MILES = 120.0
FULL_CENSUS_RADIUS_MILES = 500.0
KM_PER_MILE = 1.609344


class ExportError(RuntimeError):
    """Raised when an export cannot be created or verified safely."""


@dataclass(frozen=True)
class TableSpec:
    """Contract for one optionally supplied scientific table."""

    path: str
    argument: str
    columns: tuple[str, ...]
    sort_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    owner: str
    authority: str
    description: str


OPTIONAL_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        path="census_psd.csv",
        argument="census_psd",
        columns=("channel", "offset_khz", "db_rel_median"),
        sort_columns=("channel", "offset_khz"),
        numeric_columns=("channel", "offset_khz", "db_rel_median"),
        owner="pilot-proxy",
        authority="archive-derived",
        description="Archive-averaged pilot-region spectra used for the transmitter-census comparison.",
    ),
    TableSpec(
        path="worked_example_spectra.csv",
        argument="worked_example_spectra",
        columns=("panel", "fine_bin", "T"),
        sort_columns=("panel", "fine_bin"),
        numeric_columns=("fine_bin", "T"),
        owner="pilot-proxy",
        authority="archive-derived",
        description="Fine-spectrum rows for the worked detector example.",
    ),
    TableSpec(
        path="bao_time_vs_masking.csv",
        argument="bao_time_vs_masking",
        columns=("series", "masked_fraction", "time_year"),
        sort_columns=("series", "masked_fraction"),
        numeric_columns=("masked_fraction", "time_year"),
        owner="external-fisher-forecast",
        authority="forecast-derived",
        description="Uniform-mask observing-time curves from the BAO forecast code.",
    ),
    TableSpec(
        path="bao_convergence.csv",
        argument="bao_convergence",
        columns=("panel", "series", "time_year", "value"),
        sort_columns=("panel", "series", "time_year"),
        numeric_columns=("time_year", "value"),
        owner="external-fisher-forecast",
        authority="forecast-derived",
        description="Clean-error and residual-bias time-scaling curves.",
    ),
    TableSpec(
        path="bao_two_walls.csv",
        argument="bao_two_walls",
        columns=("channel", "evidence", "order", "masked_fraction", "r_over_rtol"),
        sort_columns=("channel", "order"),
        numeric_columns=("channel", "order", "masked_fraction", "r_over_rtol"),
        owner="pilot-proxy+external-fisher-forecast",
        authority="combined-analysis",
        description="Per-channel threshold sweeps in occupancy-versus-residual space.",
    ),
)

CENSUS_SOURCE_SCHEMA = "dtv_transmitter_census_v1"
CENSUS_EVIDENCE_STATES = frozenset(
    {
        "reported_on_air_unverified",
        "reported_on_air_licensed",
        "licensed_candidate",
    }
)

CENSUS_COLUMNS = (
    "schema_version",
    "rf_channel",
    "callsign",
    "service_class",
    "distance_km",
    "bearing_deg",
    "frequency_tolerance",
    "city",
    "state_prov",
    "evidence_status",
)

EPOCH_COLUMNS = (
    "channel",
    "epoch_key",
    "epoch_group",
    "epoch_label",
    "survey_mask_fraction",
    "fine_mask_fraction",
    "residual_ratio",
    "retained_frames",
    "status",
    "evidence_state",
    "note",
)

STATUS_COLUMNS = (
    "channel",
    "status",
    "secondary_status",
    "epoch_scope",
    "evidence_state",
    "note",
)

POLICY_COLUMNS = (
    "channel",
    "policy_key",
    "label",
    "residual_tolerance",
    "residual_multiple",
    "time_multiple",
    "correlation_time_limit_minutes",
    "evidence_state",
    "note",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportError(f"required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExportError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExportError(f"JSON root must be an object: {path}")
    return data


def _read_csv(path: Path, required_columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            missing = [column for column in required_columns if column not in fields]
            if missing:
                raise ExportError(
                    f"{path} is missing required columns: {', '.join(missing)}"
                )
            rows = [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise ExportError(f"CSV file does not exist: {path}") from exc
    return rows


def _normalise_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExportError(f"non-finite numeric value in export: {value!r}")
        return format(value, ".12g")
    return str(value)


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> int:
    materialised = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in materialised:
            writer.writerow({column: _normalise_value(row.get(column)) for column in columns})
    return len(materialised)


def _sort_key(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[Any, ...]:
    values: list[Any] = []
    for column in columns:
        raw = row.get(column, "")
        text = "" if raw is None else str(raw).strip()
        try:
            values.append((0, float(text)))
        except ValueError:
            values.append((1, text.casefold()))
    return tuple(values)


def _display_input(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _run_git(repo_root: Path, *arguments: str) -> str:
    """Run one read-only Git query or fail the provenance check closed."""

    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ExportError(
            f"cannot establish PilotProxy Git provenance in {repo_root}: {exc}"
        ) from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "Git command failed"
        raise ExportError(
            f"cannot establish PilotProxy Git provenance in {repo_root}: {detail}"
        )
    return proc.stdout


def _clean_source_commit(repo_root: Path, expected_commit: str | None) -> str:
    """Return ``HEAD`` only when it honestly describes the source checkout.

    Dissertation exports attribute their producer and tracked scientific inputs
    to a Git commit. Recording ``HEAD`` from a dirty checkout would silently
    omit the actual source state, so both staged and unstaged tracked changes
    are release-blocking. ``expected_commit`` is an assertion, not an escape
    hatch: it must resolve to the current ``HEAD``.
    """

    top_level_text = _run_git(repo_root, "rev-parse", "--show-toplevel").strip()
    if not top_level_text:
        raise ExportError(f"Git returned no repository root for {repo_root}")
    top_level = Path(top_level_text).resolve()
    if top_level != repo_root:
        raise ExportError(
            f"--repo-root must be the PilotProxy Git root: got {repo_root}, "
            f"Git root is {top_level}"
        )

    head = _run_git(
        repo_root,
        "rev-parse",
        "--verify",
        "--end-of-options",
        "HEAD^{commit}",
    ).strip()
    if not head:
        raise ExportError(f"Git returned no HEAD commit for {repo_root}")

    if expected_commit is not None:
        try:
            expected = _run_git(
                repo_root,
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{expected_commit}^{{commit}}",
            ).strip()
        except ExportError as exc:
            raise ExportError(
                f"--source-commit does not identify a commit in {repo_root}: "
                f"{expected_commit!r}"
            ) from exc
        if expected != head:
            raise ExportError(
                f"--source-commit resolves to {expected}, but PilotProxy HEAD is {head}"
            )

    status = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    if status:
        changed = [line for line in status.splitlines() if line]
        preview = ", ".join(changed[:5])
        if len(changed) > 5:
            preview += f", ... ({len(changed)} tracked paths total)"
        raise ExportError(
            "cannot create a commit-attributed dissertation export from a dirty "
            f"PilotProxy worktree; commit or restore tracked changes first: {preview}"
        )
    return head


def _validate_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != 1:
        raise ExportError("dissertation summary must declare schema_version=1")

    epochs = summary.get("epoch_operating_points")
    if not isinstance(epochs, list) or not epochs:
        raise ExportError("summary epoch_operating_points must be a non-empty list")
    required_epoch = set(EPOCH_COLUMNS) - {"retained_frames", "fine_mask_fraction", "residual_ratio"}
    for index, row in enumerate(epochs):
        if not isinstance(row, dict):
            raise ExportError(f"epoch_operating_points[{index}] is not an object")
        missing = sorted(required_epoch - set(row))
        if missing:
            raise ExportError(
                f"epoch_operating_points[{index}] is missing: {', '.join(missing)}"
            )

    groups = summary.get("channel_status_groups")
    if not isinstance(groups, list) or not groups:
        raise ExportError("summary channel_status_groups must be a non-empty list")
    seen: set[int] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ExportError(f"channel_status_groups[{index}] is not an object")
        channels = group.get("channels")
        if not isinstance(channels, list) or not channels:
            raise ExportError(f"channel_status_groups[{index}].channels must be non-empty")
        for value in channels:
            channel = int(value)
            if channel in seen:
                raise ExportError(f"channel {channel} appears in more than one status group")
            seen.add(channel)
    expected = set(range(14, 37))
    if seen != expected:
        raise ExportError(
            "channel status groups must cover physical channels 14-36 exactly; "
            f"missing={sorted(expected-seen)}, extra={sorted(seen-expected)}"
        )

    case = summary.get("bao_policy_case")
    if not isinstance(case, dict):
        raise ExportError("summary bao_policy_case must be an object")
    policies = case.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ExportError("bao_policy_case.policies must be a non-empty list")
    keys: set[str] = set()
    for index, policy in enumerate(policies):
        if not isinstance(policy, dict):
            raise ExportError(f"bao_policy_case.policies[{index}] is not an object")
        key = str(policy.get("policy_key", ""))
        if not key:
            raise ExportError(f"bao_policy_case.policies[{index}] has no policy_key")
        if key in keys:
            raise ExportError(f"duplicate BAO policy_key: {key}")
        keys.add(key)


def _census_rows(source: Path, radius_miles: float) -> list[dict[str, str]]:
    if radius_miles <= 0:
        raise ExportError("census radius must be positive")
    source_rows = _read_csv(source, CENSUS_COLUMNS)
    radius_km = radius_miles * KM_PER_MILE
    selected: list[dict[str, str]] = []
    for index, row in enumerate(source_rows, start=2):
        if row["schema_version"].strip() != CENSUS_SOURCE_SCHEMA:
            raise ExportError(
                f"unsupported census schema in {source} line {index}: "
                f"{row['schema_version']!r}"
            )
        evidence_status = row["evidence_status"].strip()
        if evidence_status not in CENSUS_EVIDENCE_STATES:
            raise ExportError(
                f"unsupported census evidence_status in {source} line {index}: "
                f"{evidence_status!r}"
            )
        try:
            distance = float(row["distance_km"])
            bearing = float(row["bearing_deg"])
            # Parsing validates the channel even though the original text is
            # retained in the exported row.
            _ = int(row["rf_channel"])
        except (TypeError, ValueError) as exc:
            raise ExportError(
                f"invalid census numeric value in {source} line {index}"
            ) from exc
        if not (math.isfinite(distance) and math.isfinite(bearing)):
            raise ExportError(f"non-finite census coordinate in {source} line {index}")
        if distance <= radius_km + 1e-9:
            selected.append(
                {
                    column: row.get(column, "").strip()
                    for column in CENSUS_COLUMNS
                }
            )
    selected.sort(
        key=lambda row: (
            float(row["distance_km"]),
            int(row["rf_channel"]),
            row["callsign"].casefold(),
        )
    )
    return selected


def _epoch_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summary["epoch_operating_points"]]
    rows.sort(key=lambda row: (int(row["channel"]), str(row["epoch_group"])))
    return rows


def _status_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in summary["channel_status_groups"]:
        for channel in group["channels"]:
            rows.append(
                {
                    "channel": int(channel),
                    "status": group["status"],
                    "secondary_status": group.get("secondary_status", ""),
                    "epoch_scope": group.get("epoch_scope", "all"),
                    "evidence_state": group["evidence_state"],
                    "note": group.get("note", ""),
                }
            )
    rows.sort(key=lambda row: int(row["channel"]))
    return rows


def _policy_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    case = summary["bao_policy_case"]
    rows: list[dict[str, Any]] = []
    for policy in case["policies"]:
        rows.append(
            {
                "channel": case["channel"],
                "policy_key": policy["policy_key"],
                "label": policy["label"],
                "residual_tolerance": case["residual_tolerance"],
                "residual_multiple": policy["residual_multiple"],
                "time_multiple": policy["time_multiple"],
                "correlation_time_limit_minutes": case[
                    "correlation_time_limit_minutes"
                ],
                "evidence_state": policy["evidence_state"],
                "note": policy.get("note", ""),
            }
        )
    return rows


def _artifact_record(
    path: Path,
    *,
    relative_path: str,
    rows: int,
    owner: str,
    authority: str,
    description: str,
    source_inputs: Sequence[str],
) -> dict[str, Any]:
    return {
        "path": relative_path,
        "status": "available",
        "owner": owner,
        "authority": authority,
        "description": description,
        "rows": rows,
        "sha256": _sha256(path),
        "source_inputs": list(source_inputs),
    }


def _pending_record(spec: TableSpec) -> dict[str, Any]:
    return {
        "path": spec.path,
        "status": "pending",
        "owner": spec.owner,
        "authority": spec.authority,
        "description": spec.description,
        "rows": None,
        "sha256": None,
        "source_inputs": [],
        "required_option": "--" + spec.argument.replace("_", "-"),
    }


def _normalise_optional_table(
    source: Path,
    destination: Path,
    spec: TableSpec,
) -> int:
    rows = _read_csv(source, spec.columns)
    normalised = [
        {column: row.get(column, "").strip() for column in spec.columns}
        for row in rows
    ]
    for index, row in enumerate(normalised, start=2):
        for column in spec.numeric_columns:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise ExportError(
                    f"invalid numeric value in {source} line {index}, column {column}"
                ) from exc
            if not math.isfinite(value):
                raise ExportError(
                    f"non-finite numeric value in {source} line {index}, column {column}"
                )
    normalised.sort(key=lambda row: _sort_key(row, spec.sort_columns))
    return _write_csv(destination, spec.columns, normalised)


def _write_export_readme(path: Path, manifest: Mapping[str, Any]) -> None:
    available = [a for a in manifest["artifacts"] if a["status"] == "available"]
    pending = [a for a in manifest["artifacts"] if a["status"] == "pending"]
    lines = [
        "# PilotProxy dissertation export",
        "",
        f"Schema: `{SCHEMA_NAME}` version {SCHEMA_VERSION}",
        f"Source commit: `{manifest['source']['commit']}`",
        "",
        "This directory is a data interface, not a figure bundle.  Copy or import",
        "it into the dissertation, which applies its own typography and visual style.",
        "",
        "## Available tables",
        "",
    ]
    lines.extend(
        f"- `{record['path']}` - {record['description']}"
        for record in available
    )
    if pending:
        lines += ["", "## Pending tables", ""]
        lines.extend(
            f"- `{record['path']}` - {record['description']} "
            f"(supply `{record['required_option']}`)"
            for record in pending
        )
    lines += [
        "",
        "`export_manifest.json` records source ownership, authority, row counts,",
        "and SHA-256 hashes.  Verify the directory before import with:",
        "",
        "```bash",
        "PYTHONPATH=src python tools/export_dissertation_data.py --verify <this-directory>",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def create_export(
    *,
    repo_root: Path,
    output_dir: Path,
    summary_path: Path,
    source_commit: str | None = None,
    census_radius_miles: float = DEFAULT_CENSUS_RADIUS_MILES,
    optional_inputs: Mapping[str, Path | None] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Create one atomic dissertation export and return its manifest."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    summary_path = summary_path.resolve()
    commit = _clean_source_commit(repo_root, source_commit)
    summary = _read_json(summary_path)
    _validate_summary(summary)
    optional_inputs = dict(optional_inputs or {})

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        artifacts: list[dict[str, Any]] = []

        census_source = repo_root / "data" / "census" / "census.csv"

        full_census_rows = _census_rows(census_source, FULL_CENSUS_RADIUS_MILES)
        full_census_path = temp_root / "census_full_500mi.csv"
        count = _write_csv(full_census_path, CENSUS_COLUMNS, full_census_rows)
        artifacts.append(
            _artifact_record(
                full_census_path,
                relative_path=full_census_path.name,
                rows=count,
                owner="pilot-proxy",
                authority="authoritative-source-export",
                description=(
                    "Conservative maximum-envelope ATSC 1.0 UHF emitter-channel "
                    f"census within {FULL_CENSUS_RADIUS_MILES:g} miles of DRAO; "
                    "schema and per-row evidence status are preserved."
                ),
                source_inputs=[str(census_source.relative_to(repo_root))],
            )
        )

        inner_census_rows = _census_rows(census_source, census_radius_miles)
        inner_census_path = temp_root / "census_inner_120mi.csv"
        count = _write_csv(inner_census_path, CENSUS_COLUMNS, inner_census_rows)
        artifacts.append(
            _artifact_record(
                inner_census_path,
                relative_path=inner_census_path.name,
                rows=count,
                owner="pilot-proxy",
                authority="authoritative-source-subset",
                description=(
                    f"Transmitter-census rows within {census_radius_miles:g} miles "
                    "of DRAO, preserving schema and per-row evidence status."
                ),
                source_inputs=[str(census_source.relative_to(repo_root))],
            )
        )

        epoch_path = temp_root / "epoch_operating_points.csv"
        count = _write_csv(epoch_path, EPOCH_COLUMNS, _epoch_rows(summary))
        artifacts.append(
            _artifact_record(
                epoch_path,
                relative_path=epoch_path.name,
                rows=count,
                owner="pilot-proxy",
                authority="curated-dissertation-snapshot",
                description="Epoch-specific operating points used by the current dissertation draft.",
                source_inputs=[_display_input(summary_path, repo_root)],
            )
        )

        status_path = temp_root / "channel_status.csv"
        count = _write_csv(status_path, STATUS_COLUMNS, _status_rows(summary))
        artifacts.append(
            _artifact_record(
                status_path,
                relative_path=status_path.name,
                rows=count,
                owner="pilot-proxy",
                authority="curated-dissertation-snapshot",
                description="Current 23-channel evidence-status matrix; not a final verdict.",
                source_inputs=[_display_input(summary_path, repo_root)],
            )
        )

        policy_path = temp_root / "bao_policy_case.csv"
        count = _write_csv(policy_path, POLICY_COLUMNS, _policy_rows(summary))
        artifacts.append(
            _artifact_record(
                policy_path,
                relative_path=policy_path.name,
                rows=count,
                owner="pilot-proxy+external-fisher-forecast",
                authority="curated-dissertation-snapshot",
                description="Channel-33 residual-policy comparison used by the current draft.",
                source_inputs=[_display_input(summary_path, repo_root)],
            )
        )

        for spec in OPTIONAL_TABLES:
            supplied = optional_inputs.get(spec.argument)
            if supplied is None:
                artifacts.append(_pending_record(spec))
                continue
            source = Path(supplied).resolve()
            destination = temp_root / spec.path
            count = _normalise_optional_table(source, destination, spec)
            artifacts.append(
                _artifact_record(
                    destination,
                    relative_path=spec.path,
                    rows=count,
                    owner=spec.owner,
                    authority=spec.authority,
                    description=spec.description,
                    source_inputs=[str(source)],
                )
            )

        artifacts.sort(key=lambda record: record["path"])
        # Recheck after reading every input. This closes the window in which a
        # concurrent edit could otherwise be exported under the earlier HEAD.
        _clean_source_commit(repo_root, commit)
        manifest: dict[str, Any] = {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "producer": {"module": PRODUCER, "version": PRODUCER_VERSION},
            "source": {
                "repository": DEFAULT_REPOSITORY,
                "commit": commit,
                "summary_snapshot_id": summary.get("snapshot_id", "unknown"),
                "summary_sha256": _sha256(summary_path),
            },
            "parameters": {
                "census_radius_miles": census_radius_miles,
                "full_census_radius_miles": FULL_CENSUS_RADIUS_MILES,
            },
            "complete": all(record["status"] == "available" for record in artifacts),
            "artifacts": artifacts,
        }
        if require_complete and not manifest["complete"]:
            missing = [record["path"] for record in artifacts if record["status"] != "available"]
            raise ExportError(
                "complete export requested, but these tables are pending: "
                + ", ".join(missing)
            )

        manifest_path = temp_root / "export_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_export_readme(temp_root / "README.md", manifest)

        checksums = [
            f"{record['sha256']}  {record['path']}"
            for record in artifacts
            if record["status"] == "available"
        ]
        checksums += [
            f"{_sha256(manifest_path)}  export_manifest.json",
            f"{_sha256(temp_root / 'README.md')}  README.md",
        ]
        (temp_root / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")

        verify_export(temp_root, require_complete=require_complete)
        # Keep this immediately before publication: a tracked input or producer
        # edit during normalization must not inherit the preflight commit.
        _clean_source_commit(repo_root, commit)

        if output_dir.exists():
            if output_dir.is_dir():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()
        os.replace(temp_root, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def verify_export(export_dir: Path, *, require_complete: bool = False) -> dict[str, Any]:
    """Verify schema, hashes, row counts, and completion state."""

    export_dir = export_dir.resolve()
    manifest_path = export_dir / "export_manifest.json"
    manifest = _read_json(manifest_path)
    schema = manifest.get("schema")
    if schema != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        raise ExportError(f"unsupported export schema in {manifest_path}: {schema!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ExportError("export manifest artifacts must be a list")

    seen: set[str] = set()
    for record in artifacts:
        if not isinstance(record, dict):
            raise ExportError("export manifest contains a non-object artifact record")
        relative = str(record.get("path", ""))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ExportError(f"unsafe artifact path in export manifest: {relative!r}")
        if relative in seen:
            raise ExportError(f"duplicate artifact path in export manifest: {relative}")
        seen.add(relative)
        status = record.get("status")
        if status == "pending":
            if require_complete:
                raise ExportError(f"required export table is pending: {relative}")
            continue
        if status != "available":
            raise ExportError(f"unknown artifact status for {relative}: {status!r}")
        path = export_dir / relative
        if not path.is_file():
            raise ExportError(f"available export artifact is missing: {path}")
        expected_hash = record.get("sha256")
        actual_hash = _sha256(path)
        if expected_hash != actual_hash:
            raise ExportError(
                f"SHA-256 mismatch for {relative}: expected {expected_hash}, got {actual_hash}"
            )
        expected_rows = record.get("rows")
        if expected_rows is not None:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                actual_rows = sum(1 for _ in csv.DictReader(handle))
            if int(expected_rows) != actual_rows:
                raise ExportError(
                    f"row-count mismatch for {relative}: expected {expected_rows}, got {actual_rows}"
                )

    complete = all(record.get("status") == "available" for record in artifacts)
    if bool(manifest.get("complete")) != complete:
        raise ExportError("export manifest complete flag disagrees with artifact statuses")
    if require_complete and not complete:
        raise ExportError("export is partial")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="PilotProxy repository root (default: inferred from this module)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/dissertation/v1"),
        help="export directory to replace atomically",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/provenance/dissertation_summary_v3.json"),
        help=(
            "curated summary snapshot relative to --repo-root unless absolute "
            "(default: current 23-channel v3 snapshot)"
        ),
    )
    parser.add_argument(
        "--source-commit",
        help=(
            "expected PilotProxy commit; the exporter fails unless it resolves "
            "to HEAD and the tracked worktree is clean"
        ),
    )
    parser.add_argument(
        "--census-radius-miles",
        type=float,
        default=DEFAULT_CENSUS_RADIUS_MILES,
    )
    for spec in OPTIONAL_TABLES:
        parser.add_argument(
            "--" + spec.argument.replace("_", "-"),
            dest=spec.argument,
            type=Path,
            help=spec.description,
        )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail if any optional scientific table is unavailable",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="DIR",
        help="verify an existing export instead of creating one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            manifest = verify_export(args.verify, require_complete=args.require_complete)
            state = "complete" if manifest["complete"] else "partial"
            print(f"Verified {state} dissertation export: {args.verify}")
            return 0

        repo_root = args.repo_root.resolve()
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
        summary_path = args.summary
        if not summary_path.is_absolute():
            summary_path = repo_root / summary_path
        optional = {
            spec.argument: getattr(args, spec.argument)
            for spec in OPTIONAL_TABLES
        }
        manifest = create_export(
            repo_root=repo_root,
            output_dir=output_dir,
            summary_path=summary_path,
            source_commit=args.source_commit,
            census_radius_miles=args.census_radius_miles,
            optional_inputs=optional,
            require_complete=args.require_complete,
        )
        state = "complete" if manifest["complete"] else "partial"
        print(f"Wrote {state} dissertation export: {output_dir}")
        for record in manifest["artifacts"]:
            print(f"  {record['status']:9s} {record['path']}")
        return 0
    except ExportError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
