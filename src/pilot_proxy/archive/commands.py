"""CHIME archive command workflows."""

from __future__ import annotations

import os
from collections import Counter
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pilot_proxy.chime.baseband_reader import ChimeBasebandReader

from . import pipeline
from .datatrail_client import Datatrail
from .instruments import load_instrument
from .interfaces import RunContext, SurveyUnavailableError
from .inventory import (
    derive_inventory_name,
    read_inventory_meta,
    resolve_inventory,
    write_inventory_meta,
)
from .invpaths import inventory_dir_for_write
from .names import validate_identifier
from .sources import CadcDatatrailSource, LocalDirectorySource

_ARCHIVE_SOURCE = "cadc-datatrail"
_LOCAL_SOURCE = "local"
_SURVEY_READER = "chime-baseband"


def _preflight(
    component: object,
    ctx: RunContext,
    *,
    label: str,
    method: str = "preflight",
) -> None:
    result = getattr(component, method)(ctx)
    ok, problems = bool(result[0]), list(result[1])
    if not ok:
        raise SystemExit(
            f"{label} preflight failed:\n  - " + "\n  - ".join(map(str, problems))
        )


def survey_chime(
    *,
    out_dir: str | Path | None = None,
    name: str | None = None,
    root: str | Path | None = None,
    scope: str | None = None,
    freq_ids: object = None,
    include_outrigger: bool = False,
    workers: int = 12,
    re_enumerate: bool = False,
    max_events: int | None = None,
    empty_age_days: int | None = None,
    strict_completeness: bool = False,
    dry_run: bool = False,
    scopes_only: bool = False,
    match: str | None = None,
    expand: bool = False,
) -> Path:
    """Build or resume a CHIME archive inventory.

    With ``scopes_only`` the walk stops at discovery: it writes a map of the
    datasets each scope lists and returns that path instead of an inventory.
    """
    instrument = load_instrument("chime")
    if workers < 1:
        raise SystemExit("chime-survey: --workers must be positive")
    if max_events is not None and max_events < 1:
        raise SystemExit("chime-survey: --max-events must be positive")
    if empty_age_days is not None and empty_age_days < 0:
        raise SystemExit("chime-survey: --empty-age-days must be zero or greater")
    if out_dir is not None and root is not None:
        raise SystemExit("chime-survey: pass either --out or --root, not both")

    if name is not None:
        try:
            inventory_name = validate_identifier(name, label="inventory name")
        except ValueError as exc:
            raise SystemExit(f"chime-survey: {exc}") from exc
    elif out_dir is not None:
        inventory_name = Path(out_dir).expanduser().resolve().name
    else:
        inventory_name = derive_inventory_name(instrument.name, freq_ids)
    try:
        inventory_name = validate_identifier(inventory_name, label="inventory name")
    except ValueError as exc:
        raise SystemExit(f"chime-survey: {exc}") from exc
    if out_dir is None:
        out = (
            Path(root).expanduser() / "data" / inventory_name
            if root is not None
            else inventory_dir_for_write(inventory_name)
        )
    else:
        out = Path(out_dir).expanduser()

    if not scopes_only:
        for flag, value in (("--match", match), ("--expand", expand)):
            if value:
                print(f"[chime-survey] {flag} applies to --scopes-only; ignoring")

    print(f"[chime-survey] {instrument.name} -> {out}")
    if dry_run:
        print("  dry-run: would survey")
        return out / "inventory.jsonl"

    if not Datatrail.installed():
        raise SystemExit(
            "chime-survey: datatrail-cli is required; install the archive extra"
        )
    api_ok, detail = Datatrail.api_available()
    if not api_ok:
        raise SystemExit(f"chime-survey: {detail}")

    options: dict[str, object] = {
        "scope": scope,
        "freq_ids": freq_ids,
        "include_outrigger": bool(include_outrigger),
        "workers": int(workers),
        "re_enumerate": bool(re_enumerate),
        "max_events": max_events,
        "empty_age_days": empty_age_days,
        "name": inventory_name,
        "scopes_only": bool(scopes_only),
        "match": match,
        "expand": bool(expand),
    }
    reader = ChimeBasebandReader()
    ctx = RunContext(instrument=instrument, options=options, reader=reader)
    source = CadcDatatrailSource()
    _preflight(source, ctx, label="chime-survey")
    _preflight(reader, ctx, label="chime-survey reader")
    try:
        inventory_path = Path(source.survey(ctx, str(out)))
    except SurveyUnavailableError as exc:
        raise SystemExit(f"chime-survey: {exc}") from exc

    if scopes_only:
        # discovery only: the result is a scope map, not a scannable inventory,
        # so it gets no inventory meta and no completeness check
        print(f"  scope map: {inventory_path}")
        return inventory_path

    meta_path = write_inventory_meta(
        inventory_path,
        instrument,
        source=_ARCHIVE_SOURCE,
        reader=_SURVEY_READER,
        freq_ids=freq_ids,
        name=inventory_name,
        scope_request=scope,
    )
    print(f"  inventory: {inventory_path}")
    print(f"  meta: {meta_path}")
    if strict_completeness:
        issues = source.survey_completeness_issues(str(out))
        if issues is None:
            raise SystemExit(
                "chime-survey: source does not expose a completeness check"
            )
        unresolved = {
            str(key): int(value) for key, value in issues.items() if int(value) > 0
        }
        if unresolved:
            detail = ", ".join(
                f"{count} {key}" for key, count in sorted(unresolved.items())
            )
            raise SystemExit("chime-survey: strict completeness failed: " + detail)
        print("  strict completeness: no unresolved omissions")
    return inventory_path


def _resolve_source(
    *,
    source: str | None,
    input_dir: str | Path | None,
    inventory: str | Path | None,
    inventory_name: str | None,
) -> str:
    has_inventory = inventory is not None or inventory_name is not None
    inferred = _ARCHIVE_SOURCE if has_inventory else _LOCAL_SOURCE
    resolved = source or inferred
    if resolved == _LOCAL_SOURCE and has_inventory:
        raise SystemExit("archive inventory options cannot be used with --source local")
    if resolved == _ARCHIVE_SOURCE and input_dir is not None:
        raise SystemExit("--input-dir cannot be used with the archive source")
    if resolved not in {_LOCAL_SOURCE, _ARCHIVE_SOURCE}:
        raise SystemExit(f"unknown source {resolved!r}")
    return resolved


def _source_context(
    *,
    source: str,
    input_dir: str | Path | None,
    inventory: str | Path | None,
    inventory_name: str | None,
    source_root: str | Path | None,
    source_glob: str,
    source_freq_id_regex: str | None,
    source_event_regex: str | None,
    extra_options: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object], Path | None]:
    options = dict(extra_options or {})
    if source == _LOCAL_SOURCE:
        local_root = input_dir if input_dir is not None else source_root
        if local_root is None:
            raise SystemExit("local source needs --input-dir or --source-root")
        options.update(
            source_root=str(local_root),
            source_glob=source_glob,
            source_freq_id_regex=source_freq_id_regex,
            source_event_regex=source_event_regex,
        )
        return LocalDirectorySource(), options, None

    inventory_path = resolve_inventory(
        inventory=inventory, name=inventory_name, root=source_root
    )
    if inventory_path is None:
        raise SystemExit("archive source needs --inventory or --inventory-name")
    options["inventory"] = str(inventory_path)
    return CadcDatatrailSource(), options, inventory_path


def _freq_id_of(unit: object) -> int | None:
    metadata = getattr(unit, "meta", None) or {}
    try:
        value = metadata.get("freq_id")
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if amount < 1024 or unit == "PB":
            return f"{int(amount)} B" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def _saved_product_unit_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        with np.load(path, allow_pickle=False) as product:
            if "unit_keys" not in product.files:
                return set()
            return {
                str(value)
                for value in np.asarray(product["unit_keys"]).reshape(-1).tolist()
            }
    except (OSError, TypeError, ValueError):
        return set()


def _validate_chime_inventory_meta(
    inventory_path: Path,
    *,
    command: str,
) -> None:
    try:
        metadata = read_inventory_meta(inventory_path)
    except ValueError as exc:
        raise SystemExit(f"{command}: {exc}") from exc
    if metadata is None:
        return
    expected = {
        "telescope": "chime",
        "source": _ARCHIVE_SOURCE,
        "reader": _SURVEY_READER,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise SystemExit(
            f"{command}: incompatible inventory metadata: " + ", ".join(mismatches)
        )


def inspect_chime_inventory(
    *,
    source: str | None = None,
    input_dir: str | Path | None = None,
    inventory: str | Path | None = None,
    inventory_name: str | None = None,
    source_root: str | Path | None = None,
    source_glob: str = "*.h5",
    source_freq_id_regex: str | None = None,
    source_event_regex: str | None = None,
) -> list[object]:
    """Summarize an inventory or local directory without staging data."""
    source_name = _resolve_source(
        source=source,
        input_dir=input_dir,
        inventory=inventory,
        inventory_name=inventory_name,
    )
    src, options, inventory_path = _source_context(
        source=source_name,
        input_dir=input_dir,
        inventory=inventory,
        inventory_name=inventory_name,
        source_root=source_root,
        source_glob=source_glob,
        source_freq_id_regex=source_freq_id_regex,
        source_event_regex=source_event_regex,
    )
    if inventory_path is not None:
        _validate_chime_inventory_meta(
            inventory_path,
            command="chime-inventory",
        )

    instrument = load_instrument("chime")
    ctx = RunContext(instrument=instrument, options=options)
    units = list(src.enumerate(ctx))
    if not units:
        raise SystemExit("chime-inventory: no data found")

    by_freq_id = Counter(
        freq_id for unit in units if (freq_id := _freq_id_of(unit)) is not None
    )
    dates = [
        str((getattr(unit, "meta", None) or {}).get("obs_date"))[:10]
        for unit in units
        if (getattr(unit, "meta", None) or {}).get("obs_date")
    ]
    total_bytes = 0
    for unit in units:
        metadata = getattr(unit, "meta", None) or {}
        size = metadata.get("size_bytes")
        if not size and metadata.get("src_path"):
            try:
                size = os.path.getsize(str(metadata["src_path"]))
            except OSError:
                size = 0
        total_bytes += int(size or 0)

    print(f"Available via source {source_name!r} for telescope 'chime'")
    print(f"  files          : {len(units)}")
    if total_bytes:
        print(f"  total volume   : {_human_bytes(total_bytes)}")
    if dates:
        print(f"  date span      : {min(dates)} .. {max(dates)}")
    if by_freq_id:
        freq_ids = sorted(by_freq_id)
        print(
            f"  freq_ids       : {len(freq_ids)} present "
            f"({freq_ids[0]}..{freq_ids[-1]})"
        )
        print("  freq_id  files")
        for freq_id in freq_ids:
            print(f"  {freq_id:<7}  {by_freq_id[freq_id]}")
        sample = ",".join(map(str, freq_ids[:3]))
        hint = (
            f"--inventory-name {inventory_name}"
            if inventory_name
            else f"--inventory {inventory_path}"
            if inventory_path is not None
            else f"--input-dir {input_dir or source_root}"
        )
        print(
            "\nScan a selection with:\n"
            f"  pilot-proxy chime-scan {hint} --select {sample} "
            "--output-dir <run>"
        )
    return units


def run_chime_control_scan(
    *,
    output_dir: str | Path,
    select: object,
    source: str | None = None,
    input_dir: str | Path | None = None,
    inventory: str | Path | None = None,
    inventory_name: str | None = None,
    source_root: str | Path | None = None,
    source_glob: str = "*.h5",
    source_freq_id_regex: str | None = None,
    source_event_regex: str | None = None,
    max_files: int | None = None,
    max_frames_per_file: int | None = None,
    frame_batch_size: int = 1,
    checkpoint_every: int = pipeline.DEFAULT_CHECKPOINT_EVERY,
    tmp_dir: str | Path | None = None,
    quarantine: str | Path | None = None,
    no_quarantine: bool = False,
    allow_partial: bool = False,
    analyzer_options: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Run one resumable control product per selected CHIME channel."""
    from .control import ControlBandAnalyzer

    if (
        isinstance(frame_batch_size, bool)
        or not isinstance(frame_batch_size, Integral)
        or frame_batch_size < 1
    ):
        raise SystemExit(
            "chime-control-scan: --frame-batch-size must be a positive integer"
        )

    for label, value in (
        ("--max-files", max_files),
        ("--max-frames-per-file", max_frames_per_file),
        ("--checkpoint-every", checkpoint_every),
    ):
        if value is not None and int(value) < 1:
            raise SystemExit(f"chime-control-scan: {label} must be positive")

    source_name = _resolve_source(
        source=source,
        input_dir=input_dir,
        inventory=inventory,
        inventory_name=inventory_name,
    )
    options = dict(analyzer_options or {})
    options["frame_batch_size"] = int(frame_batch_size)
    if max_frames_per_file is not None:
        options["max_frames_per_file"] = int(max_frames_per_file)
    src, options, inventory_path = _source_context(
        source=source_name,
        input_dir=input_dir,
        inventory=inventory,
        inventory_name=inventory_name,
        source_root=source_root,
        source_glob=source_glob,
        source_freq_id_regex=source_freq_id_regex,
        source_event_regex=source_event_regex,
        extra_options=options,
    )
    if inventory_path is not None and not inventory_path.exists():
        raise SystemExit(f"chime-control-scan: inventory not found: {inventory_path}")
    if inventory_path is not None:
        _validate_chime_inventory_meta(
            inventory_path,
            command="chime-control-scan",
        )

    instrument = load_instrument("chime")
    ctx = RunContext(instrument=instrument, options=options)
    reader = ChimeBasebandReader()
    analyzer_template = ControlBandAnalyzer()
    ctx.reader = reader
    if not dry_run:
        source_preflight = (
            "fetch_preflight" if isinstance(src, CadcDatatrailSource) else "preflight"
        )
        _preflight(
            src,
            ctx,
            label="chime-control-scan source",
            method=source_preflight,
        )
        _preflight(reader, ctx, label="chime-control-scan reader")
        _preflight(analyzer_template, ctx, label="chime-control-scan analyzer")
    runs = analyzer_template.plan_runs(ctx, select)
    if not runs:
        raise SystemExit("chime-control-scan: selection resolved to no channels")

    output = Path(output_dir).expanduser()
    scratch = Path(tmp_dir).expanduser() if tmp_dir is not None else output / "_staging"
    quarantine_path = (
        None
        if no_quarantine
        else str(
            Path(quarantine).expanduser()
            if quarantine is not None
            else output / "quarantine.jsonl"
        )
    )
    products: list[Path] = []
    failures = 0
    incomplete = 0
    print(f"[chime-control-scan] source={source_name} ({len(runs)} product(s))")
    for index, selection in enumerate(runs, 1):
        ctx.selection = selection
        units = list(src.enumerate(ctx))
        stem = (
            "_".join(map(str, selection))
            if isinstance(selection, (list, tuple))
            else str(selection)
        )
        product_path = output / f"{stem}.npz"
        saved_keys = _saved_product_unit_keys(product_path)
        current_keys = {str(unit.key) for unit in units}
        stale_keys = sorted(saved_keys - current_keys)
        if stale_keys:
            sample = ", ".join(repr(key) for key in stale_keys[:3])
            remainder = len(stale_keys) - 3
            if remainder > 0:
                sample += f", and {remainder} more"
            raise SystemExit(
                f"chime-control-scan: saved product {product_path} contains "
                f"units outside the current source scope ({sample}); refusing "
                "to modify it. Restore the original inventory or input directory, "
                "or use a fresh output directory."
            )
        print(
            f"  [{index}/{len(runs)}] select={selection} "
            f"units={len(units)} -> {product_path}"
        )
        if not units:
            incomplete += 1
            continue
        if dry_run:
            for unit in units[:3]:
                print(f"      would process: {unit.name}")
            continue
        result = pipeline.run(
            source=src,
            reader=reader,
            analyzer=ControlBandAnalyzer(),
            units=units,
            out_path=str(product_path),
            tmp_dir=str(scratch),
            ctx=ctx,
            checkpoint_every=int(checkpoint_every),
            download_workers=1,
            max_staged_files=1,
            max_files=max_files,
            max_frames_per_file=max_frames_per_file,
            quarantine_path=quarantine_path,
            verbose=len(runs) == 1,
        )
        failures += int(result.n_failed)
        if result.n_done != result.n_total or result.n_quarantined:
            incomplete += 1
        if result.product_available:
            products.append(product_path)

    if dry_run:
        return []
    print(
        f"control scan complete: {len(products)}/{len(runs)} product(s), "
        f"{failures} file failure(s)"
    )
    if (failures or incomplete) and not allow_partial:
        raise SystemExit(
            "chime-control-scan: incomplete run; rerun to resume or pass "
            "--allow-partial to accept the recorded scope"
        )
    return products


__all__ = ["inspect_chime_inventory", "run_chime_control_scan", "survey_chime"]
