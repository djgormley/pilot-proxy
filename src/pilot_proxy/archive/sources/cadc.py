"""
Data source: CADC storage + the CHIME/FRB Datatrail archive.

A source for CANFAR archive work, in three halves:

  enumerate()  read an inventory (inventory.jsonl) and yield one Unit per file,
               filtered to the selected freq_ids and/or events. The inventory is a cheap,
               offline listing, so enumerate itself never touches the network.

  survey()     build that inventory.jsonl: walk the Datatrail scope(s), discover
               every event, and verify each file the reader's archive shape
               (Reader.survey_files) declares for it at CADC -- one HDF5 per
               freq_id for baseband; whatever a different product's reader
               declares for that product. Resumable + incremental.

  fetch()      stage one file with cadcget via a CADC StorageInventoryClient,
               authenticated by a proxy certificate, with bounded retries.

Prerequisites (checked by `doctor`):
  * a valid CADC proxy cert (CADC_CERT or ~/.ssl/cadcproxy.pem)
  * the `cadcdata` / `cadcutils` packages  (for enumerate/fetch/verify)
  * the `datatrail-cli` package installed  (for survey only -- `[survey]` extra)
"""
from __future__ import annotations

import datetime
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from pathlib import Path
from typing import Iterable, List, Optional

from ..datatrail_client import DATATRAIL, Datatrail, DatatrailContractError
from ..interfaces import (
    READY,
    DataSource,
    PluginInfo,
    RunContext,
    SurveyUnavailableError,
    Unit,
)
from ..names import validate_identifier
from ..recon import match_terms, recon
from ..selection import parse_freq_ids, parse_selection
from ..survey_state import (
    STALE_VIEWS_MARKER,
    VIEW_FLUSH_INTERVAL,
    SurveyRun,
    SurveyStore,
    atomic_write_json,
    build_configuration,
    ensure_manifest,
    load_attempts,
    with_survey_output_lock,
)
from . import cadc_transport as _cadc_transport
from .cadc_inventory import (
    annotate_row,
    candidate_file,
    join_uri,
    logical_unit_key,
    parse_row,
)


def _default_shape():
    """The reader whose file shape survey uses when the caller supplied none.

    Used when the caller supplies no reader (including direct src.survey()
    calls in tests): the chime-baseband shape is this source's default naming.
    Imported lazily so merely importing this module never pulls the reader
    stack.
    """
    from pilot_proxy.chime.baseband_reader import ChimeBasebandReader
    return ChimeBasebandReader()


def _default_cert() -> str:
    """The CADC proxy cert used for the direct fetch path.

    CADC_CERT overrides (honoured even if absent -- the user's stated intent, and
    the clean way to point a headless/batch job at a different cert); otherwise the
    standard ~/.ssl/cadcproxy.pem that `cadc-get-cert` writes. preflight() checks
    the resolved path actually exists, so a missing or unrenewed cert is flagged at
    doctor time rather than mid-fetch.
    """
    return os.environ.get("CADC_CERT") or os.path.expanduser("~/.ssl/cadcproxy.pem")


STORAGE_SERVICE_ENV = "PILOT_PROXY_STORAGE_SERVICE"


def storage_service_override() -> Optional[str]:
    """Storage Inventory service to fetch through, or None for the default.

    cadcdata routes every download through ivo://cadc.nrc.ca/global/raven, which
    locates a replica before the bytes move. When that locator is degraded the
    replicas themselves stay healthy, so pointing the client straight at one
    (ivo://cadc.nrc.ca/uvic/minoc, say) keeps a run moving. Replicas serve the
    same objects, and fetch() still checks every staged file against the size
    the inventory recorded, so this changes the route the bytes travel and
    nothing about which bytes arrive. The resolved value is recorded in
    scan_scope.json, so the route a run actually took stays in its provenance.
    """
    value = (os.environ.get(STORAGE_SERVICE_ENV) or "").strip()
    return value or None


# --------------------------------------------------------------------------
# survey: defaults, geometry, and the Datatrail CLI plumbing
# --------------------------------------------------------------------------
# Final fallback when neither --scope nor the telescope's YAML `scopes` resolve.
# Each station (chime + outriggers kko/gbo/hco) declares its own baseband
# scope(s) in instruments/*.yaml, so survey defaults to those; this constant only
# applies to a telescope that declares none. (Outrigger-LABELLED chime events are
# still dropped unless --include-outrigger.)
_DEFAULT_SCOPES = ("chime.event.baseband.raw", "chime.scheduled.baseband.raw")

_DEFAULT_MINIMUM_ARCHIVE_BYTES = 1

_MAX_ATTEMPTS = 3            # per-event verification retries across resumes
_MAX_SERVICE_WAIT = 3600     # s to ride out a service/cert outage before aborting
_DEFAULT_SURVEY_WORKERS = 12
_DATASET_PROGRESS_SEGMENTS = 10
_INITIAL_SERVICE_BACKOFF = 60
_MAX_SERVICE_BACKOFF = 600
_EVENT_PROGRESS_INTERVAL = 100
_RETRY_BACKOFF_MULTIPLIER = 2
_ERROR_SUMMARY_LENGTH = 200
_DEFAULT_ARCHIVE_RETRIES = 3
_INITIAL_ARCHIVE_BACKOFF = 4.0

# Accept an all-absent ("empty") event on its FIRST sighting once the
# observation is at least this old. By then replication to CADC is long
# settled, so a definitive NotFound on every probe means the bytes aged off
# (or never landed) and no number of re-checks recovers them. Younger or
# undatable events keep the _MAX_ATTEMPTS cross-resume allowance, which
# exists for exactly one transient: archive-side replication still in
# flight for a recently registered event. Override per run: --empty-age-days.
_EMPTY_TERMINAL_AGE_DAYS = 30

_DATE_RE = re.compile(r"/raw/(\d{4})/(\d{2})/(\d{2})/")
_OUTRIGGER_RE = re.compile(r"outrigger", re.IGNORECASE)

_ENUM_CACHE_SCHEMA = 1


def _monotonic() -> float:
    """Patchable monotonic clock for the survey outage budget."""
    return time.monotonic()


def _sleep(seconds: float) -> None:
    """Patchable sleep boundary for survey outage backoff."""
    time.sleep(seconds)


def _obs_age_days(obs_date) -> Optional[int]:
    """Whole days since `obs_date` ('YYYY-MM-DD', as _DATE_RE extracts it from
    the Common Path), or None when the date is absent/unparseable ('unknown').
    None deliberately fails OPEN into the retry path: an event we cannot date
    is never written off on a single sighting."""
    try:
        d = datetime.date.fromisoformat(str(obs_date))
    except ValueError:
        return None
    return (datetime.datetime.now(datetime.timezone.utc).date() - d).days


def _resolve_freq_ids(spec, n_channels: Optional[int]) -> List[int]:
    """--freq-ids -> a sorted list of freq_id ints. Accepts 'all' (every freq_id in
    the instrument's band, i.e. range(n_channels)), a list, or the same string forms
    as --select."""
    all_ids = (list(range(n_channels)) if n_channels is not None else [])
    if spec is None:
        return all_ids
    if isinstance(spec, str) and spec.strip().lower() == "all":
        return all_ids
    freq_ids = parse_freq_ids(spec, n_channels=n_channels)
    return sorted(freq_ids) if freq_ids else all_ids


def _load_json_file(path: Path, default):
    """Read+parse JSON from `path`; return `default` if the file is missing,
    empty, or corrupt -- e.g. a checkpoint a previous run was killed mid-write.
    """
    try:
        if not path.exists():
            return default
        text = path.read_text().strip()
        return json.loads(text) if text else default
    except (json.JSONDecodeError, ValueError, OSError):
        return default


def _enumerate_events(scopes, include_outrigger, cache_path: Path, re_enumerate):
    """{(scope, event): [labels...]} across all larger-datasets, cached to disk.

    Phase 1: cheap, network-only listing. Cached so a re-run skips straight to
    verification unless --re-enumerate is given.
    """
    if cache_path.exists() and not re_enumerate:
        raw = _load_json_file(cache_path, None)
        if isinstance(raw, dict):
            if raw.get("schema") == _ENUM_CACHE_SCHEMA:
                cached_scopes = sorted(map(str, raw.get("scopes", ())))
                event_map = raw.get("events")
            else:
                event_map = None
                cached_scopes = []
            valid = (
                isinstance(event_map, dict)
                and all(isinstance(key, str) and "|" in key
                        and isinstance(labels, list)
                        and all(isinstance(label, str) for label in labels)
                        for key, labels in event_map.items())
            )
            if valid:
                if cached_scopes == sorted(map(str, scopes)):
                    return {tuple(key.split("|", 1)): list(labels)
                            for key, labels in event_map.items()}
                # The dir was enumerated for different scope(s). Serving the stale
                # list would silently survey the wrong scope, so stop loudly.
                raise SystemExit(
                    f"inventory directory {cache_path.parent} was already "
                    f"enumerated for scope(s) {cached_scopes}, but this run "
                    f"requests {sorted(scopes)}. An inventory is tied to its "
                    "scope(s); use a fresh --name/output directory.")
        # corrupt/empty cache -> fall through and re-enumerate

    membership: dict = defaultdict(set)
    for scope in scopes:
        datasets, ok = DATATRAIL.list_datasets_checked(scope)
        if not ok:
            raise SurveyUnavailableError(
                f"datatrail could not list datasets under scope {scope!r}; "
                f"the existing event cache at {cache_path} was left untouched. "
                "Re-run when the service responds.")
        # One `datatrail ls` per dataset -- a slow walk -- so show progress, but
        # not all N names (use --scopes-only recon to inspect dataset structure).
        print(f"{scope}: walking {len(datasets)} larger-dataset(s)", flush=True)
        step = max(1, len(datasets) // _DATASET_PROGRESS_SEGMENTS)
        for i, ds in enumerate(datasets, 1):
            discovered, child_ok = DATATRAIL.events_in_dataset_checked(scope, ds)
            if not child_ok:
                raise SurveyUnavailableError(
                    f"datatrail could not list children of {scope!r}/{ds!r}; "
                    f"the existing event cache at {cache_path} was left "
                    "untouched. Re-run when the service responds.")
            for ev in discovered:
                membership[(scope, ev)].add(ds)
            if i % step == 0 or i == len(datasets):
                print(f"  ...{i}/{len(datasets)} datasets", flush=True)

    out_map = {f"{s}|{e}": sorted(lbls) for (s, e), lbls in membership.items()}
    atomic_write_json(cache_path, {
        "schema": _ENUM_CACHE_SCHEMA,
        "scopes": sorted(map(str, scopes)),
        "events": out_map,
    })
    n_out = sum(1 for lbls in membership.values()
                if any(_OUTRIGGER_RE.search(x) for x in lbls))
    print(f"\nenumerated {len(membership)} unique events; {n_out} carry an "
          f"outrigger label ({'KEPT' if include_outrigger else 'BLOCKED'})", flush=True)
    return {k: sorted(v) for k, v in membership.items()}


def _commit_decision(n_errored: int, attempts: int, n_records: int,
                     max_attempts: int = _MAX_ATTEMPTS,
                     empty_max_attempts: Optional[int] = None) -> tuple:
    """Return (write_records, mark_done, incomplete) for one event.

      clean, rows resolved           -> write rows, mark done
      partial, attempts left         -> write nothing, don't mark (clean retry)
      partial, out of attempts       -> accept what verified, mark done, flag
      empty (cp resolved, 0 rows,    -> write nothing, retry across resumes, then
        0 hard errors), retries left     accept-as-empty + mark done -- so a 0-row
      empty, out of attempts            event can never silently read as "clean"

    `n_records` is how many files verified PRESENT. Zero PRESENT *and* zero
    errored means the event resolved a common path but produced no usable file:
    distinct from a fully-resolved event, and a case that must never be written
    out as a clean 0-row 'done'.
    `empty_max_attempts` caps the EMPTY case only (None -> `max_attempts`).
    survey passes 1 when the observation is old enough that absence cannot be
    replication lag (_EMPTY_TERMINAL_AGE_DAYS / --empty-age-days), so a
    definitively-absent old event is accepted on first sighting instead of
    burning cross-resume probes that were never observed to recover. The
    errored/partial path is unaffected: a hard probe error is not an answer,
    however old the observation.
    """
    if n_errored == 0:
        if n_records == 0:                        # resolved, but nothing present
            cap = (max_attempts if empty_max_attempts is None
                   else empty_max_attempts)
            if attempts + 1 >= cap:
                return False, True, False         # accept-as-empty: mark done
            return False, False, False            # retry on the next resume
        return True, True, False                  # clean, rows to write
    if attempts + 1 >= max_attempts:
        return True, True, True
    return False, False, False


class CadcDatatrailSource(DataSource):
    info = PluginInfo(
        name="cadc-datatrail",
        kind="source",
        summary="CHIME/FRB baseband on CADC; survey to build, enumerate offline.",
        status=READY,
        instruments=("chime", "kko", "gbo", "hco"),
        requires=("CADC proxy cert", "cadcdata", "cadcutils",
                  "datatrail CLI (survey only)"),
        needs_archive_config=True,
        notes="survey() walks the Datatrail scope(s) to build inventory.jsonl; "
              "enumerate() then reads it offline; fetch() uses cadcget with "
              f"{_cadc_transport.CADC_CONNECT_TIMEOUT_SECONDS:g}s connect and "
              f"{_cadc_transport.CADC_READ_TIMEOUT_SECONDS:g}s "
              "read-inactivity timeouts.",
    )

    def __init__(self) -> None:
        # One client per thread: fetch()/cadcinfo() may run concurrently (download
        # workers, or the survey verify pool), and a shared CADC client is not
        # guaranteed safe for concurrent calls.
        self._local = threading.local()
        # Resolve optional exception families before a timed network operation;
        # imports on the first failure must not consume its retry/timeout budget.
        self._expected_errors = _cadc_transport.expected_errors()
        self._last_survey_completeness: Optional[dict[str, int]] = None

    def survey_completeness_issues(
            self, out_dir: str) -> Optional[dict[str, int]]:
        """Omissions in the persisted state produced by the latest survey.

        Accepted-empty and definitive no-data events are explicit archive
        dispositions, not unresolved omissions. Strict mode fails on partial
        terminal events, contract refusals, and events still pending a later
        resumable pass (including a ``--max-events`` stop).
        """
        del out_dir
        if self._last_survey_completeness is None:
            return None
        return dict(self._last_survey_completeness)

    def _get_client(self):
        cl = getattr(self._local, "client", None)
        if cl is None:
            cl = self._make_client()
            self._local.socket_timeout_fallback = not (
                _cadc_transport.configure_request_timeout(cl))
            self._local.client = cl
        return cl

    def _make_client(self, cert=None):
        from cadcdata import StorageInventoryClient
        from cadcutils import net
        cert = cert or _default_cert()
        subj = (net.Subject(certificate=cert)
                if cert and os.path.exists(cert) else net.Subject())
        service = storage_service_override()
        if service:
            return StorageInventoryClient(subj, resource_id=service)
        return StorageInventoryClient(subj)

    # -- enumerate -----------------------------------------------------------
    def _inventory_path(self, ctx: RunContext) -> str:
        o = ctx.options or {}
        p = o.get("inventory")
        if p:
            return p
        raw_name = o.get("name") or ctx.instrument.name
        try:
            name = validate_identifier(raw_name, label="inventory name")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        base = o.get("root")
        if base:
            return os.path.join(str(base), "data", name, "inventory.jsonl")
        from ..invpaths import resolve_inventory
        return str(resolve_inventory(name))

    def enumerate(self, ctx: RunContext) -> Iterable[Unit]:
        path = self._inventory_path(ctx)
        if not os.path.exists(path):
            raise SystemExit(
                f"inventory not found: {path}\n"
                "Build one with `pilot-proxy chime-survey` "
                "(or pass --inventory <path>).")
        if (Path(path).parent / STALE_VIEWS_MARKER).exists():
            raise SystemExit(
                f"inventory views in {Path(path).parent} may be stale after an active "
                "or interrupted survey; rerun the same survey command to refresh them")
        n_channels = getattr(ctx.instrument, "n_channels", None)
        sel = parse_selection(ctx.selection, n_channels=n_channels)
        seen, units = set(), []
        with open(path) as fh:
            for line_number, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                r = parse_row(line, path, line_number)
                ch = r.get("freq_id")
                if ch is not None and n_channels is not None:
                    # The row parser proves this is an integer; the shared
                    # grammar supplies the instrument-specific upper bound.
                    parse_freq_ids([ch], n_channels=n_channels)
                if not sel.wants_freq_id(ch):
                    continue
                if not sel.wants_event(r.get("event")):
                    continue
                # Rows are self-describing since the shape moved onto the
                # reader: survey writes each file's `name`, and enumerate just
                # joins it to the common path -- no naming re-derivation, so
                # survey and read cannot drift: current inventory rows always
                # carry the exact verified relative name.
                name = r["name"]
                uri = join_uri(r["common_path"], name)
                logical_key = logical_unit_key(r["scope"], r["event"], name)
                if logical_key in seen:
                    continue
                seen.add(logical_key)
                # meta: the row, minus the URI ingredients, plus the stable
                # quarantine identity. Shape-specific columns (freq_id for
                # baseband; whatever a calibration shape wrote) ride through
                # untouched -- meta is opaque to the engine and is exactly how
                # an analyzer keys a companion lookup (e.g. gains by event).
                meta = {k: v for k, v in r.items()
                        if k not in ("common_path", "name")}
                if ch is not None:
                    meta["freq_id"] = int(ch)
                meta["size_bytes"] = int(r.get("size_bytes", 0))
                meta["cadc_uri"] = uri
                meta["quarantine_key"] = logical_key
                units.append(Unit(key=logical_key, name=name, meta=meta))
        return units

    # -- fetch ---------------------------------------------------------------
    def fetch(self, unit: Unit, dest: str,
              retries: int = _DEFAULT_ARCHIVE_RETRIES,
              base: float = _INITIAL_ARCHIVE_BACKOFF):
        """Fetch one object with finite connect/read-inactivity waits."""
        client = self._get_client()
        delay, last = base, None
        expected = (unit.meta or {}).get("size_bytes")
        expected = (int(expected)
                    if (isinstance(expected, Integral)
                        and not isinstance(expected, bool) and expected > 0)
                    else None)

        def discard_partial():
            try:
                os.unlink(dest)
            except FileNotFoundError:
                pass

        for k in range(retries + 1):
            try:
                uri = unit.meta.get("cadc_uri", unit.key)
                if getattr(self._local, "socket_timeout_fallback", True):
                    # The socket default is process-wide. This compatibility
                    # path is serialized and always restored; current CADC
                    # clients use the per-session path above and remain fully
                    # concurrent.
                    with _cadc_transport.socket_timeout(
                            _cadc_transport.CADC_READ_TIMEOUT_SECONDS):
                        client.cadcget(uri, dest=dest)
                else:
                    client.cadcget(uri, dest=dest)
                actual = os.path.getsize(dest) if os.path.exists(dest) else 0
                if expected is not None and actual != expected:
                    last = (f"size mismatch: expected {expected} bytes, "
                            f"received {actual}")
                elif actual > 0:
                    return True, ""
                else:
                    last = "empty file"
            except self._expected_errors as exc:
                if "NotFound" in type(exc).__name__:
                    discard_partial()
                    return False, "NotFound"
                last = _cadc_transport.describe_exception(exc)
            if k < retries:
                discard_partial()
                time.sleep(delay)
                delay *= _RETRY_BACKOFF_MULTIPLIER
        discard_partial()
        return False, str(last)[:_ERROR_SUMMARY_LENGTH]

    # -- survey ---------------------------------------
    # (The per-product file shape -- which files one event contributes, and
    # their names -- lives on the reader (Reader.survey_files) so survey and
    # read share one naming definition and cannot drift; survey() below
    # consults ctx.reader, falling back to the chime-baseband reader's shape.)
    def _cadc_size(self, uri, retries: int = _DEFAULT_ARCHIVE_RETRIES,
                   base: float = _INITIAL_ARCHIVE_BACKOFF, deadline=None):
        delay, last = base, None
        for k in range(retries + 1):
            try:
                client = self._get_client()
                with _cadc_transport.request_deadline(deadline):
                    if getattr(self._local, "socket_timeout_fallback", True):
                        with _cadc_transport.socket_timeout(
                                _cadc_transport.CADC_READ_TIMEOUT_SECONDS,
                                deadline=deadline):
                            info = client.cadcinfo(uri)
                    else:
                        info = client.cadcinfo(uri)
                return info.size, None
            except self._expected_errors as exc:
                if "NotFound" in type(exc).__name__:
                    return None, None                       # definitive: absent
                last = exc
                if k < retries:
                    sleep_for = delay
                    if deadline is not None:
                        remaining = float(deadline) - time.monotonic()
                        if remaining <= 0:
                            return None, last
                        sleep_for = min(sleep_for, remaining)
                    time.sleep(sleep_for)
                    delay *= _RETRY_BACKOFF_MULTIPLIER
        return None, last

    @with_survey_output_lock
    def survey(self, ctx: RunContext, out_dir: str) -> str:
        """Build inventory.jsonl for the selected scope(s) + freq_ids.

        Two phases, resumable and incremental: enumerate the unique events under
        each scope (cached), then for every not-yet-done event resolve its Common
        Path and cadcinfo each requested freq_id, writing verified rows atomically
        per event. Re-running tops up remaining cached events without
        re-surveying completed ones. Use `--re-enumerate` to discover newly
        registered events; use a fresh inventory to re-probe an already
        completed event.
        """
        o = ctx.options or {}
        scope_opt = o.get("scope")
        named = (tuple(s.strip() for s in
                       (scope_opt.split(",") if isinstance(scope_opt, str) else scope_opt)
                       if str(s).strip())
                 if scope_opt else None)
        if o.get("scopes_only"):                 # recon: recursive `datatrail ls`
            label = o.get("name")
            return recon(named, match_terms(o.get("match")), out_dir,
                         expand=bool(o.get("expand", False)),
                         telescope=getattr(ctx.instrument, "name", None),
                         map_name=(f"scopes-{label}.jsonl" if label
                                   else "scopes.jsonl"))
        inst_scopes = tuple(getattr(ctx.instrument, "scopes", ()) or ())
        scopes = named or inst_scopes or _DEFAULT_SCOPES
        n_ch = (ctx.instrument.n_channels
                if ctx.instrument is not None else None)
        freq_ids = _resolve_freq_ids(o.get("freq_ids", ctx.selection), n_ch)
        include_outrigger = bool(o.get("include_outrigger", False))
        workers = max(1, int(o.get("workers", _DEFAULT_SURVEY_WORKERS)
                             or _DEFAULT_SURVEY_WORKERS))
        max_events = o.get("max_events")
        re_enumerate = bool(o.get("re_enumerate", False))
        eag = o.get("empty_age_days")
        empty_age_days = (_EMPTY_TERMINAL_AGE_DAYS if eag is None else int(eag))
        if empty_age_days < 0:
            raise SystemExit("--empty-age-days must be zero or greater")

        # The reader owns the archive file shape (which files one event
        # contributes, and their names -- Reader.survey_files). The CLI resolves
        # the run's reader onto ctx.reader; a caller that supplied none falls
        # back to the chime-baseband shape.
        shape = ctx.reader if ctx.reader is not None else _default_shape()
        reader_name = getattr(getattr(shape, "info", None), "name",
                              type(shape).__name__)
        survey_schema = getattr(shape, "survey_schema", None)
        if (isinstance(survey_schema, bool)
                or not isinstance(survey_schema, int)
                or survey_schema < 1):
            raise SystemExit(
                f"reader {reader_name!r} must declare a positive integer "
                "survey_schema for archive surveys; bump it whenever "
                "survey_files(), annotate_row(), or inventory row semantics "
                "change")
        minimum_spec = getattr(
            shape, "minimum_archive_bytes", _DEFAULT_MINIMUM_ARCHIVE_BYTES)
        minimum_bytes = int(minimum_spec(ctx) if callable(minimum_spec)
                            else minimum_spec)
        if minimum_bytes < 1:
            raise SystemExit(
                f"reader {reader_name!r} minimum_archive_bytes must be >= 1")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ensure_manifest(
            out,
            build_configuration(ctx, scopes, freq_ids, shape,
                                include_outrigger, empty_age_days,
                                minimum_bytes, survey_schema),
        )
        with SurveyRun(out) as execution:
            inv_path = out / "inventory.jsonl"
            fid_note = (f" freq_ids={len(freq_ids)} ({freq_ids[0]}..{freq_ids[-1]})"
                        if freq_ids else "")
            print(f"[survey] scopes={list(scopes)}"
                  f" shape={reader_name} minimum_bytes={minimum_bytes}"
                  f"{fid_note} -> {inv_path}", flush=True)

            # ---- phase 1: enumerate the unique events (cached) ----
            membership = _enumerate_events(scopes, include_outrigger,
                                           out / "enum_cache.json", re_enumerate)
            events = sorted(ev for ev, lbls in membership.items()
                            if include_outrigger
                            or not any(_OUTRIGGER_RE.search(x) for x in lbls))
            print(f"to survey: {len(events)} events", flush=True)

            # ---- phase 2: verify each event's freq_ids (resumable) ----
            attempts_path = out / "attempts.json"
            attempts = load_attempts(attempts_path)
            pending_path = out / "pending_events.json"
            pending = _load_json_file(pending_path, {})
            if not isinstance(pending, dict):
                raise SystemExit(f"invalid pending survey state: {pending_path}")
            store = SurveyStore(out / "survey_state.sqlite3")
            surveyed = store.completed_keys()
            # A crash can land after the event transaction but before its attempts
            # checkpoint. The committed event wins; discard that harmless stale count.
            for key in surveyed:
                attempts.pop(key, None)
                pending.pop(key, None)
            atomic_write_json(attempts_path, attempts)
            atomic_write_json(pending_path, pending)
            # Recover JSONL/text views if a previous process stopped after committing
            # SQLite but before its periodic/final render.
            store.render_views(out)
            print(f"resume: {len(surveyed)} events already done", flush=True)

            # accept-as-empty ledger: one JSON object per written-off event, so the
            # write-off is auditable (when, how many sightings, why) and greppable.
            no_files_path = out / "no_files_events.jsonl"
            pool = ThreadPoolExecutor(max_workers=workers)
            n_new = 0
            n_committed = 0
            # run-level accounting so a 0-row inventory can never read as success
            n_rows = n_no_data = n_incomplete = n_empty_retry = n_empty_accepted = 0
            n_refused = 0

            def mark_done(key, scope, ev, status, records=(), *, incomplete=None,
                          no_files=None):
                nonlocal n_committed
                store.commit(key, scope, ev, status, records,
                             incomplete=incomplete, no_files=no_files)
                surveyed.add(key)
                attempts.pop(key, None)
                pending.pop(key, None)
                atomic_write_json(attempts_path, attempts)
                atomic_write_json(pending_path, pending)
                n_committed += 1
                if n_committed % VIEW_FLUSH_INTERVAL == 0:
                    store.render_views(out)

            def bump(key):
                attempts[key] = attempts.get(key, 0) + 1
                atomic_write_json(attempts_path, attempts)

            def verify(scope, ev, deadline):
                try:
                    cp, ps_ok = DATATRAIL.common_path(
                        scope, ev, deadline=deadline)
                except DatatrailContractError as exc:
                    # Deterministic: the service answered, the payload is not one
                    # this adapter can act on. Retrying cannot change it, and it
                    # must never masquerade as an outage (which would stall the
                    # whole survey on one event and then abort with a misleading
                    # certificate message).
                    return "refused", [], [], 0, "unknown", None, str(exc), {}
                if not ps_ok:
                    return "service_down", [], [], 0, "unknown", None, None, {}
                if not cp:
                    return "no_data", [], [], 0, "unknown", None, None, {}
                obs_date = (lambda m: f"{m[1]}-{m[2]}-{m[3]}" if m else "unknown")(
                    _DATE_RE.search(cp))
                labels = membership[(scope, ev)]
                # The candidate files one event contributes -- (name, fields) pairs
                # from the reader's shape. Baseband yields one per freq_id; a
                # per-event product may yield a single file with its own fields.
                cand = [candidate_file(item, reader_name)
                        for item in shape.survey_files(ev, cp, freq_ids, ctx)]
                candidate_names = [name for name, _fields in cand]
                if len(candidate_names) != len(set(candidate_names)):
                    raise SystemExit(
                        f"reader {reader_name!r} yielded duplicate archive names "
                        f"for event {ev!r}")

                def probe(item):
                    name, fields = item
                    size, err = self._cadc_size(
                        join_uri(cp, name), deadline=deadline)
                    return name, fields, size, err

                records, errored, sub_floor = [], [], []
                for name, fields, size, err in pool.map(probe, cand):
                    if err is not None:
                        errored.append(name)
                    elif size is not None and size >= minimum_bytes:
                        # Self-describing row: `name` is what enumerate/fetch will
                        # stage (joined to common_path), and the shape's per-file
                        # fields land verbatim as columns.
                        rec = {
                            "scope": str(scope), "event": str(ev), "name": name,
                            "size_bytes": int(size),
                            "common_path": str(cp), "obs_date": obs_date,
                            "datasets": list(labels),
                            "collection_restored": bool(getattr(cp, "collection_restored", False)),
                        }
                        rec.update(fields)
                        annotate_row(shape, rec, ctx.instrument)
                        records.append(rec)
                    elif size is not None:
                        sub_floor.append({"name": name, "size_bytes": int(size)})
                details = {
                    "collection_restored": bool(getattr(cp, "collection_restored", False)),
                    "minimum_archive_bytes": minimum_bytes,
                    "sub_floor_files": sub_floor,
                    "n_sub_floor": len(sub_floor),
                    "n_absent": len(cand) - len(records) - len(errored) - len(sub_floor),
                }
                if errored and len(errored) == len(cand):       # all errored -> outage
                    return "service_down", records, errored, len(cand), obs_date, cp, None, details
                return "progress", records, errored, len(cand), obs_date, cp, None, details

            try:
                for i, (scope, ev) in enumerate(events, 1):
                    key = f"{scope}|{ev}"
                    if key in surveyed:
                        continue
                    if max_events is not None and n_new >= int(max_events):
                        print(f"reached --max-events={max_events}; stopping "
                              f"(resumable).", flush=True)
                        break

                    # ride out a transient outage on the SAME event; only a sustained
                    # one aborts (the signature of an expired cert, which won't heal).
                    backoff = _INITIAL_SERVICE_BACKOFF
                    outage_started = _monotonic()
                    outage_deadline = outage_started + max(
                        0.0, float(_MAX_SERVICE_WAIT))
                    attempted = False

                    def unavailable(elapsed):
                        return SurveyUnavailableError(
                            "Datatrail/CADC remained unreachable for "
                            f"{elapsed:g}s. "
                            f"Partial survey state was preserved in {out}. Renew "
                            "the certificate with `cadc-get-cert -u <user>` (or "
                            "wait for the service to recover), then rerun the same "
                            "survey command."
                        )

                    while True:
                        if attempted:
                            now = _monotonic()
                            if now >= outage_deadline:
                                raise unavailable(max(0.0, now - outage_started))
                        (status, records, errored, n_cand,
                         obs_date, cp, refusal, details) = verify(scope, ev, outage_deadline)
                        attempted = True
                        if status != "service_down":
                            break
                        now = _monotonic()
                        elapsed = max(0.0, now - outage_started)
                        remaining = max(0.0, outage_deadline - now)
                        if remaining <= 0:
                            raise unavailable(elapsed)
                        sleep_for = min(float(backoff), remaining)
                        print(f"[{i}/{len(events)}] {ev}: service unreachable -- "
                              f"waiting {sleep_for:g}s "
                              f"(elapsed {elapsed:g}s)", flush=True)
                        _sleep(sleep_for)
                        backoff = min(
                            backoff * _RETRY_BACKOFF_MULTIPLIER,
                            _MAX_SERVICE_BACKOFF,
                        )

                    n_new += 1
                    if status == "refused":
                        # Committed as done (resume skips it) with the reason in
                        # the accept-as-empty ledger, so the write-off is
                        # auditable and greppable rather than a silent gap. If the
                        # named URI turns out to be a legitimate new collection,
                        # widen _MINOC_COLLECTIONS in datatrail_client.py and re-open
                        # these rows (DELETE FROM events WHERE status='refused'
                        # in survey_state.sqlite3) before re-running.
                        n_refused += 1
                        ledger = {
                            "ts": datetime.datetime.now(datetime.timezone.utc)
                                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "scope": scope, "event": ev,
                            "n_expected": 0,
                            "attempts": attempts.get(key, 0) + 1,
                            "obs_date": "unknown", "age_days": None,
                            "common_path": None,
                            "reason": "datatrail-contract-refusal",
                            "detail": refusal,
                        }
                        print(f"[{i}/{len(events)}] {ev}: datatrail contract "
                              f"refusal -- recorded and skipped: {refusal}",
                              flush=True)
                        mark_done(key, scope, ev, "refused", no_files=ledger)
                        continue
                    if status == "no_data":
                        n_no_data += 1
                        mark_done(key, scope, ev, "no-data")
                        continue
                    if n_cand == 0:
                        mark_done(key, scope, ev, "empty-selection", no_files={
                            "scope": scope, "event": ev, "n_expected": 0,
                            "common_path": str(cp), "reason": "no-selected-candidates",
                            "attempts": attempts.get(key, 0) + 1,
                            **details,
                        })
                        print(f"[{i}/{len(events)}] {ev}: reader selection produced "
                              "no candidate files; no archive absence inferred", flush=True)
                        continue

                    # `empty` = a common path resolved but every requested freq_id
                    # came back absent/sub-floor: 0 rows AND 0 hard errors. That is
                    # NOT a clean, fully-resolved event and must never be written
                    # out as a silent 0-row "done". Absence here is
                    # already definitive per probe (_cadc_size reports NotFound as
                    # an answer, not an error; outages take the service_down
                    # circuit; hard errors take the INCOMPLETE path), so the only
                    # transient it could mask is archive-side replication lag --
                    # which can only affect a RECENT observation. An empty event
                    # whose obs_date is at least `empty_age_days` old is therefore
                    # accepted on FIRST sighting; a younger or undatable one stays
                    # un-done (retried across resumes) until _MAX_ATTEMPTS. Either
                    # way the acceptance is recorded in no_files_events.jsonl, so a
                    # 0-row event can neither vanish silently nor re-probe forever.
                    empty = not records and not errored
                    sightings = attempts.get(key, 0) + 1    # incl. this run's probe
                    if empty and details["collection_restored"] and not details["n_sub_floor"]:
                        # A guessed prefix with no independently found object is
                        # not evidence of archive absence, regardless of age or
                        # repeat count. Keep it resumable and record why.
                        pending[key] = {
                            "scope": scope, "event": ev, "common_path": str(cp),
                            "reason": "unverified-restored-collection",
                            "attempts": sightings, "n_expected": n_cand, **details,
                        }
                        atomic_write_json(pending_path, pending)
                        bump(key)
                        n_empty_retry += 1
                        print(f"[{i}/{len(events)}] {ev}: restored collection "
                              "has no verified object; remains pending", flush=True)
                        continue
                    age_days = _obs_age_days(obs_date) if empty else None
                    aged_out = (empty and age_days is not None
                                and age_days >= empty_age_days)
                    write_recs, done, incomplete = _commit_decision(
                        len(errored), attempts.get(key, 0), len(records),
                        empty_max_attempts=(1 if aged_out else None))
                    if write_recs:
                        n_rows += len(records)
                    if incomplete:
                        n_incomplete += 1
                    if done:
                        no_files_record = None
                        if empty:                  # terminal 0-file event: ledger it
                            no_files_record = {
                                "ts": datetime.datetime.now(datetime.timezone.utc)
                                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "scope": scope, "event": ev,
                                "n_expected": n_cand,
                                "attempts": sightings,
                                "obs_date": obs_date,
                                "age_days": age_days,
                                "common_path": cp,
                                "reason": "below-reader-floor" if details["n_sub_floor"]
                                          else "aged-out" if aged_out else "max-attempts",
                                **details,
                            }
                            n_empty_accepted += 1
                        mark_done(
                            key, scope, ev,
                            "empty" if empty else
                            "incomplete" if incomplete else "complete",
                            records if write_recs else (),
                            incomplete=errored if incomplete else None,
                            no_files=no_files_record,
                        )
                    else:
                        bump(key)
                        if empty:
                            n_empty_retry += 1

                    # surface the empty case too -- otherwise it prints nothing and
                    # the run looks like it did nothing at all (0 events processed
                    # visibly, 0 rows on disk).
                    if (records or errored or empty
                            or i % _EVENT_PROGRESS_INTERVAL == 0):
                        if empty and done:
                            why = (f"obs {obs_date}, {age_days}d old" if aged_out
                                   else f"attempt {sightings}/{_MAX_ATTEMPTS}")
                            cause = ("objects below reader byte floor" if details["n_sub_floor"]
                                     else "not in CADC storage")
                            tag = (f" -- {cause}; accepting as empty "
                                   f"({why})")
                        elif empty:
                            cause = ("objects below reader byte floor" if details["n_sub_floor"]
                                     else "not in CADC storage")
                            tag = (f" -- {cause}; re-checking in case "
                                   f"transient ({sightings}/{_MAX_ATTEMPTS})")
                        else:
                            tag = (f" INCOMPLETE({len(errored)})" if incomplete
                                   else f" ({len(errored)} unresolved, retry)"
                                   if errored else "")
                        print(f"[{i}/{len(events)}] {ev}: "
                              f"{len(records)}/{n_cand} files{tag}", flush=True)
            finally:
                # Cancel work that has not started and join active probes before
                # releasing the output lock or returning an error. Supported CADC
                # calls carry finite per-request timeouts (and the outage deadline),
                # so shutdown is bounded by the remaining in-flight requests.
                pool.shutdown(wait=True, cancel_futures=True)
                atomic_write_json(attempts_path, attempts)
                try:
                    store.render_views(out)
                    execution.views_current()
                finally:
                    try:
                        status_counts = store.status_counts()
                        current_event_keys = {
                            f"{scope}|{event}" for scope, event in events
                        }
                        self._last_survey_completeness = {
                            "incomplete": (status_counts.get("incomplete", 0)
                                           + status_counts.get("empty-selection", 0)),
                            "refused": status_counts.get("refused", 0),
                            "pending": len(current_event_keys - surveyed),
                        }
                        execution.record["completeness"] = dict(self._last_survey_completeness)
                    finally:
                        store.close()

            # Row-level accounting: the final word on the run, so an empty
            # inventory can never hide behind "survey wrote <path>" while 0 rows
            # landed.
            total_rows = (sum(1 for ln in open(inv_path) if ln.strip())
                          if inv_path.exists() else 0)
            print(f"\nsurvey: {n_new} events this run -- {n_rows} rows written, "
                  f"{n_no_data} no-data, {n_empty_accepted} accepted-empty, "
                  f"{n_empty_retry} resolved-but-empty (retry next run), "
                  f"{n_incomplete} incomplete, {n_refused} contract-refused",
                  flush=True)
            if n_empty_accepted or n_refused:
                print(f"accepted-empty ledger: {no_files_path}", flush=True)
            if n_refused:
                print("contract refusals name the offending replica URI in the "
                      "ledger; if it is a legitimate new collection, widen "
                      "_MINOC_COLLECTIONS in pilot_proxy.archive.datatrail_client "
                      "re-open refused rows before re-running.", flush=True)
            if total_rows == 0:
                print(
                    "[warn] inventory.jsonl is EMPTY (0 rows). Every surveyed event "
                    "resolved to zero retrievable files, so nothing was written -- "
                    "usually the environment, not the survey. Sanity-check one event: "
                    "`datatrail ps <scope> <event> -s` (is a 'Common Path:' line "
                    "printed?), then `cadcinfo --cert ~/.ssl/cadcproxy.pem <cadc-uri>` "
                    "for one freq_id (NotFound = the bytes aged off storage, or a size "
                    f"under this reader's {minimum_bytes}-byte floor; pass the cert or "
                    "the CLI runs anonymously and "
                    "reports a misleading 'Unauthorized'). The lowest event IDs are the "
                    "likeliest to have aged out of the archive, so a larger "
                    "--max-events often starts filling the inventory.", flush=True)
            print(f"survey wrote {inv_path}", flush=True)
            return str(inv_path)

    # -- doctor --------------------------------------------------------------
    def fetch_preflight(self, ctx: RunContext) -> tuple[bool, list[str], list[str]]:
        """Check only the dependencies needed to read an inventory and fetch."""
        del ctx
        problems: list = []
        cert = _default_cert()
        if not os.path.exists(cert):
            problems.append(
                f"no CADC proxy cert at {cert} "
                f"(run `cadc-get-cert -u <user>` or set CADC_CERT)")
        try:
            import cadcdata  # noqa: F401
            import cadcutils  # noqa: F401
        except Exception:
            problems.append("cadcdata/cadcutils not importable "
                            "(pip install -e \".[archive]\")")
        return (not problems), problems, []

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str], list[str]]:
        _ok, problems, notes = self.fetch_preflight(ctx)

        # -- survey-only prerequisites: datatrail ------------------------------
        # scan/fetch never call datatrail; these are survey-only and stay silent
        # when datatrail is absent (`requires` already lists it). The whole
        # datatrail surface lives behind pilot_proxy.archive.datatrail_client.
        if Datatrail.installed():
            # (a) the machine-readable CLI contract survey drives (`datatrail
            # ls/ps --json`, datatrail-cli >= 0.11). A pre-0.11 install has no
            # --json flag; the live call would misread as a service outage and
            # stall survey ~an hour before a misleading 'expired cert' abort.
            # Report the real cause here, before the run, instead.
            ok, detail = Datatrail.api_available()
            if not ok:
                problems.append(
                    "datatrail is installed but the machine-readable CLI "
                    f"contract survey drives is unavailable: {detail}. survey "
                    "uses `datatrail ls/ps --json` (datatrail-cli>=0.11); "
                    "upgrade datatrail-cli (scan/fetch are unaffected).")

            # (b) validate the scope(s) survey will walk against datatrail's live
            # namespace, so a stale/renamed scope fails here instead of silently
            # walking nothing into an empty inventory. datatrail owns which scopes
            # EXIST; the instrument YAML owns which to walk.
            o = ctx.options or {}
            scope_opt = o.get("scope")
            effective = (tuple(s.strip() for s in
                              (scope_opt.split(",") if isinstance(scope_opt, str)
                               else scope_opt) if str(s).strip())
                         if scope_opt
                         else tuple(getattr(ctx.instrument, "scopes", ()) or ()))
            if effective:
                try:
                    known = set(DATATRAIL.list_scopes())
                except Exception:
                    known = set()    # datatrail unreachable -> can't validate
                if known:            # empty => transient/auth, not 'all invalid'
                    missing = [s for s in effective if s not in known]
                    if missing:
                        problems.append(
                            f"scope(s) not found in datatrail: {missing} -- survey "
                            "would walk nothing for them. Fix the instrument YAML "
                            "`scopes:` or --scope (`datatrail ls` lists valid "
                            "scopes).")
                else:
                    # couldn't list scopes (datatrail down / not installed). Don't
                    # claim the scopes are invalid -- surface a visible, non-fatal
                    # 'skipped' so doctor's READY isn't silently masking a check
                    # that never actually ran.
                    notes.append(
                        "datatrail scope(s) not validated: could not reach "
                        "datatrail to list scopes -- survey will still attempt "
                        f"{list(effective)} (re-run doctor once datatrail responds)")

        return (not problems), problems, notes
