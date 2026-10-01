"""Durable state for a resumable CADC survey.

The network/archive source decides *what* an event means and verifies its files.
This module owns the orthogonal persistence contract:

* a configuration fingerprint prevents incompatible runs sharing a directory;
* atomic JSON replacement protects small checkpoints and the event cache; and
* SQLite commits an event's inventory rows and done marker in one transaction.

``inventory.jsonl`` and the historical text ledgers are deterministic views of
the database. They can therefore be regenerated after a process stops between a
database commit and a view refresh, without duplicating rows.
"""
from __future__ import annotations

import datetime
import errno
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Mapping
from functools import wraps
from pathlib import Path

from .survey_provenance import inventory_identity, runtime_provenance

MANIFEST_SCHEMA = 1
DATABASE_SCHEMA = 1
VIEW_FLUSH_INTERVAL = 100
STALE_VIEWS_MARKER = ".survey.views-stale"

_OPERATIONAL_OPTIONS = frozenset({
    "workers", "max_events", "re_enumerate", "name", "root", "inventory",
    "scopes_only", "match", "expand", "scope", "freq_ids",
    "include_outrigger", "empty_age_days",
})
_STATE_NAMES = frozenset({
    "inventory.jsonl", "surveyed_events.txt", "attempts.json",
    "incomplete_events.txt", "no_files_events.jsonl", "enum_cache.json",
    "survey_state.sqlite3", "survey_state.sqlite3-wal",
    "survey_state.sqlite3-shm",
    STALE_VIEWS_MARKER, "survey_runs",
})
_TABLE_COLUMNS = {
    "metadata": ("key", "value"),
    "events": (
        "event_key", "scope", "event", "status", "incomplete_json",
        "no_files_json",
    ),
    "records": ("event_key", "ordinal", "row_json"),
}


class SurveyOutputLock:
    """Portable, process-scoped ownership of one survey output directory.

    The lock file is intentionally durable while the operating-system lock is
    not: an interrupted process cannot strand the directory permanently, and
    every cooperating process still sees a single owner from manifest
    validation through the final state render.
    """

    def __init__(self, out: Path) -> None:
        # Canonicalize parent-directory aliases so two paths through symlinks
        # cannot acquire different lock files for the same survey state.
        self.out = Path(out).expanduser().resolve()
        self.path = self.out / ".survey.lock"
        self._handle = None

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b", buffering=0)
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise SystemExit(
                    f"cannot lock survey output {self.out}: "
                    f"{type(exc).__name__}: {exc}") from exc
            raise SystemExit(
                f"survey output is already in use by another active survey: "
                f"{self.out}. Wait for that survey to finish, or choose a "
                "different --name/output directory.") from exc

        self._handle = handle
        # This is diagnostic only; exclusion is provided by the OS lock. The
        # contents may survive a crash without making the lock stale.
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps({"pid": os.getpid()}).encode("ascii") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        handle, self._handle = self._handle, None
        if handle is None:
            return False
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
        return False


def with_survey_output_lock(function):
    """Hold exclusive output ownership for the complete survey call."""
    @wraps(function)
    def wrapped(self, ctx, out_dir, *args, **kwargs):
        with SurveyOutputLock(Path(out_dir)):
            return function(self, ctx, out_dir, *args, **kwargs)
    return wrapped


def atomic_write_lines(path: Path, lines) -> None:
    """Replace a text file from an iterable without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(str(line))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value) -> None:
    atomic_write_lines(
        path, (json.dumps(value, sort_keys=True, separators=(",", ":")),))


class SurveyRun:
    """Record each execution separately from its stable resume configuration.

    The marker survives SIGKILL. It is cleared only after all database views
    have been refreshed; an ordinary exception alone cannot certify freshness.
    Callers must hold SurveyOutputLock and validate the manifest first.
    """
    def __init__(self, out):
        self.out = Path(out)
        self.marker = self.out / STALE_VIEWS_MARKER
        self.record = {"run_id": uuid.uuid4().hex, "status": "running",
                       "started": self.now(), "finished": None,
                       "views_current": False, **runtime_provenance()}
        self.path = self.out / "survey_runs" / (self.record["run_id"] + ".json")

    @staticmethod
    def now():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def publish(self):
        atomic_write_json(self.path, self.record)
        manifest_path = self.out / "survey_manifest.json"
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise RuntimeError("survey manifest disappeared during execution")
        manifest["latest_run"] = dict(self.record)
        atomic_write_json(manifest_path, manifest)

    def __enter__(self):
        atomic_write_json(self.marker, {
            "run_id": self.record["run_id"], "started": self.record["started"],
            "reason": "survey views may lag SQLite; resume the same survey to refresh"})
        self.publish()
        return self

    def views_current(self):
        self.marker.unlink(missing_ok=True)
        self.record["views_current"] = True

    def __exit__(self, kind, error, traceback):
        self.record.update(
            finished=self.now(),
            status="finished" if kind is None else
                   "interrupted" if issubclass(kind, KeyboardInterrupt) else "failed",
            **inventory_identity(self.out / "inventory.jsonl"))
        if kind is not None:
            self.record["error_type"] = kind.__name__
        self.publish()
        if kind is not None and issubclass(kind, KeyboardInterrupt):
            print(f"Survey interrupted; committed state was preserved in {self.out}. "
                  "Rerun the same survey command to resume.", flush=True)
        return False


def _manifest_value(value):
    """Canonical, JSON-safe value for the survey compatibility manifest."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _manifest_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_manifest_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_manifest_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    # The type identity is deterministic and ensures a changed object type
    # changes the manifest. A reader with meaningful object-valued settings can
    # expose exact data through survey_fingerprint(ctx).
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def build_configuration(ctx, scopes, freq_ids, shape,
                        include_outrigger: bool, empty_age_days: int,
                        minimum_archive_bytes: int, survey_schema: int) -> dict:
    """Return every input allowed to affect durable rows or completion state."""
    instrument = ctx.instrument
    geometry = {
        name: _manifest_value(getattr(instrument, name, None))
        for name in (
            "name", "f0_mhz", "bandwidth_mhz", "n_channels", "descending",
            "nyquist_zone", "n_feeds", "nfft", "scopes", "reader",
        )
    }
    reader = {
        "name": getattr(getattr(shape, "info", None), "name", None),
        "class": f"{type(shape).__module__}.{type(shape).__qualname__}",
        "survey_schema": int(survey_schema),
    }
    fingerprint_hook = getattr(shape, "survey_fingerprint", None)
    if callable(fingerprint_hook):
        reader["fingerprint"] = _manifest_value(fingerprint_hook(ctx))
    options = {
        str(key): _manifest_value(value)
        for key, value in sorted(
            (ctx.options or {}).items(), key=lambda pair: str(pair[0]))
        if key not in _OPERATIONAL_OPTIONS
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "source": "cadc-datatrail",
        "scopes": sorted(set(map(str, scopes))),
        "freq_ids": list(map(int, freq_ids)),
        "include_outrigger": bool(include_outrigger),
        "empty_age_days": int(empty_age_days),
        "minimum_archive_bytes": int(minimum_archive_bytes),
        "selection": _manifest_value(ctx.selection),
        "instrument": geometry,
        "reader": reader,
        "shape_options": options,
    }


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _configuration_fingerprint(configuration: dict) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_manifest(out: Path, configuration: dict) -> dict:
    """Validate the run before any cached inventory state is opened or changed."""
    path = out / "survey_manifest.json"
    current = {
        "fingerprint": _configuration_fingerprint(configuration),
        "configuration": configuration,
    }
    if path.exists():
        previous = _read_json(path)
        previous_configuration = (
            previous.get("configuration") if isinstance(previous, dict) else None)
        previous_fingerprint = (
            previous.get("fingerprint") if isinstance(previous, dict) else None)
        if (not isinstance(previous_configuration, dict)
                or not isinstance(previous_fingerprint, str)
                or previous_fingerprint
                != _configuration_fingerprint(previous_configuration)):
            raise SystemExit(
                f"survey manifest is corrupt: {path}. Use a fresh --name/output "
                "directory; existing state was not changed.")
        if previous_fingerprint != current["fingerprint"]:
            raise SystemExit(
                f"survey configuration does not match the state in {out}. "
                f"Existing fingerprint {previous_fingerprint}, requested "
                f"{current['fingerprint']}. An inventory cannot mix scopes, "
                "frequency ids, reader shape, instrument geometry, or shape "
                "options; use a fresh --name/output directory.")
        return previous
    legacy = sorted(name for name in _STATE_NAMES if (out / name).exists())
    if legacy:
        raise SystemExit(
            f"survey state in {out} predates configuration manifests "
            f"({', '.join(legacy)}). Its inputs cannot be proven compatible; "
            "use a fresh --name/output directory instead of appending to it.")
    atomic_write_json(path, current)
    return current


def load_attempts(path: Path) -> dict[str, int]:
    """Load and validate the small cross-resume retry checkpoint."""
    if not path.exists():
        return {}
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise SystemExit(
            f"survey attempt state is corrupt: {path}. "
            "Use a fresh --name/output directory.")
    if any(not isinstance(key, str) or not key or "|" not in key
           for key in raw):
        raise SystemExit(
            f"survey attempt state has invalid event keys: {path}")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in raw.values()):
        raise SystemExit(
            f"survey attempt state has non-integer counts: {path}")
    attempts = dict(raw)
    if any(value < 0 for value in attempts.values()):
        raise SystemExit(f"survey attempt state has negative counts: {path}")
    return attempts


class SurveyStore:
    """Transactional source of truth; JSONL/text files are recoverable views."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self._db = sqlite3.connect(path)
        except sqlite3.DatabaseError as exc:
            raise SystemExit(
                f"cannot open survey state database {path}: {exc}") from exc
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute("PRAGMA journal_mode = WAL")
            with self._db:
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS metadata "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS events ("
                    "event_key TEXT PRIMARY KEY, scope TEXT NOT NULL, "
                    "event TEXT NOT NULL, status TEXT NOT NULL, "
                    "incomplete_json TEXT, no_files_json TEXT)")
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS records ("
                    "event_key TEXT NOT NULL REFERENCES events(event_key) "
                    "ON DELETE CASCADE, ordinal INTEGER NOT NULL, "
                    "row_json TEXT NOT NULL, PRIMARY KEY(event_key, ordinal))")
                for table, expected in _TABLE_COLUMNS.items():
                    actual = tuple(row[1] for row in self._db.execute(
                        f"PRAGMA table_info({table})"))
                    if actual != expected:
                        raise SystemExit(
                            f"invalid survey state table {table!r} in {path}: "
                            f"expected columns {expected}, found {actual}")
                row = self._db.execute(
                    "SELECT value FROM metadata WHERE key='schema'").fetchone()
                if row is None:
                    self._db.execute(
                        "INSERT INTO metadata(key, value) VALUES('schema', ?)",
                        (str(DATABASE_SCHEMA),))
                else:
                    try:
                        schema = int(row[0])
                    except (TypeError, ValueError):
                        raise SystemExit(
                            f"invalid survey state schema {row[0]!r} in {path}")
                    if schema != DATABASE_SCHEMA:
                        raise SystemExit(
                            f"unsupported survey state schema {schema} in {path}")
        except sqlite3.DatabaseError as exc:
            self._db.close()
            raise SystemExit(
                f"invalid survey state database {path}: {exc}") from exc
        except BaseException:
            self._db.close()
            raise

    def completed_keys(self) -> set[str]:
        return {str(row[0]) for row in self._db.execute(
            "SELECT event_key FROM events")}

    def status_counts(self) -> dict[str, int]:
        """Committed event counts by terminal disposition."""
        return {
            str(status): int(count)
            for status, count in self._db.execute(
                "SELECT status, COUNT(*) FROM events GROUP BY status")
        }

    def commit(self, key: str, scope: str, event: str, status: str, records,
               *, incomplete=None, no_files=None) -> None:
        """Commit rows and the done marker together; safe to repeat after a crash."""
        incomplete_json = (json.dumps(list(incomplete), sort_keys=True)
                           if incomplete else None)
        no_files_json = (json.dumps(no_files, sort_keys=True)
                         if no_files is not None else None)
        rows = [json.dumps(dict(record), sort_keys=True) for record in records]
        with self._db:
            self._db.execute(
                "INSERT INTO events(event_key, scope, event, status, "
                "incomplete_json, no_files_json) VALUES(?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(event_key) DO UPDATE SET scope=excluded.scope, "
                "event=excluded.event, status=excluded.status, "
                "incomplete_json=excluded.incomplete_json, "
                "no_files_json=excluded.no_files_json",
                (key, scope, event, status, incomplete_json, no_files_json))
            self._db.execute("DELETE FROM records WHERE event_key=?", (key,))
            self._db.executemany(
                "INSERT INTO records(event_key, ordinal, row_json) VALUES(?, ?, ?)",
                ((key, ordinal, row) for ordinal, row in enumerate(rows)))

    def render_views(self, out: Path) -> None:
        """Atomically regenerate compatibility files from committed state."""
        atomic_write_lines(
            out / "inventory.jsonl",
            (f"{row[0]}\n" for row in self._db.execute(
                "SELECT row_json FROM records ORDER BY event_key, ordinal")))
        atomic_write_lines(
            out / "surveyed_events.txt",
            (f"{row[0]}\n" for row in self._db.execute(
                "SELECT event_key FROM events ORDER BY event_key")))

        def incomplete_lines():
            for key, raw in self._db.execute(
                    "SELECT event_key, incomplete_json FROM events "
                    "WHERE incomplete_json IS NOT NULL ORDER BY event_key"):
                names = json.loads(raw)
                yield f"{key}\tunresolved={','.join(map(str, names))}\n"

        atomic_write_lines(out / "incomplete_events.txt", incomplete_lines())
        atomic_write_lines(
            out / "no_files_events.jsonl",
            (f"{row[0]}\n" for row in self._db.execute(
                "SELECT no_files_json FROM events WHERE no_files_json IS NOT NULL "
                "ORDER BY event_key")))

    def close(self) -> None:
        self._db.close()
