"""Storage-safe archive streaming.

This is the fixed part of the tool: a storage-safe streaming loop. For every
Unit in a selection it:

    fetch (downloader thread(s) stage files onto scratch)
    read  (Reader -> iterable of arrays)
    analyze (Analyzer.consume_file accumulates)
    delete the staged file immediately
    ask the Analyzer to checkpoint its product every N successfully consumed files

Scratch usage is bounded by a semaphore: at most `max_staged_files` files are
on disk at once. Downloads may finish in any order, but analyzers that require
source order always receive it.

Restartable: on restart the analyzer re-loads its product, reports which units it
already holds, and the engine processes only the rest.

The engine knows nothing about file layouts, power spectra, or N^2 visibilities
-- only Units, Readers, and Analyzers. That is what makes one engine serve every
telescope/source/reader/analyzer combination.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import queue
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Optional

from .interfaces import (
    Analyzer,
    DataSource,
    Reader,
    RunContext,
    Unit,
    UnreadableUnitError,
    stream_compatibility,
)

_WORKER_POLL_SECONDS = 0.1
# How long an abnormal stop waits for its download workers. A failure aborts
# promptly and retains scratch under any writer still active; an interrupt is
# a request to drain, so a fetch in flight is allowed to finish or fail.
_WORKER_JOIN_SECONDS = 2.0
_INTERRUPT_JOIN_SECONDS = 30.0
_STAGE_KEY_HEX_LENGTH = 16
_PROGRESS_EVERY_UNITS = 25
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_DOWNLOAD_WORKERS = 1
DEFAULT_MAX_STAGED_FILES = 1
_OUTPUT_LOCK_SUFFIX = ".datatrawl.lock"


class StagedFileCleanupError(RuntimeError):
    """A staged file could not be removed, so its scratch slot remains held."""


class ActiveDownloadWorkersError(RuntimeError):
    """Cancellation timed out while a source fetch was still in progress.

    ``scratch_dir`` must be retained until the process exits or the worker
    finishes, because the source may still be writing there.
    """

    def __init__(self, message: str, *, scratch_dir: str):
        super().__init__(message)
        self.scratch_dir = scratch_dir


class OutputLockedError(RuntimeError):
    """Another archive scan currently owns the requested output product."""


class QuarantineLedgerLockError(RuntimeError):
    """The quarantine ledger's advisory interprocess lock could not be used."""


class AnalyzerConsumptionError(RuntimeError):
    """An analyzer reported that a unit contributed no usable stream items."""


def _require_positive_int(value, *, label: str, optional: bool = False):
    """Validate engine limits for both CLI and direct library callers."""
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        suffix = " or None" if optional else ""
        raise ValueError(f"{label} must be a positive integer{suffix}, got {value!r}")
    return int(value)


def _validate_units(units: Iterable[Unit]) -> list[Unit]:
    """Materialize and validate stable identities before any filesystem work."""
    materialized = list(units)
    seen: set[str] = set()
    for index, unit in enumerate(materialized):
        if not isinstance(unit, Unit):
            raise TypeError(f"units[{index}] must be a Unit, got {type(unit).__name__}")
        if not isinstance(unit.key, str) or not unit.key:
            raise ValueError(f"units[{index}].key must be a non-empty string")
        if unit.key in seen:
            raise ValueError(f"duplicate unit key {unit.key!r} at units[{index}]")
        seen.add(unit.key)
        if not isinstance(unit.name, str) or not unit.name:
            raise ValueError(f"units[{index}].name must be a non-empty string")
        if unit.meta is not None and not isinstance(unit.meta, Mapping):
            raise TypeError(f"units[{index}].meta must be a mapping or None")
        if unit.meta is not None and "quarantine_key" in unit.meta:
            quarantine_key = unit.meta["quarantine_key"]
            if not isinstance(quarantine_key, str) or not quarantine_key:
                raise ValueError(
                    f"units[{index}].meta['quarantine_key'] must be a "
                    "non-empty string")
    return materialized


def _rm(path: Optional[str]) -> None:
    """Remove a staged file or raise; a missing path is already clean.

    Callers may release a staging semaphore slot only after this function
    returns.  Silently swallowing an unlink failure would make the semaphore
    describe *logical* capacity while undeleted files continued consuming real
    scratch space.
    """
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise StagedFileCleanupError(
            f"could not remove staged file {path!r}: "
            f"{type(exc).__name__}: {exc}. Its scratch slot was not released; "
            "remove the file and rerun."
        ) from exc
    if os.path.lexists(path):
        raise StagedFileCleanupError(
            f"staged file {path!r} still exists after removal. Its scratch "
            "slot was not released; remove the file and rerun."
        )


def _staged_copy_incomplete(dest, unit) -> str | None:
    """Why a staged copy cannot be judged: missing, or short of the archive's
    recorded size. Either is a staging failure to retry, never a quarantine."""
    if not dest or not os.path.exists(dest):
        return "staged copy missing (errno 2)"
    meta = getattr(unit, "meta", None) or {}
    expected = meta.get("size_bytes")
    try:
        expected = int(expected) if expected is not None else None
    except (TypeError, ValueError):
        expected = None
    actual = os.path.getsize(dest)
    if expected is not None and 0 < expected and actual < expected:
        return f"staged copy short ({actual} of {expected} bytes)"
    return None


def _stage_name(unit: Unit) -> str:
    """Scratch filename derived from the stable key, so two units with the same
    basename (the same file under different paths, or a name reused across a
    selection) never collide on disk and corrupt each other's run."""
    h = hashlib.sha256(unit.key.encode("utf-8")).hexdigest()[:_STAGE_KEY_HEX_LENGTH]
    return f"{h}_{os.path.basename(unit.name) or 'file'}"


class _OutputLock:
    """Non-blocking, cross-process advisory lock for one analyzer product.

    The sidecar is intentionally retained after release.  Deleting an advisory
    lock file permits a second process to lock a newly-created inode while a
    first process still holds the old one, defeating mutual exclusion.
    """

    def __init__(self, out_path: str):
        absolute = os.path.abspath(out_path)
        # Resolve parent-directory symlinks so aliases of the same output cannot
        # acquire independent sidecars and bypass mutual exclusion.
        lock_target = os.path.realpath(absolute)
        directory, basename = os.path.split(lock_target)
        self.out_path = absolute
        self.path = os.path.join(
            directory, f".{basename}{_OUTPUT_LOCK_SUFFIX}")
        self._fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self._fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                # msvcrt locks an existing byte from the current file offset.
                import msvcrt

                if os.fstat(self._fh.fileno()).st_size == 0:
                    self._fh.write(b"\0")
                    self._fh.flush()
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise OutputLockedError(
                f"output product {self.out_path!r} is already locked by "
                "another archive scan. Wait for that run to finish, or choose "
                f"a different --out path. Lock: {self.path!r}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._fh is None:
            return False
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None
        return False


class _QuarantineLedgerLock:
    """Blocking advisory lock for one shared quarantine ledger.

    Different analyzer products intentionally have different output locks but
    can share a source/reader quarantine ledger. This short-lived lock
    serializes each ledger read or append without holding it for an entire scan.

    The sidecar is kept on the same filesystem and is never deleted: deleting a
    locked sidecar can split ownership across two inodes. As with ``_OutputLock``,
    exclusion depends on every writer cooperating and on the backing filesystem
    correctly implementing POSIX ``flock`` or Windows byte-range locks.
    """

    def __init__(self, ledger_path: str):
        absolute = os.path.abspath(ledger_path)
        lock_target = os.path.realpath(absolute)
        directory, basename = os.path.split(lock_target)
        self.ledger_path = absolute
        self.path = os.path.join(
            directory, f".{basename}{_OUTPUT_LOCK_SUFFIX}")
        self._fh = None

    def __enter__(self):
        try:
            os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
            self._fh = open(self.path, "a+b")
            if os.name == "nt":
                import msvcrt

                if os.fstat(self._fh.fileno()).st_size == 0:
                    self._fh.write(b"\0")
                    self._fh.flush()
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            raise QuarantineLedgerLockError(
                f"cannot lock quarantine ledger {self.ledger_path!r} via "
                f"{self.path!r}: {type(exc).__name__}: {exc}. The ledger must "
                "live on a filesystem that honors advisory file locks; choose "
                "a supported shared --quarantine path or use --no-quarantine."
            ) from exc
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._fh is None:
            return False
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None
        return False


@dataclass
class RunResult:
    out_path: str
    n_total: int
    n_done: int
    n_new: int
    n_failed: int
    n_quarantined: int = 0
    # True only when out_path is a validated resumed product or was written by
    # this run.  In particular, an all-quarantined/all-failed run has no product.
    product_available: bool = False
    # Distinguish a no-op resume from a checkpoint/final save in this invocation.
    product_written: bool = False
    resumed: bool = False
    stopped_after_checkpoint: bool = False


@dataclass
class _ReadyItem:
    ordinal: int
    unit: Unit
    dest: str
    ok: bool
    error: str


@dataclass
class _WorkerFailure:
    ordinal: Optional[int]
    unit: Optional[Unit]
    error: BaseException


class _ReaderIterationError(RuntimeError):
    """A reader failed while yielding arrays from a staged file."""

    def __init__(self, message: str, *, quarantinable: bool):
        super().__init__(message)
        self.quarantinable = quarantinable


def _reader_arrays(reader: Reader, path: str, ctx: RunContext):
    """Yield reader arrays while preserving the reader/analyzer error boundary."""
    try:
        yield from reader.iter_arrays(path, ctx)
    except UnreadableUnitError as exc:
        raise _ReaderIterationError(
            f"{type(exc).__name__}: {exc}", quarantinable=True
        ) from exc
    except Exception as exc:                              # noqa: BLE001
        raise _ReaderIterationError(
            f"{type(exc).__name__}: {exc}", quarantinable=False
        ) from exc


def _quarantine_key(unit: Unit) -> str:
    """Stable logical identity used by the quarantine ledger.

    Unit.key is the generic default. A source may provide `quarantine_key` in
    metadata when the physical fetch URI can change while the logical file stays
    the same.
    """
    meta = unit.meta or {}
    value = meta.get("quarantine_key")
    return str(unit.key if value is None else value)


def _load_quarantine_unlocked(path: str) -> set[str]:
    """Parse one ledger while its caller holds ``_QuarantineLedgerLock``.

    A malformed or historical name-only record fails closed: basename matching
    can exclude an unrelated unit, and silently skipping a damaged line can
    reintroduce a file the user intentionally quarantined.
    """
    keys: set[str] = set()
    if not os.path.exists(path):
        return keys
    with open(path) as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"malformed quarantine record at {path}:{line_number}; "
                    "repair or remove the ledger before rerunning") from exc
            if not isinstance(rec, dict) or not isinstance(
                    rec.get("quarantine_key"), str):
                raise RuntimeError(
                    f"invalid quarantine record at {path}:{line_number}: "
                    "expected a string quarantine_key; migrate or remove this "
                    "historical ledger before rerunning")
            keys.add(rec["quarantine_key"])
    return keys


def _load_quarantine(path: Optional[str]) -> set[str]:
    """Return a consistent snapshot of stable quarantined logical keys."""
    if not path:
        return set()
    with _QuarantineLedgerLock(path):
        return _load_quarantine_unlocked(path)


def _append_quarantine(path: Optional[str], unit: Unit, reason: str) -> None:
    """Append one durable record, serialized and deduplicated across scans."""
    if not path:
        return
    rec = {
        "quarantine_key": _quarantine_key(unit),
        "unit_key": unit.key,
        "unit_name": unit.name,
        "reason": reason,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with _QuarantineLedgerLock(path):
        # Two active products may discover the same bad unit from a stale
        # run-start snapshot. Check again under the ledger lock so that race
        # produces one disposition record rather than duplicate lines.
        if rec["quarantine_key"] in _load_quarantine_unlocked(path):
            return
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def run(
    *,
    source: DataSource,
    reader: Reader,
    analyzer: Analyzer,
    units: Iterable[Unit],
    out_path: str,
    tmp_dir: str,
    ctx: RunContext,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    max_staged_files: int = DEFAULT_MAX_STAGED_FILES,
    max_files: Optional[int] = None,
    max_frames_per_file: Optional[int] = None,
    quarantine_path: Optional[str] = None,
    stop_after_checkpoint: bool = False,
    verbose: bool = True,
) -> RunResult:
    """Run one storage-bounded analysis while exclusively owning its product."""
    with _OutputLock(out_path):
        return _run_with_output_lock_held(
            source=source,
            reader=reader,
            analyzer=analyzer,
            units=units,
            out_path=out_path,
            tmp_dir=tmp_dir,
            ctx=ctx,
            checkpoint_every=checkpoint_every,
            download_workers=download_workers,
            max_staged_files=max_staged_files,
            max_files=max_files,
            max_frames_per_file=max_frames_per_file,
            quarantine_path=quarantine_path,
            stop_after_checkpoint=stop_after_checkpoint,
            verbose=verbose,
        )


def _run_with_output_lock_held(
    *,
    source: DataSource,
    reader: Reader,
    analyzer: Analyzer,
    units: Iterable[Unit],
    out_path: str,
    tmp_dir: str,
    ctx: RunContext,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    max_staged_files: int = DEFAULT_MAX_STAGED_FILES,
    max_files: Optional[int] = None,
    max_frames_per_file: Optional[int] = None,
    quarantine_path: Optional[str] = None,
    stop_after_checkpoint: bool = False,
    verbose: bool = True,
) -> RunResult:
    checkpoint_every = _require_positive_int(
        checkpoint_every, label="checkpoint_every")
    download_workers = _require_positive_int(
        download_workers, label="download_workers")
    max_staged_files = _require_positive_int(
        max_staged_files, label="max_staged_files")
    max_files = _require_positive_int(max_files, label="max_files", optional=True)
    max_frames_per_file = _require_positive_int(
        max_frames_per_file, label="max_frames_per_file", optional=True)

    contract = stream_compatibility(reader.info, analyzer.info)
    if not contract.compatible:
        raise SystemExit(
            f"incompatible reader/analyzer stream: {contract.detail}. "
            "Choose a matching pair.")

    all_units = _validate_units(units)
    current_unit_keys = {unit.key for unit in all_units}
    n_total = len(all_units)
    units = all_units
    if max_files:
        units = units[:max_files]

    requires_in_order = bool(getattr(analyzer, "requires_in_order", False))

    # Make engine-level run parameters visible to the analyzer BEFORE resume, so it
    # can stamp them into its product and refuse an incompatible resume (e.g. a
    # capped smoke-test product must not be silently "completed" by a full run).
    # RunContext accepts any Mapping, including immutable mappings supplied by
    # library callers. Copy at the engine boundary before adding run invariants.
    ctx.options = dict(ctx.options or {})
    ctx.options["max_frames_per_file"] = max_frames_per_file

    # Resume: let the analyzer re-load (and validate) its own product.  Do not
    # coerce this result: False means specifically "the product was absent";
    # None, 0, or another false-y value signals a broken analyzer contract.
    resume_result = analyzer.resume(out_path, ctx)
    analyzer_name = getattr(
        getattr(analyzer, "info", None),
        "name",
        type(analyzer).__name__,
    )
    if type(resume_result) is not bool:
        raise TypeError(
            f"analyzer {analyzer_name!r} resume() must return exactly True or "
            f"False, got {resume_result!r} ({type(resume_result).__name__})"
        )
    resumed = resume_result
    if resumed and not os.path.exists(out_path):
        raise RuntimeError(
            f"analyzer {analyzer_name!r} reported that it resumed {out_path!r}, "
            "but no product exists at that path."
        )
    if not resumed and os.path.exists(out_path):
        raise RuntimeError(
            f"analyzer {analyzer_name!r} reported that it did not resume "
            f"{out_path!r}, but a product already exists at that path. The "
            "existing product was left unchanged; fix the analyzer's resume() "
            "implementation or choose a different --out path."
        )
    done_keys = set(analyzer.processed_keys()) if resumed else set()
    stale_keys = sorted(done_keys - current_unit_keys)
    if stale_keys:
        sample = ", ".join(repr(key) for key in stale_keys[:3])
        remainder = len(stale_keys) - 3
        if remainder > 0:
            sample += f", and {remainder} more"
        raise SystemExit(
            f"existing product {out_path!r} contains processed units outside "
            f"the current source scope ({sample}); refusing to modify it. "
            "Restore the original source scope or choose a different output path."
        )

    quarantined_keys = _load_quarantine(quarantine_path)
    if requires_in_order and resumed:
        recorded_order = analyzer.processed_key_order()
        if recorded_order is None:
            raise TypeError(
                f"analyzer {analyzer_name!r} requires in-order delivery but does "
                "not expose its resumed processed-key order"
            )
        if (
            any(not isinstance(key, str) or not key for key in recorded_order)
            or len(recorded_order) != len(set(recorded_order))
            or set(recorded_order) != done_keys
        ):
            raise RuntimeError(
                f"analyzer {analyzer_name!r} resumed an invalid processed-key order"
            )
        current_order = [
            unit.key
            for unit in all_units
            if _quarantine_key(unit) not in quarantined_keys
        ]
        expected_order = current_order[: len(recorded_order)]
        if recorded_order != expected_order:
            raise SystemExit(
                f"existing product {out_path!r} was processed in an order that "
                "does not match the current source scope; refusing to append. "
                "Restore the original source scope or choose a different output path."
            )
    if verbose and resumed:
        print(f"resume: {len(done_keys)} unit(s) already in {out_path}")

    # Quarantine: records use a stable source-defined identity; name-only
    # historical records are rejected because they can match unrelated units.
    selected_quarantined = {
        _quarantine_key(unit) for unit in units
        if _quarantine_key(unit) in quarantined_keys
    }
    n_quarantined = len(selected_quarantined)
    if verbose and n_quarantined:
        print(f"quarantine: {n_quarantined} file(s) excluded as bad "
              f"(recorded in {quarantine_path})")

    todo = [u for u in units if u.key not in done_keys
            and _quarantine_key(u) not in quarantined_keys]
    if verbose:
        print(f"{n_total} unit(s) total, {len(done_keys)} done, "
              f"{n_quarantined} quarantined, {len(todo)} to process "
              f"-> {out_path}")
    if not todo:
        if resumed:
            print("nothing to do -- selection already complete in this product")
            print(f"product: {out_path}")
        else:
            print("nothing to do -- no product exists; no selected unit "
                  "requires processing")
        return RunResult(
            out_path=out_path,
            n_total=n_total,
            n_done=len(done_keys),
            n_new=0,
            n_failed=0,
            n_quarantined=n_quarantined,
            product_available=resumed,
            product_written=False,
            resumed=resumed,
        )

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Producer/consumer: downloader thread(s) stage files; the MAIN thread does
    # all reading, accumulation, and checkpointing as the sole writer while the
    # run-level output lock prevents a second process from writing this product.
    #
    # A slot is held from fetch through analysis and deletion.
    n_workers = min(download_workers, max_staged_files, len(todo))
    n_slots = max_staged_files
    stage_slots = threading.BoundedSemaphore(n_slots)
    work_q: "queue.Queue[tuple[int, Unit]]" = queue.Queue()
    for ordinal, unit in enumerate(todo):
        work_q.put((ordinal, unit))
    ready_q: "queue.Queue[_ReadyItem]" = queue.Queue(maxsize=n_slots + 1)
    worker_failures: "queue.Queue[_WorkerFailure]" = queue.Queue()
    pending: dict[int, _ReadyItem | _WorkerFailure] = {}
    cleanup_failures: list[StagedFileCleanupError] = []
    cleanup_lock = threading.Lock()
    stop = threading.Event()
    ordered_stop_reason: Optional[BaseException] = None

    def _record_cleanup_failure(exc: StagedFileCleanupError) -> None:
        with cleanup_lock:
            cleanup_failures.append(exc)

    def _queue_ready(item: _ReadyItem) -> bool:
        """Queue a fetched item, but let cancellation wake blocked producers."""
        while not stop.is_set():
            try:
                ready_q.put(item, timeout=_WORKER_POLL_SECONDS)
                return True
            except queue.Full:
                pass
        return False

    def _discard_item(item: _ReadyItem | _WorkerFailure) -> None:
        """Delete one buffered file and release its slot."""
        if isinstance(item, _WorkerFailure):
            return
        try:
            _rm(item.dest)
        except StagedFileCleanupError as exc:
            _record_cleanup_failure(exc)
        else:
            stage_slots.release()

    def _discard_ready() -> None:
        """Delete buffered files, retaining slots whose delete fails."""
        buffered = list(pending.values())
        pending.clear()
        while True:
            try:
                buffered.append(ready_q.get_nowait())
            except queue.Empty:
                break
        for item in buffered:
            _discard_item(item)

    def _downloader() -> None:
        ordinal: Optional[int] = None
        unit: Optional[Unit] = None
        dest: Optional[str] = None
        owns_slot = False
        try:
            while not stop.is_set():
                while not stop.is_set():
                    if stage_slots.acquire(timeout=_WORKER_POLL_SECONDS):
                        owns_slot = True
                        break
                if not owns_slot:
                    return
                if stop.is_set():
                    stage_slots.release()
                    owns_slot = False
                    return

                try:
                    ordinal, unit = work_q.get_nowait()
                except queue.Empty:
                    stage_slots.release()
                    owns_slot = False
                    return

                dest = os.path.join(tmp_dir, _stage_name(unit))
                # Expected transient/file-level failures are the source's
                # explicit ``(False, detail)`` disposition. An exception means
                # the source implementation or environment failed and must stop
                # the run; treating it as a retryable file would hide the bug.
                fetch_result = source.fetch(unit, dest)
                if (not isinstance(fetch_result, tuple)
                        or len(fetch_result) != 2
                        or not isinstance(fetch_result[0], bool)
                        or not isinstance(fetch_result[1], str)):
                    raise TypeError(
                        f"source.fetch() must return (bool, str), got "
                        f"{fetch_result!r}")
                ok, err = fetch_result
                if not _queue_ready(_ReadyItem(ordinal, unit, dest, ok, err)):
                    return
                # The consumer now owns cleanup and the corresponding slot.
                owns_slot = False
                ordinal = None
                unit = None
                dest = None
        except BaseException as exc:                           # noqa: BLE001
            # BaseException (notably SystemExit) otherwise terminates this daemon
            # silently and leaves the main thread waiting forever for len(todo)
            # ready items.  The unbounded terminal queue cannot be back-pressured.
            worker_failures.put(_WorkerFailure(ordinal, unit, exc))
        finally:
            if owns_slot:
                try:
                    _rm(dest)
                except StagedFileCleanupError as exc:
                    _record_cleanup_failure(exc)
                else:
                    stage_slots.release()

    def _raise_worker_failure(failure: _WorkerFailure) -> None:
        label = failure.unit.name if failure.unit is not None else "<unknown>"
        exc = failure.error
        raise RuntimeError(
            f"download worker terminated while fetching {label}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    def _store_pending(item: _ReadyItem | _WorkerFailure) -> None:
        ordinal = item.ordinal
        if ordinal is None:
            if not isinstance(item, _WorkerFailure):
                raise RuntimeError("download outcome has no unit ordinal")
            _raise_worker_failure(item)
        if ordinal in pending:
            _discard_item(item)
            raise RuntimeError(f"duplicate download outcome for unit {ordinal}")
        pending[ordinal] = item

    def _next_ready(expected: Optional[int] = None) -> _ReadyItem:
        """Return one staged file, preserving order when requested."""
        while True:
            while True:
                try:
                    failure = worker_failures.get_nowait()
                except queue.Empty:
                    break
                if expected is None:
                    _raise_worker_failure(failure)
                _store_pending(failure)
            if expected is not None and expected in pending:
                item = pending.pop(expected)
                if isinstance(item, _WorkerFailure):
                    _raise_worker_failure(item)
                return item
            try:
                item = ready_q.get(timeout=_WORKER_POLL_SECONDS)
            except queue.Empty:
                if not any(worker.is_alive() for worker in workers):
                    while True:
                        try:
                            failure = worker_failures.get_nowait()
                        except queue.Empty:
                            break
                        if expected is None:
                            _raise_worker_failure(failure)
                        _store_pending(failure)
                    if expected is not None and expected in pending:
                        continue
                    raise RuntimeError(
                        "all download workers exited before reporting an outcome "
                        "for every selected unit"
                    )
                continue
            if expected is None or item.ordinal == expected:
                return item
            _store_pending(item)

    workers = [threading.Thread(target=_downloader, daemon=True)
               for _ in range(n_workers)]
    started_workers: list[threading.Thread] = []
    started = False
    product_written = False
    t0, got, fail, quar = time.time(), 0, 0, 0
    run_failure: Optional[BaseException] = None
    planned_stop = False
    try:
        for worker in workers:
            try:
                worker.start()
            except BaseException:                           # noqa: BLE001
                # A custom/threading implementation may start and then raise.
                # Track any thread with an identity so cleanup still joins it.
                if worker.ident is not None:
                    started_workers.append(worker)
                raise
            else:
                started_workers.append(worker)
        if verbose:
            print(f"streaming with {len(started_workers)} download worker(s), "
                  f"<= {n_slots} file(s) on scratch", flush=True)

        for ordinal in range(len(todo)):
            item = _next_ready(ordinal if requires_in_order else None)
            unit, dest, ok, err = item.unit, item.dest, item.ok, item.error
            try:
                if not ok:
                    # fetch failure == transient (network/cert) -> retry on re-run
                    fail += 1
                    print(f"  FAIL fetch {unit.name}: {err}", file=sys.stderr)
                    if requires_in_order:
                        ordered_stop_reason = RuntimeError(
                            f"fetch failed for {unit.name}: {err}"
                        )
                        stop.set()
                        break
                    continue
                try:
                    # Source-supplied unit metadata (the inventory row:
                    # scope, event, obs_date, ...) is opaque to the engine
                    # but load-bearing for analyzers; no file attribute
                    # carries it. The probe reads the actual staged bytes,
                    # so it wins on any key collision.
                    meta = {**dict(unit.meta), **dict(reader.probe(dest))}
                except UnreadableUnitError as exc:
                    incomplete = _staged_copy_incomplete(dest, unit)
                    if incomplete is not None:
                        # The staged copy vanished or is shorter than the
                        # archive says the object is: a staging failure, not
                        # a property of the archived bytes. Leave the unit
                        # unrecorded so the next resume fetches it again.
                        # (2026-09-01: a staging directory removed under a
                        # live scan quarantined 14 readable units this way.)
                        fail += 1
                        print(f"  FAIL read {unit.name}: {incomplete}; "
                              "will refetch on resume", file=sys.stderr)
                        if requires_in_order:
                            ordered_stop_reason = RuntimeError(
                                f"staged copy incomplete for {unit.name}: "
                                f"{incomplete}"
                            )
                            stop.set()
                            break
                        continue
                    # The reader explicitly classified this as a deterministic
                    # problem with the staged bytes. Only that narrow class is
                    # safe to remember in the persistent quarantine ledger.
                    reason = f"probe/read: {type(exc).__name__}: {exc}"
                    if quarantine_path:
                        quar += 1
                        quarantined_keys.add(_quarantine_key(unit))
                        _append_quarantine(quarantine_path, unit, reason)
                        print(f"  QUARANTINE {unit.name}: {reason}", file=sys.stderr)
                    else:
                        fail += 1
                        print(f"  FAIL read {unit.name}: {reason}", file=sys.stderr)
                    if requires_in_order and not quarantine_path:
                        ordered_stop_reason = RuntimeError(
                            f"read failed for {unit.name}: {reason}"
                        )
                        stop.set()
                        break
                    continue
                except Exception as exc:                    # noqa: BLE001
                    reader_name = getattr(
                        getattr(reader, "info", None),
                        "name",
                        type(reader).__name__,
                    )
                    raise RuntimeError(
                        f"reader {reader_name!r} failed while probing "
                        f"{unit.name}: {type(exc).__name__}: {exc}. The file "
                        "was not quarantined because the reader did not "
                        "classify the error as an unreadable unit."
                    ) from exc
                # Source/inventory metadata supplies selection/provenance fields,
                # while the Reader reports facts measured from the staged bytes.
                # If both provide a key, the probe is authoritative: an inventory
                # must never relabel the scientific content that was actually read.
                probe_meta = meta
                meta = dict(unit.meta or {})
                meta.update(probe_meta)
                meta["unit_key"] = unit.key
                meta["unit_name"] = unit.name
                if not started:
                    analyzer.begin(ctx, meta)     # may raise SystemExit on a bad resume
                    started = True
                try:
                    reader_arrays = _reader_arrays(reader, dest, ctx)
                    try:
                        arrays = reader_arrays
                        if max_frames_per_file:
                            arrays = itertools.islice(
                                reader_arrays, max_frames_per_file)
                        consumed = analyzer.consume_file(arrays, meta)
                    finally:
                        # An analyzer may stop early, and islice deliberately does
                        # not close its input. Close the underlying generator while
                        # the staged path still exists (essential for HDF5/Windows).
                        reader_arrays.close()
                except _ReaderIterationError as exc:
                    reason = f"read: {exc}"
                    if quarantine_path and exc.quarantinable:
                        quarantined_keys.add(_quarantine_key(unit))
                        _append_quarantine(quarantine_path, unit, reason)
                        print(f"  QUARANTINE {unit.name}: {reason}", file=sys.stderr)
                        disposition = (
                            "The file was quarantined; rerun the same command to "
                            "resume without it."
                        )
                    elif exc.quarantinable:
                        disposition = (
                            "Quarantine is disabled; fix or remove the file before "
                            "rerunning."
                        )
                    else:
                        disposition = (
                            "The file was not quarantined because the reader did "
                            "not classify the error as an unreadable unit."
                        )
                    raise RuntimeError(
                        f"reader failed while streaming {unit.name}: {exc}. "
                        "The current analyzer state was not checkpointed. "
                        f"{disposition}"
                    ) from exc
                except Exception as exc:                    # noqa: BLE001
                    analyzer_name = getattr(
                        getattr(analyzer, "info", None),
                        "name",
                        type(analyzer).__name__,
                    )
                    raise RuntimeError(
                        f"analyzer {analyzer_name!r} failed on {unit.name}: "
                        f"{type(exc).__name__}: {exc}. The file was not "
                        "quarantined; analyzer exceptions are run-level errors."
                    ) from exc
                if (isinstance(consumed, bool)
                        or not isinstance(consumed, Integral)
                        or consumed <= 0):
                    analyzer_name = getattr(
                        getattr(analyzer, "info", None),
                        "name",
                        type(analyzer).__name__,
                    )
                    raise AnalyzerConsumptionError(
                        f"analyzer {analyzer_name!r} returned {consumed!r} for "
                        f"{unit.name}; consume_file() must return a positive "
                        "integer item count. The unit was not marked done or "
                        "checkpointed."
                    )
                got += 1
                done_keys.add(unit.key)
                if verbose and got % _PROGRESS_EVERY_UNITS == 0:
                    s = analyzer.summary()
                    rate = (s.get("count", 0)) / max(time.time() - t0, 1e-9)
                    print(f"  [{len(done_keys)}/{n_total}] {got} new, {fail} failed, "
                          f"{quar} quarantined, {rate:.1f} unit-frames/s  {s}",
                          flush=True)
                if got % checkpoint_every == 0:
                    analyzer.save(out_path)
                    if not os.path.exists(out_path):
                        raise RuntimeError(
                            f"analyzer save returned without creating {out_path!r}"
                        )
                    product_written = True
                    if verbose:
                        print(f"  ...checkpoint ({got} new, {len(done_keys)} total)",
                              flush=True)
                    if stop_after_checkpoint:
                        planned_stop = True
                        stop.set()
                        break
            finally:
                _rm(dest)               # delete the staged file...
                stage_slots.release()   # ...then free its slot for the next download
    except BaseException as exc:                            # noqa: BLE001
        run_failure = exc
    finally:
        stop.set()
        _discard_ready()
        join_window = _WORKER_JOIN_SECONDS
        if planned_stop:
            for w in started_workers:
                w.join()
        else:
            if isinstance(run_failure, KeyboardInterrupt):
                join_window = _INTERRUPT_JOIN_SECONDS
            join_deadline = time.monotonic() + join_window
            for w in started_workers:
                w.join(timeout=max(0.0, join_deadline - time.monotonic()))
        # A fetch may have completed while the workers were being joined.
        _discard_ready()
        if planned_stop and run_failure is None:
            try:
                failure = worker_failures.get_nowait()
            except queue.Empty:
                pass
            else:
                try:
                    _raise_worker_failure(failure)
                except BaseException as exc:               # noqa: BLE001
                    run_failure = exc

    ordered_prefix_saved = False
    prefix_save_failure: Optional[BaseException] = None
    if ordered_stop_reason is not None and started:
        try:
            analyzer.save(out_path)
            if not os.path.exists(out_path):
                raise RuntimeError(
                    f"analyzer save returned without creating {out_path!r}"
                )
        except BaseException as exc:                         # noqa: BLE001
            prefix_save_failure = exc
        else:
            product_written = True
            ordered_prefix_saved = True

    active_workers = [worker for worker in started_workers if worker.is_alive()]
    with cleanup_lock:
        cleanup_failure = cleanup_failures[0] if cleanup_failures else None
    base_cause = run_failure or ordered_stop_reason
    if cleanup_failure is not None and base_cause is not None:
        cleanup_failure.__cause__ = base_cause
        cleanup_failure.__suppress_context__ = True
    if prefix_save_failure is not None:
        prefix_cause = cleanup_failure or base_cause
        if prefix_cause is not None:
            prefix_save_failure.__cause__ = prefix_cause
            prefix_save_failure.__suppress_context__ = True
    primary_failure = (
        prefix_save_failure or cleanup_failure or run_failure
        or ordered_stop_reason
    )
    if active_workers:
        active_error = ActiveDownloadWorkersError(
            f"{len(active_workers)} download worker(s) did not stop within "
            f"{join_window:g} seconds; source.fetch() is still in "
            f"progress. Scratch was retained at {tmp_dir!r} to prevent deletion "
            "under an active writer. Do not start another run with this scratch "
            "directory. Exit this process or wait for the source operation to "
            "finish before removing or reusing it.",
            scratch_dir=tmp_dir,
        )
        if primary_failure is not None:
            raise active_error from primary_failure
        raise active_error
    if prefix_save_failure is not None:
        raise prefix_save_failure
    if cleanup_failure is not None:
        if base_cause is not None:
            raise cleanup_failure from base_cause
        raise cleanup_failure
    if run_failure is not None:
        raise run_failure

    if started and not ordered_prefix_saved and not planned_stop:
        analyzer.save(out_path)
        if not os.path.exists(out_path):
            raise RuntimeError(
                f"analyzer save returned without creating {out_path!r}"
            )
        product_written = True
    product_available = resumed or product_written
    quar_note = f", {quar} quarantined" if quar else ""
    label = "planned checkpoint stop" if planned_stop else "done"
    print(f"{label}: {len(done_keys)}/{n_total} units, {got} new this run, "
          f"{fail} failed{quar_note} | {analyzer.summary()}")
    if product_available:
        print(f"product: {out_path}")
    else:
        print("product: not created (no unit was successfully analyzed)")
    return RunResult(
        out_path=out_path,
        n_total=n_total,
        n_done=len(done_keys),
        n_new=got,
        n_failed=fail,
        n_quarantined=quar,
        product_available=product_available,
        product_written=product_written,
        resumed=resumed,
        stopped_after_checkpoint=planned_stop,
    )
