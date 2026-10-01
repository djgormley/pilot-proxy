"""Interfaces for archive sources, readers, and analyzers.

The engine itself (pipeline.py) is fixed: for every *unit* in a discovered
inventory it does

    fetch (stage one file)  ->  read (file -> frames)  ->  analyze (frames -> product)
                            ->  delete the local file   ->  checkpoint

Everything telescope- or science-specific lives behind one of three interfaces
defined here, so adding a new use is implementing a plugin, not editing the
engine:

  DataSource   WHERE the data is and how to enumerate/stage it.
               (the reference one is CADC + the CHIME/FRB Datatrail archive)

  Reader       WHAT a staged file looks like: turn a path into an iterable of
               arrays plus the per-unit metadata the analyzer needs. The reader
               also owns the ARCHIVE FILE SHAPE -- which files one event
               contributes and what they are named (survey_files) -- so an
               archive survey and a later read share one naming definition.
               (the reference one is CHIME 4+4-bit baseband HDF5)

  Analyzer      the SCIENCE: consume those arrays in a single streaming pass and
               accumulate a product that can be checkpointed and saved.
               (the reference one is the averaged power-spectrum analyzer)

A fourth axis, the *instrument* (band/channelization geometry), is data, not
code -- it is a YAML file under instruments/ and is handled by instruments.py.

Plugins advertise themselves through `describe()` so the `list` / `doctor`
discovery commands can show a newcomer what exists and what is still a stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Optional

# --------------------------------------------------------------------------
# Discovery metadata
# --------------------------------------------------------------------------
READY = "ready"            # implemented and validated for production use
EXPERIMENTAL = "experimental"  # works, but not yet trusted for science
STUB = "stub"             # interface only -- TODOs inside; will raise if run

_STATUS_ORDER = {READY: 0, EXPERIMENTAL: 1, STUB: 2}


# Semantic types for the items a Reader yields to an Analyzer. These are not
# filenames or persisted product schemas: they describe the in-memory stream at
# the reader/analyzer boundary. A new kind should be specific enough that two
# plugins declaring the same value really can be composed without guessing.
STREAM_COMPLEX_BASEBAND = "complex-baseband-frame"
STREAM_VISIBILITY_CHUNK = "visibility-chunk"
STREAM_ANY = "*"          # analyzer only: deliberately accepts every stream kind


class SurveyUnavailableError(RuntimeError):
    """A survey stopped cleanly because an external service stayed unavailable.

    Sources raise this after preserving their resumable state. The CLI turns it
    into a nonzero exit without presenting the partial survey as complete.
    """


class UnreadableUnitError(RuntimeError):
    """A reader determined that this unit's bytes/schema are unusable.

    The engine may quarantine a unit only for this explicit data disposition.
    Dependency, environment, and reader-programming exceptions must remain
    run-level failures so good data is never permanently excluded by mistake.
    """


@dataclass(frozen=True)
class PluginInfo:
    """One row in the discovery tables (`list` / `doctor`)."""
    name: str
    kind: str                      # "source" | "reader" | "analyzer"
    summary: str
    status: str = READY            # READY | EXPERIMENTAL | STUB
    instruments: tuple[str, ...] = ()   # telescopes this plugin is known to fit ("*" = any)
    produces: str = ""             # e.g. "<freq_id>.npz", "n2_<scope>.npz"
    requires: tuple[str, ...] = () # human-readable prerequisites (env, creds, deps)
    notes: str = ""
    # Sources only: True if this source pulls from the CADC archive, so `doctor`'s
    # ready-combos want the telescope to declare a default baseband `scopes` list
    # in YAML (a geometry-only telescope still works if you pass --scope). A "local"
    # source leaves this False, so a geometry-only telescope is usable with it even
    # before any archive scope is configured.
    needs_archive_config: bool = False
    # Reader/analyzer stream contract. A Reader declares the one semantic kind
    # yielded by iter_arrays(); an Analyzer declares every kind it can consume.
    # Empty declarations are invalid for an executable reader/analyzer pair.
    stream_kind: str = ""
    accepts_stream_kinds: tuple[str, ...] = ()

    @property
    def status_rank(self) -> int:
        return _STATUS_ORDER.get(self.status, 99)


@dataclass(frozen=True)
class StreamCompatibility:
    """Result for one explicitly declared reader -> analyzer stream contract."""

    reader_name: str
    analyzer_name: str
    compatible: bool
    reader_kind: str
    accepted_kinds: tuple[str, ...]

    @property
    def detail(self) -> str:
        """Human-readable explanation suitable for CLI diagnostics."""
        if self.compatible is True:
            if STREAM_ANY in self.accepted_kinds:
                return (f"analyzer {self.analyzer_name!r} explicitly accepts "
                        "every stream kind")
            return (f"reader {self.reader_name!r} emits {self.reader_kind!r}; "
                    f"analyzer {self.analyzer_name!r} accepts it")
        if not self.reader_kind or not self.accepted_kinds:
            missing = []
            if not self.reader_kind:
                missing.append(f"reader {self.reader_name!r} stream_kind")
            if not self.accepted_kinds:
                missing.append(
                    f"analyzer {self.analyzer_name!r} accepts_stream_kinds")
            return ("stream contract is incomplete (missing " +
                    " and ".join(missing) + ")")
        if self.compatible is False:
            return (f"reader {self.reader_name!r} emits {self.reader_kind!r}, but "
                    f"analyzer {self.analyzer_name!r} accepts only "
                    f"{list(self.accepted_kinds)!r}")
        raise AssertionError("unreachable stream compatibility state")


def stream_compatibility(reader_info: PluginInfo,
                         analyzer_info: PluginInfo) -> StreamCompatibility:
    """Return the declared compatibility of a Reader and Analyzer.

    This is a semantic metadata check; it deliberately does not sample or infer
    array shapes at runtime. Missing and mismatched declarations are both
    rejected before any data is staged.
    """

    raw_reader_kind = getattr(reader_info, "stream_kind", "")
    reader_kind = (raw_reader_kind.strip()
                   if isinstance(raw_reader_kind, str) else "")
    raw_accepted = getattr(analyzer_info, "accepts_stream_kinds", ())
    accepted = (
        tuple(kind.strip() for kind in raw_accepted
              if isinstance(kind, str) and kind.strip())
        if isinstance(raw_accepted, tuple) else ()
    )
    if not reader_kind or not accepted:
        compatible = False
    elif STREAM_ANY in accepted:
        compatible = True
    else:
        compatible = reader_kind in accepted
    return StreamCompatibility(
        reader_name=str(getattr(reader_info, "name", "reader")),
        analyzer_name=str(getattr(analyzer_info, "name", "analyzer")),
        compatible=compatible,
        reader_kind=reader_kind,
        accepted_kinds=accepted,
    )


# --------------------------------------------------------------------------
# A "unit" of work and the streaming context
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Unit:
    """One stage-able item (typically one file) the engine will process.

    `key`   source-namespaced logical identity used for resume. It must remain
            stable if a physical archive URI or storage location changes. A
            source must avoid duplicate logical units within one enumeration.
    `name`  local filename to stage it under.
    `meta`  source-specific fields a reader/analyzer may need (event id, freq_id,
            obs date, size, ...). Opaque to the engine.
    """
    key: str
    name: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Shared, read-only-ish context threaded to reader + analyzer for a run."""
    instrument: Any                      # archive instrument definition
    selection: Any = None                # what the user asked to process (freq_ids,
                                         # events, scope, ...) -- sources parse it
                                         # parsed by archive.selection
    options: Mapping[str, Any] = field(default_factory=dict)
    reader: Any = None                   # the run's Reader instance, when the caller
                                         # has resolved one. survey() consults it for
                                         # the archive file shape (Reader.survey_files)
                                         # so survey and read share one definition of
                                         # what a unit's file is named.


# --------------------------------------------------------------------------
# DataSource: discovery + staging
# --------------------------------------------------------------------------
class DataSource:
    """Where the data lives and how to enumerate/stage it.

    Two responsibilities, deliberately split so the cheap discovery step can be
    cached and re-run without ever touching bulk data:

      enumerate()  -> iterable of Unit, the inventory for a given selection.
      fetch(unit)  -> stage one unit to a local path; return (ok, error).

    Implementations should make `fetch` retry transient failures, use finite
    network connect/read timeouts, and return exactly ``(False, detail: str)``
    for an expected retryable/file failure. Exceptions and malformed returns
    are run-level source failures. A fetch must never require more than one
    staged file to exist at a time; the engine deletes each file right after it
    is analyzed.

    Thread-safety: with the default engine settings (download_workers=1) fetch is
    called serially, in enumerate() order. With multiple workers and multiple
    staging slots, the engine may call fetch() on a single source instance from
    several threads at once. A parallel-capable source must therefore keep no
    mutable per-call state on self (give each thread its own client, e.g. via
    threading.local, or guard a shared one with a lock).
    """
    info: PluginInfo

    def enumerate(self, ctx: RunContext) -> Iterable[Unit]:
        raise NotImplementedError

    def fetch(self, unit: Unit, dest: str) -> tuple[bool, str]:
        raise NotImplementedError

    # Optional: a one-shot survey that writes a persistent inventory file.
    # Sources whose enumerate() is expensive (network listing) override this so
    # `survey` can cache to disk; cheap sources can leave it as enumerate().
    def survey(self, ctx: RunContext, out_dir: str) -> str:
        raise NotImplementedError(
            f"{self.info.name}: survey-to-disk not implemented; this source "
            f"enumerates on demand.")

    def survey_completeness_issues(
            self, out_dir: str) -> Optional[Mapping[str, int]]:
        """Return final-state omission counts after ``survey()``, if supported.

        ``None`` means the source has no completeness contract. A mapping with
        nonzero counts lets the CLI's opt-in strict mode preserve the inventory
        and metadata while returning a failure status for automation.
        """
        return None

    # Optional self-check for `doctor`. Return (ok, problems) -- problems make
    # doctor report NOT READY -- or (ok, problems, notes) to also surface
    # non-fatal "couldn't check" caveats, which doctor renders as a visible [--]
    # skipped line without failing readiness.
    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        return True, []


# --------------------------------------------------------------------------
# Reader: file -> arrays + metadata
# --------------------------------------------------------------------------
class Reader:
    """Turn one staged file into the stream items an analyzer consumes.

    `probe(path)`            -> dict of per-file metadata (e.g. channel center freq,
                                shape, sample rate) WITHOUT reading bulk data.
    `iter_arrays(path, ctx)` -> yield items in streaming order. Most readers yield
                                NumPy-compatible arrays; structured formats may
                                yield mappings when their declared stream contract
                                requires one.

    The reader owns the on-disk format knowledge (dataset names, dtype packing,
    attribute conventions). A different file format = a different reader; the
    engine and analyzer do not change.
    """
    info: PluginInfo
    # Archive sources may consult this when deciding whether a remotely reported
    # object is large enough to be a usable file of this format. It may be a
    # positive byte count or a callable accepting RunContext. The streaming
    # engine itself imposes no minimum file size.
    minimum_archive_bytes: int = 1
    # Archive-capable readers must set a positive schema revision. Increment it
    # whenever survey_files() or annotate_row() changes persisted row semantics;
    # survey resume refuses state created under a different revision.
    survey_schema: Optional[int] = None

    def probe(self, path: str) -> Mapping[str, Any]:
        raise NotImplementedError

    def iter_arrays(self, path: str, ctx: RunContext) -> Iterator:
        raise NotImplementedError

    # -- archive file shape (optional) ------------------------------------
    # An archive survey needs to know, for one event, WHICH files this
    # reader's product contributes and what they are called. That is format
    # knowledge, so it lives on the reader -- the same class that will later
    # open those files -- and survey + read can never drift apart on naming.
    # (A reader that only ever scans pre-listed local files can leave both
    # methods untouched.)
    def survey_files(self, event, common_path, selection,
                     ctx: RunContext) -> Iterable[tuple]:
        """Yield (filename, fields) for every candidate file of one event.

        `filename` is relative to the event's archive common path (it may
        contain a sub-path). `fields` is a mapping of per-file inventory
        columns this format defines -- e.g. the baseband reader yields
        ({"freq_id": ch}) per channel; a per-event calibration product might
        yield a single file with no fields at all. `selection` is whatever
        per-survey spec the source resolved (the baseband survey passes the
        freq_id list); a shape that is not selected that way ignores it.
        Everything yielded here lands verbatim in the inventory row, and the
        row's `name` is what enumerate/fetch later stage -- no re-derivation.
        """
        raise NotImplementedError(
            f"reader {getattr(self.info, 'name', type(self).__name__)!r} does "
            f"not declare an archive file shape (survey_files); an archive "
            f"survey needs a reader that does. See docs/ADDING_A_READER.md.")

    def annotate_row(self, row: dict, instrument) -> None:
        """Optionally enrich one verified inventory row in place.

        Called by survey after the file's size is known, with the run's
        instrument (which may be None). The baseband reader adds ``freq_mhz``
        and a file-size-derived ``n_frames_estimate`` here; the default adds
        nothing.
        """
        return None

    def survey_fingerprint(self, ctx: RunContext) -> Any:
        """Return extra JSON-safe inputs that determine survey inventory rows.

        Archive survey state uses this value to reject an incompatible resume.
        Readers whose row shape depends only on their class, selection,
        instrument, and ``minimum_archive_bytes`` can keep the default.
        """
        return None

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        return True, []


# --------------------------------------------------------------------------
# Analyzer: streaming accumulation -> saveable product
# --------------------------------------------------------------------------
class Analyzer:
    """The science. Accumulate a product over a stream of arrays, in one pass.

    The analyzer OWNS its product file (the saved .npz / .h5 / ...): it writes it,
    re-loads it on resume, and reports which units are already in it. Owning the
    file is what lets an analyzer keep a specific, downstream-readable product
    schema (e.g. the spectrum analyzer writes a self-describing <freq_id>.npz a
    downstream tool can read directly) instead of being wrapped in an opaque
    engine checkpoint.

    Lifecycle, driven by the engine:

        resume(path, ctx)          -> bool : load an existing product if present
                                             AND compatible; True if it resumed
        processed_keys()           -> set  : Unit.key values already in the product
        begin(ctx, first_meta)             : once, when the first new file is read
        consume_file(arrays, meta) -> n    : per file; update accumulators
        save(path)                         : persist the product, provenance, and
                                             processed keys for recovery
        summary()                  -> dict : small human-readable status line

    Keeping all accumulator writes on the engine's main thread (the engine only
    parallelises fetch) means an analyzer needs no locking.

    Ordering: set `requires_in_order = True` if the product depends on the order
    files are consumed -- e.g. a running/trailing statistic, a CFAR baseline, or
    anything that is not a commutative accumulation. The engine buffers early
    fetch completions and presents these analyzers with exact source order at any
    supported worker and staging count. Leave it False (the default) for a
    commutative accumulation like a summed PSD.
    """
    info: PluginInfo
    requires_in_order: bool = False

    def resume(self, path: str, ctx: RunContext) -> bool:
        """Load an existing product to continue it; return False only if absent.

        Must raise (not silently continue) if `path` exists but was built with
        incompatible parameters, so two runs can never be mixed into one product.
        """
        raise NotImplementedError

    def processed_keys(self) -> set:
        """Unit.key values already accumulated (for resume skip)."""
        raise NotImplementedError

    def processed_key_order(self) -> Optional[list[str]]:
        """Accumulated Unit.key values in consumption order, when recorded."""
        return None

    def begin(self, ctx: RunContext, first_meta: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def resolve_selection(self, ctx: RunContext, spec: Any) -> Any:
        """Interpret a user `--select` spec for this analysis (optional).

        Default is identity. The spectrum analyzer overrides this to interpret
        `--select` as explicit freq_ids (`844`, `614,706`, `506-552`); other
        analyses give it their own meaning (a scope, a date range, a feed
        list, ...).
        """
        return spec

    def plan_runs(self, ctx: RunContext, spec: Any) -> list:
        """Split a selection into INDEPENDENT runs, each its own product.

        Returns a list of sub-selections; the engine runs once per sub-selection
        with a FRESH analyzer instance, so each gets its own resumable product.

        Default: one run over the whole resolved selection. A per-item analysis
        overrides this -- e.g. the spectrum analyzer returns one sub-selection per
        freq_id, so `--select 614,706` becomes two independent <freq_id>.npz
        products that resume (and fail) independently. Returning a single-item
        list `[freq_id]` makes the engine name that product `<freq_id>.npz`.
        """
        return [self.resolve_selection(ctx, spec)]

    def consume_file(self, arrays: Iterable, meta: Mapping[str, Any]) -> int:
        """Consume one file and return a positive integer item count.

        Returning zero, a boolean, or a non-integer is an error: the engine
        will not checkpoint or mark that unit complete. A reader that proves
        the staged bytes cannot yield a usable item should raise
        ``UnreadableUnitError`` so the engine can apply its quarantine policy.

        Ordering: analyzers with ``requires_in_order = True`` receive files in
        exact source order even when downloads finish out of order. Other
        analyzers may receive files in completion order when concurrent fetch is
        enabled.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def summary(self) -> Mapping[str, Any]:
        return {}

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        return True, []
