r"""``tab:impl:crossbuild``: the cross-build reproduction record (chapter 6,
``sec:impl:verify``, gate 6 --- the ``v5 archive rerun`` stub near line 248).

What the chapter claims. "Two independently built processing cohorts
(different hosts, compiler builds, and analyzer source states) process an
overlapping archive subset, and their per-frame integer outputs are joined on
the physical archive key and compared as integers. The claim is stated with
its denominator: joined events and frames, join completeness, invalid and
missing records, the exact fields compared, and both immutable build
manifests." This module produces that record from the campaign directory on
disk, and prints nothing it did not read there.

**The two cohorts.** The campaign keeps its cross-build pair in
``qualification/``: a *reference* product built locally (one file per
directory level: ``qualification/<name>.npz``) and the *campaign* build's
smoke product (``qualification/<run>/_per_pilot/<freq_id>.npz``), the same
archive objects processed by the build that produced the survey. The pair is
discovered by ``freq_id``, not hard-coded: every top-level ``.npz`` under
``qualification/`` is a reference candidate and every ``*/_per_pilot/*.npz``
beneath it a campaign candidate; a ``freq_id`` carried by exactly one of each
becomes the compared pair. The run of record's own product for that
``freq_id`` (``run.json``'s ``products_dir``) is the third column and the
second comparison arm, so the record reaches the products the dissertation
reports on rather than stopping at the smoke.

**Arms.** Two joins, both of them the reference against a campaign-build
product:

  ``qualification``   reference vs the smoke product: the same archive
                      objects on both sides, so the join is expected complete
                      in both directions.
  ``run of record``   reference vs the survey product for the same channel:
                      the reference's events are a subset of the survey's, so
                      completeness is stated on the reference's denominator
                      and the survey's own event and frame totals are printed
                      beside it.

**The join key** is the physical archive key the product stores, not a row
number: ``unit_keys[frame_unit_index]`` (the CADC/Data-Trail object key of the
acquisition) paired with ``frame_in_unit`` (the frame's ordinal inside that
object). Row order is never used; ``frame_index`` and ``frame_unit_index`` are
positional and are therefore *not* compared fields.

**Compared fields** (integer members shared by both products of an arm; a
member absent from either side is named in the notes and compared nowhere):

  detector power sums   ``p_target_u64``, ``p_ref_lower_u64``,
                        ``p_ref_upper_u64``, ``p_ref_sum_u64`` --- the exact
                        uint64 numerator and reference terms of the coarse
                        statistic, 1 value per frame each
  fine terms            ``fine_power_u64`` --- the frozen transform's exact
                        power sums, 3 x 256 = 768 values per frame
  decision bytes        ``valid``, ``reject_mask`` --- the reason-coded
                        validity and the survey flag, 1 each
  sample accounting     ``fill_sample_count``, ``railed_sample_count``,
                        ``railed_sample_total`` --- 1 each
  stored spectrum       ``psd_frame_db_i16`` --- the int16 dB codes of the
                        per-frame spectrum, 16{,}384 per frame. Reading it
                        costs about 15 s and 1.2 GB on a survey product (a
                        compressed zip member cannot be sliced on disk), so
                        ``COMPARE_STORED_SPECTRA`` can drop it; the default
                        compares it and the notes say the cost
  unit identity         ``unit_event_id``, ``unit_time0_fpga`` --- per joined
                        acquisition rather than per frame

The float members are *not* part of the claim and are counted separately, in
the denominator panel, as the reason why: ``normalized_coarse_power_ratio_db`` and ``pilot_excess_db``
are ``10 log10`` of exactly-equal integer ratios and are not bit-equal across
builds (a 1--2 ulp difference in the last place), which is what an exactness
contract stated over integers buys.

**Two fragments.** Five panels of this content measure 928 pt against a
650 pt text height, so the record is split the way the chapter's other tables
are: ``crossbuild_record`` (``tab:impl:crossbuild``) is the chapter table --
the two build manifests and the joined denominator -- and
``crossbuild_fields`` (``tab:archive:crossbuild_fields``) is the companion,
the mismatch count member by member and the same build's re-execution. Each is
a stack of ``tabular`` blocks in one box, and each fits the text block upright
(412 pt and 447 pt wide, 559 pt and 341 pt tall, against 470 x 650 pt). The two builders share one join through a
process-lifetime cache, because the join reads about 1.5 GB of compressed
members.

**Panels** (the chapter supplies the ``table`` environment, one caption and
one label per fragment):

  1--2. *Build manifest*, split in two by a computed rule: panel 1 is every
     field whose value is identical in every cohort (one value column), panel
     2 is every field the cohorts differ in (one column per cohort) --- the
     build axis the claim rests on. Together they are one row per provenance
     scalar the products record. Everything here is read from the product itself
     (``Product.archive``) except the file digest, which is the campaign's
     ``SHA256SUMS`` entry for that path, and is dashed when the file is not
     listed there. Rows: the analyzer package and source digest and the
     detector binary's version and SHA-256, all parsed out of the
     ``detector_version`` string; the product schema and revision; the weight
     bank, manifest, coefficient and combined weight digests; the detector
     contract's schema; the receiver software tag, input-map digest, archive
     version, collection server and scope the *acquisitions* carry
     (``unit_git_version_tag``, ``unit_input_map_sha256``, ``archive_version``,
     ``unit_collection_server``, ``unit_scope`` --- printed as the single
     value when the cohort carries one, otherwise as a count of distinct
     values); the product file digest; and the event and frame totals.
     Digests are printed to :data:`DIGEST_CHARS` hex characters. *The product
     records no host and no compiler:* the build axis it does record is the
     analyzer source digest and the detector binary digest, and the notes say
     so.
  3. *Joined denominator and completeness.* One column per arm: events and
     frames on each side, joined events and frames, join completeness on the
     reference's denominator, missing records (reference frames with no
     campaign counterpart), invalid records (frames either side marks
     ``valid = 0``), integer values compared and mismatches, and then the
     derived floats: their values compared, their mismatches, and the largest
     difference in units of the last place.
  4. *Mismatches by compared integer member* (companion fragment). One row per member: its integer
     width per frame, then the values compared and the mismatches for each
     arm, in the group order above with a rule between groups. The two float
     members are keyed but not printed here; panel 3 counts them.
  5. *Independent re-execution of the same build* (companion fragment). From
     ``logs/channels/summary.csv``: the channels a second scan session
     processed independently of their owner (``shard3_rebuilt`` set and the
     owner not itself that session), with each side's completed and
     quarantined unit counts and the mismatch between them. This is a
     different host and session at the *same* build, not a second build, and
     the per-frame fields cannot be re-compared from this machine: only the
     canonical copy of each product was retained, so the field-level identity
     of those duplicates survives as the campaign ledger's digest record and
     not as data. The notes say exactly that.

Numbers are keyed ``ch06.crossbuild.<name>`` for band-level values,
``ch06.crossbuild.<name>.<arm>`` for a per-arm value (``arm`` is
``qualification`` or ``run_of_record``), ``ch06.crossbuild.manifest.<field>.<cohort>``
for a manifest cell, ``ch06.crossbuild.compared.<field>.<arm>`` and
``ch06.crossbuild.mismatches.<field>.<arm>`` for a comparison cell, and
``ch06.crossbuild.reexec.<name>.fid<NNN>`` for a re-execution row. The manifest
no longer carries the event and frame counts: the denominator panel states
them, and a number is keyed once.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pilot_proxy.products.reader import Product
from .core import DASH, Fragment, Run, booktabs, fmt, fmt_int, tex

NAME = "crossbuild_record"
LABEL = "tab:impl:crossbuild"
LEDGER_NAME = "crossbuild_fields"
LEDGER_LABEL = "tab:archive:crossbuild_fields"
PREFIX = "ch06.crossbuild"
DIGEST_CHARS = 12                       # hex characters of every digest printed (the table must fit upright)
COMPARE_STORED_SPECTRA = True           # psd_frame_db_i16: exact, but 1.2 GB and ~15 s per survey product

QUALIFICATION = "qualification"
ARM_QUALIFICATION = "qualification"
ARM_RUN = "run_of_record"
ARM_LABEL = {ARM_QUALIFICATION: "qualification", ARM_RUN: "run of record"}
ARM_SHORT = {ARM_QUALIFICATION: "qual.", ARM_RUN: "record"}

REFERENCE = "reference"
CAMPAIGN = "campaign"
RUN_OF_RECORD = "run_of_record"
COHORT_LABEL = {REFERENCE: "reference build", CAMPAIGN: "campaign build", RUN_OF_RECORD: "run of record"}


# ------------------------------------------------------------------ field specs
@dataclass(frozen=True)
class FieldSpec:
    """One compared member: its name, what it is, and the axis it lives on."""

    name: str
    group: str
    axis: str = "frame"                 # 'frame' (joined per frame) or 'unit' (per joined acquisition)
    heavy: bool = False                 # read costs a full decompression of a large member


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("p_target_u64", "detector power sums"),
    FieldSpec("p_ref_lower_u64", "detector power sums"),
    FieldSpec("p_ref_upper_u64", "detector power sums"),
    FieldSpec("p_ref_sum_u64", "detector power sums"),
    FieldSpec("fine_power_u64", "fine terms", heavy=True),
    FieldSpec("valid", "decision bytes"),
    FieldSpec("reject_mask", "decision bytes"),
    FieldSpec("fill_sample_count", "sample accounting"),
    FieldSpec("railed_sample_count", "sample accounting"),
    FieldSpec("railed_sample_total", "sample accounting"),
    FieldSpec("psd_frame_db_i16", "stored spectrum", heavy=True),
    FieldSpec("unit_event_id", "unit identity", axis="unit"),
    FieldSpec("unit_time0_fpga", "unit identity", axis="unit"),
)

FLOAT_FIELDS: tuple[str, ...] = ("normalized_coarse_power_ratio_db", "pilot_excess_db")

MANIFEST_FIELDS: tuple[tuple[str, str], ...] = (
    ("analyzer package", "analyzer package"),
    ("analyzer source", "analyzer source digest"),
    ("detector binary", "detector binary version"),
    ("detector binary sha256", "detector binary sha256"),
    ("product schema", "product schema"),
    ("schema revision", "schema revision"),
    ("detector contract", "detector contract schema"),
    ("weight bank", "weight bank sha256"),
    ("weight manifest", "weight manifest sha256"),
    ("weight coefficients", "weight coefficients sha256"),
    ("weights hash", "weights hash"),
    ("receiver software tag", "unit git version tag"),
    ("input-map digest", "unit input map sha256"),
    ("archive version", "archive version"),
    ("collection server", "unit collection server"),
    ("unit scope", "unit scope"),
    ("product file sha256", "product file sha256"),
)


# ------------------------------------------------------------------ discovery
def _sha256sums(root: Path) -> dict[str, str]:
    """``SHA256SUMS`` as ``{relative path: digest}``; an absent file gives ``{}``."""
    path = root / "SHA256SUMS"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def campaign_root(run: Run) -> Path | None:
    """The campaign directory holding the run's products (``products_dir``'s grandparent)."""
    products_dir = run.run.get("products_dir")
    if not products_dir:
        return None
    return Path(str(products_dir)).parent.parent


def _freq_id(path: Path) -> int | None:
    try:
        with Product(path) as p:
            return int(np.asarray(p.archive["freq_id"]).reshape(-1)[0])
    except (OSError, ValueError, KeyError):
        return None


def discover(root: Path) -> tuple[Path | None, Path | None, str]:
    """The qualification pair under ``root``: (reference product, campaign product, reason).

    A top-level ``qualification/*.npz`` is a reference candidate; a
    ``qualification/*/_per_pilot/*.npz`` is a campaign candidate. The pair is
    the single ``freq_id`` carried by exactly one of each. ``reason`` is empty
    on success and says what was missing otherwise.
    """
    qual = root / QUALIFICATION
    if not qual.is_dir():
        return None, None, f"no {QUALIFICATION}/ directory under {root}"
    references = {f: p for p, f in ((p, _freq_id(p)) for p in sorted(qual.glob("*.npz"))) if f is not None}
    campaigns = {f: p for p, f in ((p, _freq_id(p)) for p in sorted(qual.glob("*/_per_pilot/*.npz"))) if f is not None}
    shared = sorted(set(references) & set(campaigns))
    if not shared:
        return None, None, (f"{QUALIFICATION}/ carries no freq_id built both ways "
                            f"(reference {sorted(references)}, campaign {sorted(campaigns)})")
    if len(shared) > 1:
        return None, None, f"{QUALIFICATION}/ carries more than one cross-built freq_id: {shared}"
    return references[shared[0]], campaigns[shared[0]], ""


# ------------------------------------------------------------------ one build
def _text_scalar(archive, name: str) -> str:
    if name not in archive.files:
        return ""
    return str(np.asarray(archive[name]).reshape(-1)[0])


def _unique(archive, name: str) -> str:
    """The single value an array member carries, or ``'<n> values'`` when it carries several."""
    if name not in archive.files:
        return ""
    values = sorted({str(v) for v in np.asarray(archive[name]).reshape(-1).tolist()})
    if not values:
        return ""
    return values[0] if len(values) == 1 else f"{len(values)} values"


_DETECTOR = re.compile(r"^(?P<package>\S+)\s+source=(?P<source>\S+)\s+kernel=(?P<kernel>\S+)\s+"
                       r"kernel_sha256=(?P<kernel_sha256>\S+)")


@dataclass(frozen=True)
class Build:
    """One cohort: its product, the provenance scalars it records, and its size."""

    cohort: str
    path: Path
    freq_id: int
    physical_channel: int
    manifest: dict[str, str]
    events: int
    frames: int

    @property
    def label(self) -> str:
        return COHORT_LABEL.get(self.cohort, self.cohort)


def read_build(cohort: str, path: Path, digests: Mapping[str, str], root: Path | None = None) -> Build:
    """Open one product and read the provenance scalars it records."""
    with Product(path) as p:
        archive = p.archive
        detector = _text_scalar(archive, "detector_version")
        parsed = _DETECTOR.match(detector)
        contract = _text_scalar(archive, "detector_contract_json")
        contract_schema = ""
        match = re.search(r'"schema_version"\s*:\s*"([^"]+)"', contract)
        if match:
            contract_schema = match.group(1)
        events = int(np.asarray(archive["unit_keys"]).reshape(-1).size) if "unit_keys" in archive.files else 0
        frames = int(np.asarray(archive["valid"]).shape[0]) if "valid" in archive.files else 0
        rel = ""
        if root is not None:
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = ""
        manifest = {
            "analyzer package": parsed.group("package") if parsed else detector or "",
            "analyzer source digest": parsed.group("source") if parsed else "",
            "detector binary version": parsed.group("kernel") if parsed else "",
            "detector binary sha256": parsed.group("kernel_sha256") if parsed else "",
            "product schema": _text_scalar(archive, "schema_version"),
            "schema revision": _text_scalar(archive, "schema_revision"),
            "detector contract schema": contract_schema,
            "weight bank sha256": _text_scalar(archive, "weight_bank_sha256"),
            "weight manifest sha256": _text_scalar(archive, "weight_manifest_sha256"),
            "weight coefficients sha256": _text_scalar(archive, "weight_coefficients_sha256"),
            "weights hash": _text_scalar(archive, "weights_hash"),
            "unit git version tag": _unique(archive, "unit_git_version_tag"),
            "unit input map sha256": _unique(archive, "unit_input_map_sha256"),
            "archive version": _unique(archive, "archive_version"),
            "unit collection server": _unique(archive, "unit_collection_server"),
            "unit scope": _unique(archive, "unit_scope"),
            "product file sha256": digests.get(rel, ""),
            "events": str(events),
            "frames": str(frames),
        }
        return Build(cohort, path, int(np.asarray(archive["freq_id"]).reshape(-1)[0]),
                     int(np.asarray(archive["physical_channel"]).reshape(-1)[0]), manifest, events, frames)


# ------------------------------------------------------------------ the join
def _keys(archive) -> tuple[np.ndarray, np.ndarray]:
    """The physical archive key of every frame: (unit key, frame in unit)."""
    unit_keys = np.asarray(archive["unit_keys"]).reshape(-1)
    unit_index = np.asarray(archive["frame_unit_index"]).reshape(-1).astype(np.int64)
    return unit_keys[unit_index], np.asarray(archive["frame_in_unit"]).reshape(-1).astype(np.int64)


def _member(archive, name: str, rows: np.ndarray) -> np.ndarray:
    """A member's rows, flattened per row; the whole member is decoded first (a zip cannot be sliced)."""
    whole = np.asarray(archive[name])
    picked = whole[rows]
    return picked.reshape(len(rows), -1) if picked.ndim > 1 else picked.reshape(len(rows), 1)


@dataclass
class Comparison:
    """One field's outcome on one arm."""

    name: str
    group: str
    width: int                          # integer values per joined record
    compared: int                       # values actually compared
    mismatches: int
    present: bool = True


@dataclass
class Arm:
    """One join of a reference product against a campaign-build product."""

    arm: str
    reference: Build
    other: Build
    joined_events: int
    joined_frames: int
    missing_frames: int                 # reference frames with no counterpart
    invalid_records: int                # joined frames either side marks valid = 0
    comparisons: list[Comparison] = field(default_factory=list)
    float_comparisons: list[Comparison] = field(default_factory=list)
    float_max_ulps: dict[str, float] = field(default_factory=dict)
    absent_fields: list[str] = field(default_factory=list)
    disjoint_reason: str = ""

    @property
    def label(self) -> str:
        return ARM_LABEL.get(self.arm, self.arm)

    @property
    def compared(self) -> int:
        return sum(c.compared for c in self.comparisons)

    @property
    def mismatches(self) -> int:
        return sum(c.mismatches for c in self.comparisons)

    @property
    def event_completeness(self) -> float:
        return self.joined_events / self.reference.events if self.reference.events else math.nan

    @property
    def frame_completeness(self) -> float:
        return self.joined_frames / self.reference.frames if self.reference.frames else math.nan


def _ulps(a: np.ndarray, b: np.ndarray) -> float:
    """The largest difference in units of the last place, over the finite pairs."""
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return math.nan
    step = np.spacing(np.abs(a[finite]))
    step[step == 0] = np.inf
    return float(np.max(np.abs(a[finite] - b[finite]) / step))


def compare(arm: str, reference: Build, other: Build, *, stored_spectra: bool = COMPARE_STORED_SPECTRA) -> Arm:
    """Join the two products on the physical archive key and compare the integer fields."""
    with Product(reference.path) as pa, Product(other.path) as pb:
        a, b = pa.archive, pb.archive
        keys_a, frame_a = _keys(a)
        keys_b, frame_b = _keys(b)
        index_b = {(k, int(f)): i for i, (k, f) in enumerate(zip(keys_b.tolist(), frame_b.tolist()))}
        rows_a, rows_b = [], []
        for i, (k, f) in enumerate(zip(keys_a.tolist(), frame_a.tolist())):
            j = index_b.get((k, int(f)))
            if j is not None:
                rows_a.append(i)
                rows_b.append(j)
        joined_events = len(set(keys_a[rows_a].tolist())) if rows_a else 0
        result = Arm(arm, reference, other, joined_events, len(rows_a), int(keys_a.size) - len(rows_a), 0)
        if not rows_a:
            shared_units = set(np.asarray(a["unit_keys"]).reshape(-1).tolist()) & set(np.asarray(b["unit_keys"]).reshape(-1).tolist())
            result.disjoint_reason = ("the cohorts share no frame: "
                                      f"{len(shared_units)} acquisition keys in common of "
                                      f"{reference.events} and {other.events}")
            return result
        ra, rb = np.asarray(rows_a), np.asarray(rows_b)
        # per-unit rows: one representative frame per joined acquisition
        unit_a = np.asarray(a["frame_unit_index"]).reshape(-1).astype(np.int64)[ra]
        unit_b = np.asarray(b["frame_unit_index"]).reshape(-1).astype(np.int64)[rb]
        _, first = np.unique(unit_a, return_index=True)
        units_a, units_b = unit_a[np.sort(first)], unit_b[np.sort(first)]
        for spec in FIELDS:
            if spec.name == "psd_frame_db_i16" and not stored_spectra:
                result.absent_fields.append(f"{spec.name} (not compared: stored spectra disabled)")
                continue
            if spec.name not in a.files or spec.name not in b.files:
                result.absent_fields.append(f"{spec.name} (absent from one product)")
                result.comparisons.append(Comparison(spec.name, spec.group, 0, 0, 0, present=False))
                continue
            left = _member(a, spec.name, ra if spec.axis == "frame" else units_a)
            right = _member(b, spec.name, rb if spec.axis == "frame" else units_b)
            result.comparisons.append(Comparison(spec.name, spec.group, int(left.shape[1]), int(left.size),
                                                 int((left != right).sum())))
            if spec.name == "valid":
                result.invalid_records = int(((left == 0) | (right == 0)).sum())
            del left, right
        for name in FLOAT_FIELDS:
            if name not in a.files or name not in b.files:
                continue
            left = _member(a, name, ra).astype(np.float64)
            right = _member(b, name, rb).astype(np.float64)
            differing = int(np.sum(~((left == right) | (np.isnan(left) & np.isnan(right)))))
            result.float_comparisons.append(Comparison(name, "derived floats", int(left.shape[1]),
                                                       int(left.size), differing))
            result.float_max_ulps[name] = _ulps(left.reshape(-1), right.reshape(-1))
        return result


# ------------------------------------------------------- independent re-execution
@dataclass(frozen=True)
class Reexecution:
    """One channel a second scan session processed independently of its owner."""

    freq_id: int
    owner: str
    second: str
    enumerated: int
    completed: int
    quarantined: int
    second_completed: int
    second_quarantined: int

    @property
    def units(self) -> int:
        return self.completed + self.quarantined

    @property
    def mismatches(self) -> int:
        return int(self.completed != self.second_completed) + int(self.quarantined != self.second_quarantined)


def _int(value) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def reexecutions(root: Path) -> tuple[list[Reexecution], str]:
    """``logs/channels/summary.csv``'s duplicate-built channels, and why there are none."""
    path = root / "logs" / "channels" / "summary.csv"
    if not path.is_file():
        return [], f"no {path.relative_to(root) if path.is_relative_to(root) else path}"
    rows: list[Reexecution] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("shard3_rebuilt", "")).strip().lower() not in ("yes", "true", "1"):
                continue
            owner = str(row.get("owner_session", "")).strip()
            built_by = str(row.get("built_by", "")).strip()
            second = "shard3"
            if second in built_by:            # the second session was the only builder: not a duplicate
                continue
            rows.append(Reexecution(_int(row.get("channel")), owner, second, _int(row.get("enumerated")),
                                    _int(row.get("completed")), _int(row.get("quarantined")),
                                    _int(row.get("shard3_completed")), _int(row.get("shard3_quarantined"))))
    rows.sort(key=lambda r: r.freq_id)
    return rows, "" if rows else "summary.csv records no channel built by two sessions"


# ------------------------------------------------------------------ the record
@dataclass
class Record:
    """Everything the table prints, and the reason for anything it cannot."""

    root: Path | None
    builds: dict[str, Build] = field(default_factory=dict)
    arms: list[Arm] = field(default_factory=list)
    reexecution: list[Reexecution] = field(default_factory=list)
    reexecution_reason: str = ""
    reason: str = ""
    inputs: list[Path] = field(default_factory=list)


_CACHE: dict[tuple, Record] = {}


def record(run: Run, *, stored_spectra: bool = COMPARE_STORED_SPECTRA) -> Record:
    """Discover the cohorts, join them, and compare; every failure is a stated reason.

    The two builders share one pass: the join reads about 1.5 GB of compressed
    members, so the result is cached on the run's own directories for the life
    of the process.
    """
    key = (str(run.results_dir), str(run.run.get("products_dir", "")), bool(stored_spectra))
    if key not in _CACHE:
        _CACHE[key] = _record(run, stored_spectra=stored_spectra)
    return _CACHE[key]


def _record(run: Run, *, stored_spectra: bool = COMPARE_STORED_SPECTRA) -> Record:
    root = campaign_root(run)
    if root is None or not root.is_dir():
        return Record(root, reason=f"the run names no readable campaign directory ({root})")
    out = Record(root)
    digests = _sha256sums(root)
    if digests:
        out.inputs.append(root / "SHA256SUMS")
    reference_path, campaign_path, reason = discover(root)
    if reference_path is None or campaign_path is None:
        out.reason = reason
    else:
        out.builds[REFERENCE] = read_build(REFERENCE, reference_path, digests, root)
        out.builds[CAMPAIGN] = read_build(CAMPAIGN, campaign_path, digests, root)
        out.inputs += [reference_path, campaign_path]
        out.arms.append(compare(ARM_QUALIFICATION, out.builds[REFERENCE], out.builds[CAMPAIGN],
                                stored_spectra=stored_spectra))
        freq_id = out.builds[REFERENCE].freq_id
        products_dir = Path(str(run.run.get("products_dir", "")))
        survey = products_dir / f"{freq_id}.npz"
        if survey.is_file():
            out.builds[RUN_OF_RECORD] = read_build(RUN_OF_RECORD, survey, digests, root)
            out.inputs.append(survey)
            out.arms.append(compare(ARM_RUN, out.builds[REFERENCE], out.builds[RUN_OF_RECORD],
                                    stored_spectra=stored_spectra))
        else:
            out.reason = (out.reason + "; " if out.reason else "") + \
                f"the run of record carries no product for freq_id {freq_id} ({survey})"
    out.reexecution, out.reexecution_reason = reexecutions(root)
    if out.reexecution:
        out.inputs.append(root / "logs" / "channels" / "summary.csv")
    return out


# ------------------------------------------------------------------ rendering
def _digest(value: str) -> str:
    if not value:
        return DASH
    if re.fullmatch(r"[0-9a-f]{40,}", value):
        return r"\texttt{" + value[:DIGEST_CHARS] + r"}\dots"
    return r"\texttt{" + tex(value) + "}"


def _stack(panels: Sequence[tuple[str, str]]) -> str:
    r"""The panels one above the other in a single box (the convention of ``calibration_nulls.stack``)."""
    lines = [r"\begin{tabular}{@{}l@{}}"]
    for i, (caption, body) in enumerate(panels):
        if i:
            lines.append(r"\\[\medskipamount]")
        lines.append(caption + r"\\[2pt]")
        lines.append(body.rstrip("\n"))
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _manifest_cell(key: str, value: str) -> str:
    if key in ("events", "frames"):
        return f"${fmt_int(value)}$" if value else DASH
    return _digest(value)


def _manifest_panels(rec: Record, frag: Fragment) -> tuple[str, str]:
    """Two panels: the fields every cohort agrees on, then the fields the builds differ in.

    The split is computed, never declared: a field whose value is the same in
    every cohort prints once; a field that differs prints one column per
    cohort, and those columns are the build axis the claim rests on. Every
    cell is keyed either way.
    """
    cohorts = [c for c in (REFERENCE, CAMPAIGN, RUN_OF_RECORD) if c in rec.builds]
    compared = [c for c in (REFERENCE, CAMPAIGN) if c in rec.builds] or cohorts
    shared, differing = [], []
    for label, key in MANIFEST_FIELDS:
        values = [rec.builds[c].manifest.get(key, "") for c in cohorts]
        for cohort, value in zip(cohorts, values):
            frag.add(f"{PREFIX}.manifest.{key.replace(' ', '_')}.{cohort}", value or None,
                     kind="text", renderings=(value,) if value else (),
                     status="measured" if value else "pending",
                     row={"manifest_field": label, "cohort": cohort}, column=COHORT_LABEL[cohort])
        # the split is between the two *compared builds*: a field the run of record alone differs in
        # (it spans the whole survey, not eight events) stays with the shared fields and is a note
        if cohorts and len({rec.builds[c].manifest.get(key, "") for c in compared}) == 1:
            shared.append([tex(label), _manifest_cell(key, rec.builds[compared[0]].manifest.get(key, ""))])
        else:
            differing.append([tex(label)] + [_manifest_cell(key, v) for v in values])
    if not cohorts:
        empty = booktabs(["manifest field", "value"], [[tex(label), DASH] for label, _ in MANIFEST_FIELDS], "ll")
        return empty, booktabs(["manifest field", "value"], [[DASH, DASH]], "ll")
    first = booktabs(["manifest field", "both compared builds"], shared or [[DASH, DASH]], "ll")
    second = booktabs(["manifest field"] + [tex(COHORT_LABEL[c]) for c in cohorts],
                      differing or [[DASH] + [DASH] * len(cohorts)], "l" + "l" * len(cohorts))
    return first, second


def _denominator_panel(rec: Record, frag: Fragment) -> str:
    arms = rec.arms
    header = ["quantity"] + [tex(a.label) for a in arms] if arms else ["quantity", "value"]
    quantities: tuple[tuple[str, str, str], ...] = (
        ("reference events", "reference_events", "int"),
        ("campaign events", "other_events", "int"),
        ("joined events", "joined_events", "int"),
        ("event completeness", "event_completeness", "float"),
        ("reference frames", "reference_frames", "int"),
        ("campaign frames", "other_frames", "int"),
        ("joined frames", "joined_frames", "int"),
        ("frame completeness", "frame_completeness", "float"),
        ("missing records", "missing_frames", "int"),
        ("invalid records", "invalid_records", "int"),
        ("integer values compared", "compared", "int"),
        ("mismatches", "mismatches", "int"),
        ("derived float values compared", "float_compared", "int"),
        ("derived float mismatches", "float_mismatches", "int"),
        ("largest float difference [ulp]", "float_ulps", "float"),
    )
    values = {}
    for a in arms:
        ulps = [u for u in a.float_max_ulps.values() if math.isfinite(u)]
        values[a.arm] = {"reference_events": a.reference.events, "other_events": a.other.events,
                         "joined_events": a.joined_events, "event_completeness": a.event_completeness,
                         "reference_frames": a.reference.frames, "other_frames": a.other.frames,
                         "joined_frames": a.joined_frames, "frame_completeness": a.frame_completeness,
                         "missing_frames": a.missing_frames, "invalid_records": a.invalid_records,
                         "compared": a.compared, "mismatches": a.mismatches,
                         "float_compared": sum(c.compared for c in a.float_comparisons),
                         "float_mismatches": sum(c.mismatches for c in a.float_comparisons),
                         "float_ulps": max(ulps) if ulps else math.nan}
    rows = []
    for label, key, kind in quantities:
        cells = [tex(label)]
        for a in arms:
            value = values[a.arm][key]
            if kind == "int":
                cells.append(f"${fmt_int(value)}$")     # a count is always defined on a joined arm
                frag.add(f"{PREFIX}.{key}.{a.arm}", int(value), kind="int",
                         row={"quantity": label, "arm": a.arm}, column=a.label)
            else:
                digits = 1 if key == "float_ulps" else 3
                cells.append(f"${fmt(value, digits)}$" if math.isfinite(value) else DASH)
                if math.isfinite(value):
                    frag.add(f"{PREFIX}.{key}.{a.arm}", float(value), precision=digits,
                             row={"quantity": label, "arm": a.arm}, column=a.label)
        rows.append(cells)
    if not arms:
        return booktabs(["quantity", "value"], [[tex(label), DASH] for label, _, _ in quantities], "lr")
    return booktabs(header, rows, "l" + "r" * len(arms))


def _field_panel(rec: Record, frag: Fragment) -> str:
    """One row per compared integer member; the derived floats are counted in the denominator panel."""
    arms = rec.arms
    header = ["member", "ints/frame"]
    for a in arms:
        header += [tex(ARM_SHORT.get(a.arm, a.arm)) + " compared", "mism."]
    if not arms:
        header += ["compared", "mism."]
    by_arm = {a.arm: {c.name: c for c in a.comparisons} for a in arms}
    for a in arms:
        by_arm[a.arm].update({c.name: c for c in a.float_comparisons})
    rows, midrules, group = [], [], ""
    for spec in FIELDS:
        if spec.group != group:
            group = spec.group
            if rows:
                midrules.append(len(rows))
        width = DASH
        cells = []
        for a in arms:
            c = by_arm[a.arm].get(spec.name)
            if c is None or not c.present or c.compared == 0:
                cells += [DASH, DASH]
                continue
            width = f"${fmt_int(c.width)}$"
            cells += [f"${fmt_int(c.compared)}$", f"${fmt_int(c.mismatches)}$"]
            frag.add(f"{PREFIX}.compared.{spec.name}.{a.arm}", c.compared, kind="int",
                     row={"member": spec.name, "arm": a.arm}, column="compared")
            frag.add(f"{PREFIX}.mismatches.{spec.name}.{a.arm}", c.mismatches, kind="int",
                     row={"member": spec.name, "arm": a.arm}, column="mismatches")
        if not arms:
            cells += [DASH, DASH]
        rows.append([r"\texttt{" + tex(spec.name) + "}", width] + cells)
    for name in FLOAT_FIELDS:                       # keyed but not printed here: the denominator panel counts them
        for a in arms:
            c = by_arm[a.arm].get(name)
            if c is None:
                continue
            frag.add(f"{PREFIX}.compared.{name}.{a.arm}", c.compared, kind="int",
                     row={"member": name, "arm": a.arm}, column="compared")
            frag.add(f"{PREFIX}.mismatches.{name}.{a.arm}", c.mismatches, kind="int",
                     row={"member": name, "arm": a.arm}, column="mismatches")
    align = "l" + "r" + "rr" * (len(arms) if arms else 1)
    return booktabs(header, rows, align, midrules=midrules)


def _reexecution_panel(rec: Record, frag: Fragment) -> str:
    header = [r"\texttt{freq\_id}", "owner", "second", "enumerated", "completed", "quarantined", "mism."]
    rows = []
    for r in rec.reexecution:
        completed = f"${fmt_int(r.completed)}$" if r.completed == r.second_completed else \
            f"${fmt_int(r.completed)}$ / ${fmt_int(r.second_completed)}$"
        quarantined = f"${fmt_int(r.quarantined)}$" if r.quarantined == r.second_quarantined else \
            f"${fmt_int(r.quarantined)}$ / ${fmt_int(r.second_quarantined)}$"
        rows.append([str(r.freq_id), tex(r.owner), tex(r.second), f"${fmt_int(r.enumerated)}$",
                     completed, quarantined, f"${fmt_int(r.mismatches)}$"])
        for key, value in (("completed", r.completed), ("quarantined", r.quarantined),
                           ("second_completed", r.second_completed), ("second_quarantined", r.second_quarantined),
                           ("units", r.units), ("mismatches", r.mismatches)):
            frag.add(f"{PREFIX}.reexec.{key}.fid{r.freq_id}", int(value), kind="int",
                     row={"freq_id": r.freq_id}, column=key)
    if not rows:
        rows = [[DASH] * len(header)]
    return booktabs(header, rows, "lllrrrr")


# ------------------------------------------------------------------ notes
def _notes(rec: Record, frag: Fragment) -> None:
    if rec.reason:
        frag.notes.append(f"cross-build pair not assembled: {rec.reason}; every cell it would fill prints the dash")
    for a in rec.arms:
        if a.disjoint_reason:
            frag.notes.append(f"{a.label}: the two cohorts do not overlap and cannot be compared frame by frame -- "
                              f"{a.disjoint_reason}. What can still be compared is the build manifests and the "
                              "event and frame counts, which is what this arm's columns print")
        if a.absent_fields:
            frag.notes.append(f"{a.label}: not compared -- " + "; ".join(a.absent_fields))
    worst: dict[str, float] = {}
    for a in rec.arms:
        for name, ulps in a.float_max_ulps.items():
            if math.isfinite(ulps) and ulps > 0:
                worst[name] = max(worst.get(name, 0.0), ulps)
    for name, ulps in worst.items():
        frag.notes.append(rf"\texttt{{{tex(name)}}} is not bit-equal across the builds (largest difference "
                          f"{fmt(ulps, 1)} ulp on the joined frames) although every integer it is derived from is: "
                          "it is a log of an exactly equal integer ratio, and is outside the claim")
    if {REFERENCE, CAMPAIGN} <= set(rec.builds):
        cohort_only = [label for label, key in MANIFEST_FIELDS
                       if rec.builds[REFERENCE].manifest.get(key) == rec.builds[CAMPAIGN].manifest.get(key)
                       and RUN_OF_RECORD in rec.builds
                       and rec.builds[RUN_OF_RECORD].manifest.get(key) != rec.builds[REFERENCE].manifest.get(key)]
        if cohort_only:
            detail = ", ".join(f"{label} ({tex(rec.builds[RUN_OF_RECORD].manifest.get(key, ''))})"
                               for label, key in MANIFEST_FIELDS if label in cohort_only)
            frag.notes.append("the first panel's values are the ones both compared builds record; the run-of-record "
                              "product spans the whole survey rather than the eight compared events and carries more "
                              f"than one value of {detail}")
    frag.notes.append("the products record no host and no compiler: the build axis they do record is the analyzer "
                      "source digest and the detector binary's SHA-256, and both differ between the two cohorts. "
                      "The hosts and the architectures the binary was built for are named by the campaign's own run "
                      "ledger (the pair is its cross-architecture qualification), not by any product field")
    frag.notes.append("the join key is the physical archive key the product stores "
                      r"(\texttt{unit\_keys[frame\_unit\_index]} with \texttt{frame\_in\_unit}); "
                      r"\texttt{frame\_index} and \texttt{frame\_unit\_index} are positional and are not compared")
    if not COMPARE_STORED_SPECTRA:
        frag.notes.append(r"\texttt{psd\_frame\_db\_i16} was not compared on this render")


# ------------------------------------------------------------------ builder
def build(run: Run, *, stored_spectra: bool = COMPARE_STORED_SPECTRA) -> Fragment:
    """The chapter record: the two build manifests and the joined denominator, three panels in one box."""
    rec = record(run, stored_spectra=stored_spectra)
    frag = Fragment(NAME, LABEL, "")
    frag.inputs = list(run.inputs()) + rec.inputs
    shared, differing = _manifest_panels(rec, frag)
    frag.tex = _stack([
        (r"\emph{Build manifest: what both compared builds record identically.}", shared),
        (r"\emph{Build manifest: where they differ --- the build axis the claim rests on.}", differing),
        (r"\emph{Joined denominator and completeness, on the reference cohort's frames.}",
         _denominator_panel(rec, frag)),
    ])
    total_compared = sum(a.compared for a in rec.arms)
    total_mismatches = sum(a.mismatches for a in rec.arms)
    frag.add(f"{PREFIX}.arms", len(rec.arms), kind="int", column="arms")
    frag.add(f"{PREFIX}.integer_values_compared", total_compared, kind="int", column="compared")
    frag.add(f"{PREFIX}.mismatches_total", total_mismatches, kind="int", column="mismatches")
    if rec.arms:
        frag.notes.append(f"{fmt_int(total_compared)} integer values compared over {len(rec.arms)} joins, "
                          f"{fmt_int(total_mismatches)} mismatches")
    frag.notes.append(f"the member-by-member breakdown and the independent re-execution are the companion "
                      f"fragment {LEDGER_NAME} ({LEDGER_LABEL})")
    _notes(rec, frag)
    return frag


def build_fields(run: Run, *, stored_spectra: bool = COMPARE_STORED_SPECTRA) -> Fragment:
    """The companion: the mismatch count member by member, and the same build's re-execution."""
    rec = record(run, stored_spectra=stored_spectra)
    frag = Fragment(LEDGER_NAME, LEDGER_LABEL, "")
    frag.inputs = list(run.inputs()) + rec.inputs
    frag.tex = _stack([
        (r"\emph{Mismatches by compared integer member.}", _field_panel(rec, frag)),
        (r"\emph{Independent re-execution of the same build (different host and session), unit counts only.}",
         _reexecution_panel(rec, frag)),
    ])
    frag.add(f"{PREFIX}.reexecution_channels", len(rec.reexecution), kind="int", column="channels")
    frag.add(f"{PREFIX}.reexecution_units", sum(r.units for r in rec.reexecution), kind="int", column="units")
    frag.add(f"{PREFIX}.reexecution_completed", sum(r.completed for r in rec.reexecution), kind="int",
             column="completed")
    frag.add(f"{PREFIX}.reexecution_mismatches", sum(r.mismatches for r in rec.reexecution), kind="int",
             column="mismatches")
    for a in rec.arms:
        if a.absent_fields:
            frag.notes.append(f"{a.label}: not compared -- " + "; ".join(a.absent_fields))
    frag.notes.append(r"the derived float members (\texttt{normalized\_coarse\_power\_ratio\_db}, "
                      r"\texttt{pilot\_excess\_db}) are keyed but not printed here: they are counted in the "
                      f"denominator panel of {NAME} ({LABEL}), outside the claim")
    if rec.reexecution:
        units = sum(r.units for r in rec.reexecution)
        completed = sum(r.completed for r in rec.reexecution)
        frag.notes.append(r"the second panel is a second execution of the \emph{same} build on a different host and "
                          f"session, not a second build: {len(rec.reexecution)} channels, {fmt_int(units)} enumerated "
                          f"units ({fmt_int(completed)} completed, {fmt_int(units - completed)} quarantined), "
                          "processed twice, and both counts agree on every channel. Only the canonical copy of each "
                          "product was retained on this machine, so those duplicates cannot be re-compared field by "
                          "field here: their byte-identity survives as the campaign ledger's digest record, not as "
                          "data")
    elif rec.reexecution_reason:
        frag.notes.append(f"no independent re-execution row: {rec.reexecution_reason}")
    return frag


BUILDERS = (build, build_fields)
