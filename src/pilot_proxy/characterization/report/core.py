"""Ledger access, rendering conventions and the report writer."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from . import numbers as nb

# Frozen on-disk values (the dissertation's import_archive_report.py enforces them): the report
# schema token and the repository name the vendored export manifest carries.
SCHEMA = {"name": "rfisher-archive-report", "version": 1}
REPOSITORY = "WVURAIL/RFIsher"
DASH = "--"                      # an absent or undefined value in a table cell


# ------------------------------------------------------------------ the ledger
@dataclass(frozen=True)
class Channel:
    """One channel's ledger record: the per-module sections by prefix."""

    channel: int
    freq_id: int
    product: str
    product_sha256: str
    notes: tuple[str, ...]
    sections: Mapping[str, Mapping | None]
    path: Path

    def section(self, name: str) -> Mapping:
        """A section's row, or an empty mapping when the module recorded it as absent."""
        row = self.sections.get(name)
        return row if row is not None else {}

    def has(self, name: str) -> bool:
        return self.sections.get(name) is not None

    def get(self, name: str, key: str, default=None):
        return self.section(name).get(key, default)

    @property
    def era(self) -> Mapping:
        return self.section("era")

    @property
    def selection(self) -> Mapping:
        return self.section("selection")

    @property
    def screening(self) -> Mapping:
        return self.section("screening")

    @property
    def null(self) -> Mapping:
        return self.section("null")

    @property
    def chain(self) -> Mapping:
        return self.section("chain")

    @property
    def tolerance(self) -> Mapping:
        return self.section("tolerance")

    @property
    def anchor(self) -> Mapping:
        return self.section("anchor")

    @property
    def containment(self) -> Mapping:
        return self.section("containment")

    @property
    def blocks(self) -> Mapping:
        return self.section("blocks")


@dataclass(frozen=True)
class Run:
    """One archive run: ``ledger/run.json`` and every channel record."""

    results_dir: Path
    run: Mapping
    channels: tuple[Channel, ...]

    @property
    def commit(self) -> str:
        return str(self.run.get("producer", {}).get("commit", "")) if isinstance(self.run.get("producer"), Mapping) else ""

    @property
    def generated(self) -> str:
        return str(self.run.get("generated", ""))

    def by_channel(self) -> dict[int, Channel]:
        return {c.channel: c for c in self.channels}

    def inputs(self) -> list[Path]:
        return [self.results_dir / "ledger" / "run.json"] + [c.path for c in self.channels]


def load_run(results_dir: Path | str) -> Run:
    results_dir = Path(results_dir)
    run_path = results_dir / "ledger" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    channels = []
    for rel in run.get("channels_files", run.get("channels", [])):
        if not isinstance(rel, str):
            continue
        path = results_dir / "ledger" / rel
        d = json.loads(path.read_text(encoding="utf-8"))
        channels.append(Channel(int(d["channel"]), int(d["freq_id"]), str(d["product"]), str(d["product_sha256"]),
                               tuple(d.get("notes", ())), d["sections"], path))
    if not channels:      # run.json lists channel numbers: read the directory
        for path in sorted((results_dir / "ledger" / "channels").glob("ch*_fid*.json")):
            d = json.loads(path.read_text(encoding="utf-8"))
            channels.append(Channel(int(d["channel"]), int(d["freq_id"]), str(d["product"]), str(d["product_sha256"]),
                                    tuple(d.get("notes", ())), d["sections"], path))
    channels.sort(key=lambda c: c.channel)
    return Run(results_dir, run, tuple(channels))


# ------------------------------------------------------------------ rendering
_TEX_SPECIALS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(text) -> str:
    """Escape a plain-text cell for LaTeX."""
    return "".join(_TEX_SPECIALS.get(ch, ch) for ch in str(text))


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value, digits: int = 3, *, sig: bool = False, plus: bool = False, dash: str = DASH) -> str:
    """A number to ``digits`` decimals (``sig``: significant figures); NaN and None print as ``dash``.

    Math mode is the caller's: the string carries no ``$``. Negative signs use
    the ASCII hyphen, which the chapters' math mode renders as a minus.
    """
    if not _finite(value):
        return dash
    x = float(value)
    if sig:
        if x == 0.0:
            text = "0"
        else:
            mag = int(math.floor(math.log10(abs(x))))
            decimals = max(digits - 1 - mag, 0)
            text = f"{x:.{decimals}f}"
    else:
        text = f"{x:.{digits}f}"
    if plus and x > 0:
        text = "+" + text
    return text


def fmt_int(value, *, thousands: bool = True, dash: str = DASH) -> str:
    """An integer with ``{,}`` thousands groups (the chapters' convention inside math mode)."""
    if not _finite(value):
        return dash
    n = int(round(float(value)))
    text = f"{abs(n):,}".replace(",", "{,}") if thousands else str(abs(n))
    return ("-" if n < 0 else "") + text


def fmt_month(label) -> str:
    """A ``YYYY-MM`` month label as printed (unchanged; empty prints as the dash)."""
    return str(label) if label else DASH


def fmt_range(low, high, digits: int = 2, dash: str = DASH) -> str:
    if not (_finite(low) and _finite(high)):
        return dash
    return f"{fmt(low, digits)}--{fmt(high, digits)}"


def booktabs(header: Sequence[str], rows: Iterable[Sequence[str]], align: str, *, midrules: Iterable[int] = (),
             longtable: bool = False, caption: str | None = None, label: str | None = None) -> str:
    """A booktabs table: cells are final LaTeX (escape with :func:`tex` first).

    ``longtable=True`` emits a ``longtable`` instead of a ``tabular``, with the
    header repeated on every page. A table taller than the text block silently
    loses its last rows inside a float, so any fragment that can outgrow a page
    -- anything with a row per channel per world, say -- must ask for one. The
    caption and label then belong to the environment rather than to a float
    around it, so pass them here and do not wrap the fragment in a ``table``.
    """
    rows = [list(r) for r in rows]
    n = len(header)
    if len(align.replace("|", "")) != n:
        raise ValueError(f"alignment '{align}' does not match {n} columns")
    for r in rows:
        if len(r) != n:
            raise ValueError(f"row has {len(r)} cells, header {n}: {r}")
    breaks = set(midrules)
    head = " & ".join(header) + r" \\"
    if longtable:
        lines = [f"\\begin{{longtable}}{{{align}}}"]
        if caption is not None:
            lines.append(f"\\caption{{{caption}}}" + (f"\\label{{{label}}}" if label else "") + r" \\")
        lines += ["\\toprule", head, "\\midrule", "\\endfirsthead",
                  "\\toprule", head, "\\midrule", "\\endhead",
                  "\\midrule", f"\\multicolumn{{{n}}}{{r}}{{\\emph{{continued on the next page}}}} \\\\",
                  "\\endfoot", "\\bottomrule", "\\endlastfoot"]
    else:
        lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", head, "\\midrule"]
    for i, r in enumerate(rows):
        if i in breaks and i > 0:
            lines.append("\\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += ["\\end{longtable}"] if longtable else ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ fragments
@dataclass
class Fragment:
    """One table fragment with the numbers it prints and the inputs it read."""

    name: str                                   # file stem: 'calibration_eras' -> tables/calibration_eras.tex
    label: str                                  # the chapter's \\label, for the manifest
    tex: str
    numbers: list = field(default_factory=list) # archive.numbers.Number
    inputs: list = field(default_factory=list)  # Paths read
    notes: list = field(default_factory=list)   # what the table leaves out and why

    def add(self, key: str, value, *, precision: int | None = None, kind: str = "float", status: str = "measured",
            renderings: Sequence[str] = (), row: Mapping | None = None, column: str = "") -> None:
        source = {"table": f"{self.name}.tex", "row": dict(row or {}), "column": column}
        self.numbers.append(nb.Number(key, value, kind=kind, precision=precision, status=status,
                                      renderings=tuple(renderings), source=source))


Builder = Callable[[Run], Fragment]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_report(run: Run, out_dir: Path | str, builders: Sequence[Builder], *, commit: str, generated: str,
                 extra_artifacts: Sequence[Path] = ()) -> dict:
    """Render every builder, write the fragments, numbers and manifest; return the manifest."""
    from ..run import producer_identity
    render_producer = producer_identity()
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "numbers").mkdir(parents=True, exist_ok=True)
    artifacts = []
    for build in builders:
        frag = build(run)
        tex_path = out / "tables" / f"{frag.name}.tex"
        tex_path.write_text(frag.tex, encoding="utf-8")
        doc = nb.NumbersDocument.new(frag.name, repository=REPOSITORY, commit=commit,
                                     script=f"pilot_proxy.characterization.report.{frag.name}", generated=generated)
        doc.producer.update({"source_digest": render_producer["source_digest"],
                             "dirty": render_producer["dirty"]})
        for path in frag.inputs or run.inputs():
            doc.add_input(path)
        for number in frag.numbers:
            doc.add(number)
        num_path = doc.write(out / "numbers" / f"{frag.name}.numbers.json")
        artifacts.append({"name": frag.name, "label": frag.label, "table": f"tables/{tex_path.name}",
                          "numbers": f"numbers/{num_path.name}", "sha256": _sha256(tex_path),
                          "numbers_sha256": _sha256(num_path), "count": len(frag.numbers), "notes": list(frag.notes)})
    for path in extra_artifacts:
        artifacts.append({"name": Path(path).stem, "label": "", "table": str(Path(path).relative_to(out)) if Path(path).is_relative_to(out) else str(path),
                          "numbers": "", "sha256": _sha256(Path(path)), "numbers_sha256": "", "count": 0, "notes": []})
    producer = run.run.get("producer") if isinstance(run.run.get("producer"), Mapping) else {}
    manifest = {"schema": SCHEMA, "source": {"repository": REPOSITORY, "commit": commit},
                "render_producer": render_producer,
                # the run's own producer: a report is only as clean as the run it renders
                "producer": {k: producer.get(k) for k in ("commit", "dirty", "dirty_files", "source_digest",
                                                          "source_changed_during_run") if k in producer},
                "generated": generated,
                "run": {"results_dir": str(run.results_dir), "generated": run.generated,
                        "era_config_digest": run.run.get("era_config_digest", ""),
                        "products": run.run.get("products", {}), "channels": [c.channel for c in run.channels]},
                "artifacts": artifacts}
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
