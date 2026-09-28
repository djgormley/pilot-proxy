"""The report of one archive run: every table fragment and figure, one manifest.

``build_report(results_dir, out_dir)`` loads the ledger, renders every
registered table builder into ``out_dir/tables`` with its ``numbers.json``,
renders the figures into ``out_dir/figures`` (each figure module exposes
``render(run, out_dir) -> [paths]`` and, where it prints numbers, a
``build(run) -> Fragment``), and writes ``export_manifest.json``. A module
registers its builders as ``BUILDERS`` (a tuple of ``build(run)`` callables)
or a single ``build``. Modules are imported lazily so a missing optional
dependency (matplotlib for the figures) disables only what needs it.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Sequence

from .core import Fragment, Run, load_run, write_report

TABLE_MODULES: tuple[str, ...] = ("accounting", "crossbuild", "detection")
FIGURE_MODULES: tuple[str, ...] = ("detection",)


def _module(name: str):
    return importlib.import_module(f"{__package__}.{name}")


def table_builders(modules: Sequence[str] = TABLE_MODULES) -> list[Callable[[Run], Fragment]]:
    """Every table builder, in the registry's order."""
    out: list[Callable[[Run], Fragment]] = []
    for name in modules:
        mod = _module(name)
        builders = getattr(mod, "BUILDERS", None) or (mod.build,)
        out.extend(builders)
    return out


def build_report(results_dir: Path | str, out_dir: Path | str | None = None, *, commit: str | None = None,
                 generated: str | None = None, figures: bool = True, modules: Sequence[str] | None = None,
                 figure_modules: Sequence[str] | None = None, require_tex: bool = True) -> dict:
    """Render tables, numbers and figures for one run; return the manifest.

    ``require_tex`` is the document's typography contract: the figures are set
    in Latin Modern through LaTeX, which the dissertation's figure audit
    enforces. Pass ``False`` only for a preview on a machine without a TeX
    installation, and never for a report that will be vendored.
    """
    import datetime as dt

    # the registries are read here rather than bound as default arguments: a default is
    # evaluated once at definition, so a caller (or a test) that replaces TABLE_MODULES or
    # FIGURE_MODULES on the module would have been ignored and the full registry rendered
    modules = TABLE_MODULES if modules is None else modules
    figure_modules = FIGURE_MODULES if figure_modules is None else figure_modules
    run = load_run(results_dir)
    out = Path(out_dir) if out_dir is not None else run.results_dir / "dissertation"
    from .numbers import git_commit
    commit = commit or git_commit(Path(__file__).resolve().parents[4])
    generated = generated or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    builders = table_builders(modules)
    extra: list[Path] = []
    if figures:
        # the document's own style, Latin Modern through LaTeX: the figure audit refuses a
        # substituted font, so a report rendered without it cannot be vendored
        from pilot_proxy import figure_style as style

        style.configure(require_tex=require_tex)
        fig_dir = out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for name in figure_modules:
            mod = _module(name)
            extra.extend(Path(p) for p in mod.render(run, fig_dir))
            # a module registered in both registries (a table beside its figure) is already built
            if hasattr(mod, "build") and name not in modules:
                builders.append(mod.build)
    return write_report(run, out, builders, commit=commit, generated=generated, extra_artifacts=extra)


def main(argv=None) -> int:
    """``pilot-proxy characterize report``: the thesis's detector tables of one archive run."""
    import argparse
    import json
    import os

    from . import accounting, detection

    parser = argparse.ArgumentParser(prog="pilot-proxy characterize report", description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the archive run (its ledger/ is read)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modules", nargs="+", default=list(TABLE_MODULES), choices=list(TABLE_MODULES))
    parser.add_argument("--figures", nargs="*", default=list(FIGURE_MODULES), choices=list(FIGURE_MODULES))
    parser.add_argument("--require-tex", action="store_true",
                        help="set the figures through LaTeX (required for anything vendored)")
    parser.add_argument("--inventory", type=Path, default=None,
                        help=f"the inventory export the accounting table counts (default ${accounting.FREEZE_ENV})")
    parser.add_argument("--sweep", type=Path, default=None,
                        help=f"the estimator-transfer sweep the detection table reads (default ${detection.SWEEP_ENV})")
    parser.add_argument("--ota", type=Path, default=None,
                        help=f"the over-the-air transfer capture (default ${detection.OTA_ENV})")
    args = parser.parse_args(argv)
    for value, name in ((args.inventory, accounting.FREEZE_ENV), (args.sweep, detection.SWEEP_ENV),
                        (args.ota, detection.OTA_ENV)):
        if value is not None:
            os.environ[name] = str(value)
    manifest = build_report(args.results, args.out, modules=tuple(args.modules), figure_modules=tuple(args.figures),
                            figures=bool(args.figures), require_tex=args.require_tex)
    print(json.dumps({"artifacts": [a["name"] for a in manifest["artifacts"]], "out": str(args.out)}))
    return 0
