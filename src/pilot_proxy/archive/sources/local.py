r"""
Local-directory data source.

For data already sitting on /arc (or anywhere on the filesystem) rather than in a
remote archive. `enumerate` globs files for the selection; `fetch` hardlinks (or
copies) the file into the engine's scratch dir -- so the engine's "delete after
analysis" only ever removes the scratch link, never your original.

Two real uses:
  * collaborators whose data is staged on /arc, not pulled from CADC;
  * the test/demo path, where a synthetic library stands in for an archive.

Selection (`ctx.selection`) uses the archive selection grammar:
freq_ids parse as exact integers from the filename (`--source-freq-id-regex`,
default `_(\d+)\.h5$`, so selecting 44 does not match `..._844.h5`), events
via `--source-event-regex` (default `baseband_(\d+)_`), and None / "all"
takes every file matching `--source-glob`.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import shutil
from typing import Iterable, List

from ..interfaces import READY, DataSource, PluginInfo, RunContext, Unit
from ..selection import parse_selection

# Default event capture, matching the canonical baseband naming
# baseband_<event>_<freq_id>.h5. Override with --source-event-regex for other
# layouts; a filename with no parseable event simply has no event to filter on.
_DEFAULT_EVENT_RE = r"baseband_(\d+)_"


def _file_snapshot(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"local input changed while hashing: {path}")
    return int(after.st_size), digest.hexdigest()


class LocalDirectorySource(DataSource):
    info = PluginInfo(
        name="local",
        kind="source",
        summary="Files already on disk (/arc or elsewhere); no archive access.",
        status=READY,
        instruments=("*",),
        requires=("a directory of input files",),
        notes="Stages by hardlink/copy into scratch, so your originals are never "
              "deleted. Set --source-root and optionally --source-glob.",
    )

    def enumerate(self, ctx: RunContext) -> Iterable[Unit]:
        o = ctx.options or {}
        root = o.get("source_root")
        if not root:
            raise SystemExit("local source: pass --source-root <dir>")
        root = os.path.abspath(os.path.expanduser(str(root)))
        pattern = o.get("source_glob", "*.h5")
        paths = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
        n_channels = getattr(ctx.instrument, "n_channels", None)
        sel = parse_selection(ctx.selection, n_channels=n_channels)
        # Parse the freq_id as an exact integer from the filename (so selecting 44
        # does not match baseband_..._844.h5). Override the capture for other
        # naming schemes with --source-freq-id-regex; likewise the event with
        # --source-event-regex.
        # `or` (not a .get default) so an explicit None/"" from a caller -- e.g.
        # `explore`, which always populates this key -- still falls back instead
        # of reaching re.compile(None).
        freq_id_re = re.compile(o.get("source_freq_id_regex") or r"_(\d+)\.h5$")
        event_re = re.compile(o.get("source_event_regex") or _DEFAULT_EVENT_RE)
        units: List[Unit] = []
        for p in paths:
            src_path = os.path.abspath(p)
            name = os.path.basename(p)
            match = freq_id_re.search(name)
            freq_id = int(match.group(1)) if match else None
            ev_match = event_re.search(name)
            event = ev_match.group(1) if ev_match else None
            if not sel.wants_freq_id(freq_id):
                continue
            if not sel.wants_event(event):
                continue
            size_bytes, sha256 = _file_snapshot(src_path)
            relative_path = os.path.relpath(src_path, root)
            meta: dict[str, object] = {
                "src_path": src_path,
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "scope": "local",
            }
            if freq_id is not None:
                # `explore` reads this metadata to summarize the available bands.
                meta["freq_id"] = freq_id
            if event is not None:
                # the companion-lookup key: an analyzer side-loading per-event
                # auxiliaries (gains, flags) keys them off meta["event"].
                meta["event"] = event
            units.append(Unit(key=src_path, name=name, meta=meta))
        return units

    def fetch(self, unit: Unit, dest: str) -> tuple[bool, str]:
        src = unit.meta.get("src_path", unit.key)
        try:
            if os.path.exists(dest):
                os.remove(dest)
            try:
                os.link(src, dest)                 # cheap: hardlink if same fs
            except OSError:
                shutil.copy2(src, dest)            # fallback: copy across fs
            expected_size = unit.meta.get("size_bytes")
            expected_sha256 = unit.meta.get("sha256")
            actual_size, actual_sha256 = _file_snapshot(dest)
            if actual_size != expected_size or actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"local input changed after enumeration: {src}"
                )
            return (os.path.getsize(dest) > 0), ""
        except Exception as exc:                   # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        root = (ctx.options or {}).get("source_root")
        if not root:
            return False, ["--source-root not set"]
        if not os.path.isdir(root):
            return False, [f"--source-root is not a directory: {root}"]
        return True, []
