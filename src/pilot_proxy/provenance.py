# coding=utf-8
"""Small provenance helpers for reproducible PilotProxy products."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path


def file_sha256(path: Path | str | None) -> str | None:
    """Return the SHA256 hex digest for an existing file, or None otherwise."""
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=8)
def package_source_sha256(package_root: Path | str | None = None) -> str:
    """Hash the installed Python implementation, independent of absolute paths.

    Development versions can span many commits while retaining the same package
    version.  This digest makes checkpoint compatibility depend on the actual
    implementation that produced the product rather than only ``__version__``.

    The digest is memoized per process: a long-running scan stamps every
    product with the tree as first observed, even if source files change on
    disk mid-run (for example, patches applied while a survey is processing).
    The imported code cannot change within a process, so the first observation
    is the closest available proxy for the implementation actually running; a
    relaunched process observes the new tree.
    """
    root = (
        Path(__file__).resolve().parent
        if package_root is None
        else Path(package_root).resolve()
    )
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def sha256_file(path: Path | str) -> str:
    """SHA256 hex digest of a file (raises if it does not exist)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def producer_identity(roots=None) -> dict:
    """The producing code of a run, recorded as information (a consumer never requires it).

    The repository commit and whether the package tree is dirty (scoped to
    ``src/pilot_proxy``), a digest of the package's own sources
    (:func:`package_source_sha256`, fixed at first use in the process), the
    package version, and the sha256 of the default project's detector
    register. ``roots`` may name further source roots whose digests are added.
    """
    import subprocess

    from pilot_proxy import __version__
    from pilot_proxy.config.project import default_project

    package = Path(__file__).resolve().parent
    commit, dirty, status = "", None, ""
    try:
        top = subprocess.run(["git", "-C", str(package), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=30)
        if top.returncode == 0:
            root = top.stdout.strip()
            commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                    timeout=30).stdout.strip()
            status = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no", "--",
                                     str(package)], capture_output=True, text=True, timeout=30).stdout.strip()
            dirty = bool(status)
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", None
    out = {"repository": "WVURAIL/pilot-proxy", "package": "pilot_proxy", "version": __version__,
           "commit": commit, "dirty": dirty, "dirty_files": status, "source_root": str(package),
           "source_digest": package_source_sha256(), "register_sha256": default_project().register.sha256()}
    if roots:
        out["other_source_digests"] = {name: package_source_sha256(Path(root)) for name, root in sorted(roots.items())}
    return out


def sidecar_manifest_path(path: Path | str | None) -> Path | None:
    """Return the conventional manifest sidecar path."""
    if path is None:
        return None
    return Path(f"{Path(path)}.manifest.json")


# -- detector_version token policy -----------------------------------------
#
# A detector_version string is
#
#     pilot-proxy/<version> source=<tree hash> kernel=<v> kernel_sha256=<hash>
#     <schema tag> K=<K>
#
# and its tokens fall into two classes. ``pilot-proxy/<version>`` and
# ``source=<tree hash>`` both name the BUILD that produced a product: the
# release label a maintainer typed, and the digest of the implementation that
# actually ran. Neither constrains the numbers. A version bump made for a
# release (0.3.0.dev0 -> 1.0.0) changes both tokens without touching detector
# math, and conversely a tree can move many commits while staying at one
# version. Thus the label is the weaker of the two, and the source digest is
# what genuinely identifies an implementation.
#
# The remaining tokens (kernel version, kernel binary hash, schema tag and K)
# are geometry: they are what cross-pilot stacking depends on, alongside the
# separately compared weights hashes, detector contract JSON, mask rule and
# reference placement. Resume is intentionally stricter and compares the full
# detector_version, including both build tokens, so one checkpoint can never
# append frames from two Python implementations.
#
# Both classes are recorded in full in every product, and combine reports the
# distinct build strings it stacked, so relaxing the gate loses no provenance.
DETECTOR_VERSION_BUILD_TOKEN_PREFIXES = ("pilot-proxy/", "source=")


def detector_version_geometry(version: object) -> tuple[str, ...]:
    """Return the geometry-bearing tokens of a ``detector_version`` string.

    Build-identity tokens (see ``DETECTOR_VERSION_BUILD_TOKEN_PREFIXES``) are
    dropped; everything else is preserved in order. Two complete per-pilot
    products whose geometries compare equal may be stacked together while
    retaining their distinct build provenance. This helper does not define the
    resume policy; resume requires an exact full-version match.
    """
    return tuple(
        token
        for token in str(version).split()
        if not token.startswith(DETECTOR_VERSION_BUILD_TOKEN_PREFIXES)
    )


def detector_version_build_id(version: object, digest_chars: int = 12) -> str:
    """Return a short human-readable build identity, ``<version>@<source>``.

    Used in the operator-facing notes printed when products from different
    builds are resumed into or stacked. Missing tokens render as ``?``.
    """
    label = "?"
    source = "?"
    for token in str(version).split():
        if token.startswith("pilot-proxy/"):
            label = token[len("pilot-proxy/"):] or "?"
        elif token.startswith("source="):
            source = token[len("source="):][:digest_chars] or "?"
    return f"{label}@{source}"


__all__ = [
    "DETECTOR_VERSION_BUILD_TOKEN_PREFIXES",
    "detector_version_build_id",
    "detector_version_geometry",
    "file_sha256",
    "package_source_sha256",
    "sidecar_manifest_path",
]
