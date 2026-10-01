"""Local, public provenance for survey execution and inventory snapshots."""
from __future__ import annotations

import functools
import importlib.metadata
import os
import re
import subprocess
from pathlib import Path
from subprocess import run as _run_git

from pilot_proxy import __version__
from pilot_proxy.provenance import file_sha256, package_source_sha256


@functools.lru_cache(maxsize=1)
def _software():
    root = Path(__file__).resolve().parents[3]
    revision = None
    if (root / ".git").exists():
        try:
            result = _run_git(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                text=True, timeout=5, check=False)
            value = result.stdout.strip()
            if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", value):
                revision = value
        except (OSError, subprocess.TimeoutExpired):
            pass
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {
        "source_revision": revision,
        "package_source_sha256": package_source_sha256(),
        "pilot_proxy_version": __version__,
        "datatrail_cli_version": version("datatrail-cli"),
        "cadcdata_version": version("cadcdata"),
    }


def runtime_provenance():
    """Identify software and the public certificate; never record private keys.

    Missing optional software/certificates remain explicitly unavailable. The
    package digest follows the existing first-observation-per-process contract.
    """
    result = dict(_software())
    result.update(certificate_not_after=None, certificate_sha256=None,
                  certificate_status="unavailable")
    certificate = Path(os.environ.get("CADC_CERT") or "~/.ssl/cadcproxy.pem").expanduser()
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        cert = x509.load_pem_x509_certificate(certificate.read_bytes())
        # Only the public X.509 certificate is fingerprinted, never its PEM
        # container (which can also contain a private proxy key).
        result.update(certificate_not_after=cert.not_valid_after_utc.isoformat(),
                      certificate_sha256=cert.fingerprint(hashes.SHA256()).hex(),
                      certificate_status="read")
    except (ImportError, OSError, ValueError, AttributeError):
        pass
    return result


def inventory_identity(path):
    path = Path(path)
    if not path.is_file():
        return {"inventory_sha256": None, "row_count": None}
    with path.open(encoding="utf-8") as stream:
        count = sum(bool(line.strip()) for line in stream)
    return {"inventory_sha256": file_sha256(path), "row_count": count}
