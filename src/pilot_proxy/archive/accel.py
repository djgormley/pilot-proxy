"""Resolve the CuPy build used by archive analyzers.

Policy: prefer the CuPy the CANFAR session image already ships. A pinned cupy in
this package would shadow or mismatch the image's CUDA module, so the GPU extra is
empty and this module does the right thing at run time instead:

  * `import_cupy()`        -> the image's cupy, or None if it isn't importable.
  * `get_array_module(g)`  -> numpy, or the image's cupy when g is true.
  * `detect_cuda_major()`  -> the image's CUDA major version, best-effort.
  * `ensure_cupy(install)` -> the image's cupy, optionally pip-installing the
                              matching `cupy-cuda<major>x` wheel when it is absent.

Auto-install lives only in `ensure_cupy(install=True)`. A scan never installs
packages on its own.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Optional

_CUDA_COMMAND_TIMEOUT_SECONDS = 30
_DEFAULT_CUDA_HOME = "/usr/local/cuda"


def import_cupy():
    """Return the environment's CuPy, or None only when it is not installed."""
    try:
        import cupy as cp  # noqa: F401
        return cp
    except ModuleNotFoundError as exc:
        if exc.name == "cupy":
            return None
        raise RuntimeError(
            "cupy is installed but could not import one of its dependencies "
            f"({exc}). Check that the CuPy wheel matches the installed CUDA "
            "toolkit; do not install a second CuPy package alongside it.") from exc
    except Exception as exc:
        raise RuntimeError(
            "cupy is installed but failed to initialize "
            f"({type(exc).__name__}: {exc}). Check that the CuPy wheel matches "
            "the installed CUDA toolkit.") from exc


def _cuda_major_from_text(text: str) -> Optional[int]:
    m = re.search(r"CUDA Version[:\s]+(\d+)\.", text)          # nvidia-smi header
    if m:
        return int(m.group(1))
    m = re.search(r"release\s+(\d+)\.", text)                  # nvcc --version
    if m:
        return int(m.group(1))
    return None


def _cuda_major_from_file(path: str) -> Optional[int]:
    """Read NVIDIA's JSON or text version-file formats."""
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None
    if path.endswith(".json"):
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None
        cuda = payload.get("cuda") if isinstance(payload, dict) else None
        if isinstance(cuda, dict):
            version = cuda.get("version")
        elif isinstance(cuda, str):
            version = cuda
        else:
            version = payload.get("version") if isinstance(payload, dict) else None
        match = re.match(r"\s*(\d+)(?:\.|\s|$)", str(version or ""))
        return int(match.group(1)) if match else None
    return _cuda_major_from_text(text)


def _cuda_major_from_command(command: list[str]) -> Optional[int]:
    try:
        out = subprocess.run(
            command, capture_output=True, text=True,
            timeout=_CUDA_COMMAND_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    return _cuda_major_from_text((out.stdout or "") + (out.stderr or ""))


def detect_cuda_major() -> Optional[int]:
    """Best-effort CUDA major version of the running image.

    Prefer the installed toolkit (active nvcc, then its version file) over
    ``nvidia-smi``, whose header is the driver's maximum supported CUDA version
    and may not identify the toolkit a CuPy wheel must match.
    """
    configured_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    cuda_home = configured_home or _DEFAULT_CUDA_HOME
    if configured_home:
        major = _cuda_major_from_command(
            [os.path.join(cuda_home, "bin", "nvcc"), "--version"])
        if major:
            return major
        for fname in ("version.json", "version.txt"):
            major = _cuda_major_from_file(os.path.join(cuda_home, fname))
            if major:
                return major

    major = _cuda_major_from_command(["nvcc", "--version"])
    if major:
        return major

    if not configured_home:
        for fname in ("version.json", "version.txt"):
            major = _cuda_major_from_file(os.path.join(cuda_home, fname))
            if major:
                return major

    # Last resort only: this is driver capability, not proof of an installed
    # toolkit, but it still gives setup-cupy a useful best-effort wheel choice.
    return _cuda_major_from_command(["nvidia-smi"])


def cupy_package(major: int) -> str:
    """The pip wheel name for a given CUDA major version, e.g. 12 -> cupy-cuda12x."""
    return f"cupy-cuda{int(major)}x"


def ensure_cupy(install: bool = False, quiet: bool = False):
    """Return the image's cupy, optionally installing the matching wheel.

    If cupy is already importable (the common CANFAR case), it is returned as-is.
    Otherwise, when install=True, the image's CUDA major version is detected and
    `cupy-cuda<major>x` is pip-installed into the active environment, then imported.
    Raises RuntimeError with an actionable message if it cannot be resolved.
    """
    cp = import_cupy()
    if cp is not None:
        return cp

    if not install:
        raise RuntimeError(
            "cupy is not importable in this environment. Run setup_env.sh "
            "or install the CuPy wheel matching this image's CUDA version.")

    major = detect_cuda_major()
    if major is None:
        raise RuntimeError(
            "cupy is missing and the CUDA version could not be detected "
            "(no nvidia-smi/nvcc and no CUDA version file). Install the cupy build "
            "matching your session image manually, e.g. `pip install cupy-cuda12x`.")

    pkg = cupy_package(major)
    if not quiet:
        print(f"[gpu] no cupy found; detected CUDA {major}.x -> installing {pkg}",
              flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--break-system-packages", pkg])

    cp = import_cupy()
    if cp is None:
        raise RuntimeError(
            f"installed {pkg} but cupy is still not importable; the wheel may not "
            f"match this image's CUDA module. Check `nvidia-smi` and install by hand.")
    return cp


def get_array_module(use_gpu: bool):
    """numpy, or the image's cupy when use_gpu is true.

    A scan path calls this; a missing cupy raises a clear setup error.
    """
    import numpy as np
    if not use_gpu:
        return np
    try:
        cp = import_cupy()
    except RuntimeError as exc:
        raise SystemExit(f"--gpu was requested but {exc}") from exc
    if cp is None:
        raise SystemExit(
            "--gpu was requested but cupy is not importable. Run setup_env.sh "
            "first, or drop --gpu to run on CPU.")
    return cp
