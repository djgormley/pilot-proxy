# coding=utf-8
"""GPU availability checks for the kernel tests (on the test path through pyproject.toml)."""

from __future__ import annotations

import ctypes

try:
    import cupy as cp

    _CUPY_IMPORT_ERROR: Exception | None = None
except Exception as cupy_import_error:
    cp = None  # type: ignore[assignment]
    _CUPY_IMPORT_ERROR = cupy_import_error

NVIDIA_SMI_TIMEOUT_SECONDS = 5
CUDA_DRIVER_SUCCESS = 0


def _cupy_unavailable_reason() -> str:
    if _CUPY_IMPORT_ERROR is None:
        return "CuPy import unavailable"
    return f"CuPy import failed: {_CUPY_IMPORT_ERROR}"


def cuda_available() -> tuple[bool, str]:
    """Return availability plus a reason string when no GPU is found."""
    if cp is None:
        return False, _cupy_unavailable_reason()
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as cuda_runtime_error:
        import logging

        logging.debug(f"CUDA availability check failed: {cuda_runtime_error}")
        return False, (
            f"CuPy runtime check failed: {cuda_runtime_error}; "
            f"{_nvidia_smi_summary()}; {_cuda_driver_summary()}"
        )
    if count < 1:
        return False, (
            "CuPy runtime found no devices; "
            f"{_nvidia_smi_summary()}; {_cuda_driver_summary()}"
        )
    return True, ""


def _nvidia_smi_summary() -> str:
    """Return a short NVML visibility diagnostic via nvidia-smi."""
    try:
        import subprocess

        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except Exception as nvidia_smi_error:
        return f"nvidia-smi diagnostic failed: {nvidia_smi_error}"

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        detail = f": {stderr}" if stderr else ""
        return f"nvidia-smi failed with exit={proc.returncode}{detail}"
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return "nvidia-smi reports no GPUs"
    return "nvidia-smi sees " + "; ".join(lines)


def _cuda_driver_summary() -> str:
    # Intentionally separate from the CuPy runtime gate. On WSL and mixed
    # toolkit installs, NVML/driver visibility can succeed while the CUDA
    # runtime library selected by CuPy fails to initialize.
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        version_err = int(lib.cuDriverGetVersion(ctypes.byref(version)))
        init_err = int(lib.cuInit(0))
        count = ctypes.c_int()
        count_err = int(lib.cuDeviceGetCount(ctypes.byref(count)))
    except Exception as cuda_driver_error:
        return f"CUDA driver API diagnostic failed: {cuda_driver_error}"

    if (
        version_err == CUDA_DRIVER_SUCCESS
        and init_err == CUDA_DRIVER_SUCCESS
        and count_err == CUDA_DRIVER_SUCCESS
    ):
        return (
            "CUDA driver API sees "
            f"{int(count.value)} device(s), driver_version={int(version.value)}"
        )
    return (
        "CUDA driver API errors: "
        f"cuDriverGetVersion={version_err}, cuInit={init_err}, "
        f"cuDeviceGetCount={count_err}"
    )
