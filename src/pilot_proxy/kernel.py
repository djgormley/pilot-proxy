# coding=utf-8
"""Kernel library interface: specs, features, handle management, config parsing."""

from __future__ import annotations

import ctypes
import operator
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .paths import DEFAULT_LIB_PATH

FINE_WINDOWS_PER_STREAM = 128


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class KernelSpecs:
    """Compile-time kernel parameters."""

    K: int          # detector_window_samples
    N: int          # num_weight_terms
    bits: int       # sample_bits_per_component
    reference_offset_bins: int

    @property
    def detector_window_samples(self) -> int:
        """Time samples per detector weight/window."""
        return int(self.K)

    @property
    def num_weight_terms(self) -> int:
        """Number of detector weight terms, currently target/lower/upper."""
        return int(self.N)

    @property
    def sample_bits_per_component(self) -> int:
        """Packed complex bits per real/imaginary component."""
        return int(self.bits)

    def as_descriptive_dict(self) -> dict[str, int]:
        """Serialize using non-ambiguous public names."""
        return {
            "detector_window_samples": self.detector_window_samples,
            "num_weight_terms": self.num_weight_terms,
            "sample_bits_per_component": self.sample_bits_per_component,
            "reference_offset_bins": int(self.reference_offset_bins),
        }


@dataclass(frozen=True)
class KernelFeatures:
    """Compiled implementation feature switches."""

    use_dp4a: bool
    use_uint64_power_accumulation: bool
    block_threads: int
    use_constant_weight_lanes: bool = False
    use_shared_weight_lanes: bool = False
    grid_max_blocks: int = 0

    @property
    def dot_product_path(self) -> str:
        """Short label for the compiled dot-product implementation."""
        return "dp4a_packed_int8_lanes" if self.use_dp4a else "scalar_integer"

    @property
    def power_accumulator(self) -> str:
        """Short label for the compiled block-power accumulator."""
        return (
            "uint64_integer"
            if self.use_uint64_power_accumulation
            else "float_diagnostic"
        )

    def as_dict(self) -> dict[str, object]:
        """Serialize compiled implementation features."""
        return {
            "dot_product_path": self.dot_product_path,
            "use_dp4a": bool(self.use_dp4a),
            "power_accumulator": self.power_accumulator,
            "use_uint64_power_accumulation": bool(
                self.use_uint64_power_accumulation
            ),
            "block_threads": int(self.block_threads),
            "use_constant_weight_lanes": bool(self.use_constant_weight_lanes),
            "use_shared_weight_lanes": bool(self.use_shared_weight_lanes),
            "grid_max_blocks": int(self.grid_max_blocks),
        }


@dataclass(frozen=True)
class KernelVersion:
    """Locked kernel core version."""

    major: int
    minor: int
    patch: int

    def as_tuple(self) -> tuple[int, int, int]:
        return int(self.major), int(self.minor), int(self.patch)

    def as_string(self) -> str:
        major, minor, patch = self.as_tuple()
        return f"{major}.{minor}.{patch}"


# =============================================================================
# Kernel Interface
# =============================================================================


def _has_symbol(lib: ctypes.CDLL, name: str) -> bool:
    # ctypes.CDLL never raises AttributeError for missing symbols — the only
    # reliable check is inspecting the underlying function pointer address.
    try:
        fn = getattr(lib, name)
        return bool(ctypes.cast(fn, ctypes.c_void_p).value)
    except (AttributeError, TypeError, ValueError):
        return False


def _last_error_from_library(lib: Any) -> str:
    """Read the thread-local error exposed by a compatible kernel library."""
    last_error_fn = getattr(lib, "FStat_LastError", None)
    if last_error_fn is None:
        return ""
    raw = last_error_fn()
    if not raw:
        return ""
    if isinstance(raw, bytes):
        return raw.decode(errors="replace")
    return str(raw)


def _raise_library_error(lib: Any, operation: str) -> None:
    """Raise when a void C API call recorded an error."""
    message = _last_error_from_library(lib)
    if message:
        raise RuntimeError(f"{operation} failed: {message}")


def _checked_uint64(value: object, *, field: str, positive: bool = False) -> int:
    """Validate one exact integer before crossing the ctypes boundary."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, not a boolean.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an integer.") from exc
    lower = 1 if positive else 0
    if not lower <= result < (1 << 64):
        interval = "[1, 2**64 - 1]" if positive else "[0, 2**64 - 1]"
        raise ValueError(f"{field} must be in uint64 range {interval}; got {result}.")
    return result


def _checked_c_int(
    value: object,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate an exact integer and its semantic range before ctypes casts it."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, not a boolean.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an integer.") from exc

    bits = ctypes.sizeof(ctypes.c_int) * 8
    c_min = -(1 << (bits - 1))
    c_max = (1 << (bits - 1)) - 1
    lower = c_min if minimum is None else max(c_min, minimum)
    upper = c_max if maximum is None else min(c_max, maximum)
    if not lower <= result <= upper:
        raise ValueError(
            f"{field} must be in range [{lower}, {upper}]; got {result}."
        )
    return result


class FStatKernel:
    """Wrapper for the F-statistic CUDA kernel library."""

    def __init__(self, lib_path: str | os.PathLike[str] = DEFAULT_LIB_PATH):
        """Load the kernel shared library and initialize metadata."""
        lib_path_text = str(lib_path)
        if not os.path.exists(lib_path_text):
            raise FileNotFoundError(f"Kernel library not found: {lib_path_text}")

        self._lib = ctypes.CDLL(lib_path_text)
        self._setup_signatures()
        self.specs = self._get_specs()
        self.features = self._get_features()
        self.version = self._get_version()

    def _setup_signatures(self):
        """Configure ctypes function signatures."""
        if not _has_symbol(self._lib, "FStat_GetSpecs"):
            raise RuntimeError("Kernel library does not expose FStat_GetSpecs.")
        self._lib.FStat_GetSpecs.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4
        self._lib.FStat_GetSpecs.restype = None
        self._has_features = _has_symbol(self._lib, "FStat_GetFeatures")
        if self._has_features:
            self._lib.FStat_GetFeatures.argtypes = [
                ctypes.POINTER(ctypes.c_int)
            ] * 3
            self._lib.FStat_GetFeatures.restype = None
        self._has_optimization_features = _has_symbol(
            self._lib, "FStat_GetOptimizationFeatures"
        )
        if self._has_optimization_features:
            self._lib.FStat_GetOptimizationFeatures.argtypes = [
                ctypes.POINTER(ctypes.c_int)
            ] * 3
            self._lib.FStat_GetOptimizationFeatures.restype = None
        if not _has_symbol(self._lib, "FStat_GetVersion"):
            raise RuntimeError("Kernel library does not expose FStat_GetVersion.")
        self._lib.FStat_GetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
        self._lib.FStat_GetVersion.restype = None
        self._has_last_error = _has_symbol(self._lib, "FStat_LastError")
        if self._has_last_error:
            self._lib.FStat_LastError.argtypes = []
            self._lib.FStat_LastError.restype = ctypes.c_char_p

        if not _has_symbol(self._lib, "FStat_Create"):
            raise RuntimeError("Kernel library does not expose FStat_Create.")
        self._lib.FStat_Create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._lib.FStat_Create.restype = ctypes.c_void_p

        self._has_batch_create = _has_symbol(self._lib, "FStat_Create_Batch")
        if self._has_batch_create:
            self._lib.FStat_Create_Batch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self._lib.FStat_Create_Batch.restype = ctypes.c_void_p

        if not _has_symbol(self._lib, "FStat_Compute_DiagnosticFloat"):
            raise RuntimeError(
                "Kernel library does not expose FStat_Compute_DiagnosticFloat."
            )
        self._lib.FStat_Compute_DiagnosticFloat.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # weights pointer
        ]
        self._lib.FStat_Compute_DiagnosticFloat.restype = None

        self._lib.FStat_Destroy.argtypes = [ctypes.c_void_p]
        self._lib.FStat_Destroy.restype = None

        self._has_powers = _has_symbol(self._lib, "FStat_Compute_Powers")
        if self._has_powers:
            self._lib.FStat_Compute_Powers.argtypes = [
                ctypes.c_void_p,  # handle
                ctypes.c_void_p,  # weights pointer
            ]
            self._lib.FStat_Compute_Powers.restype = None
        self._has_powers_u64 = _has_symbol(self._lib, "FStat_Compute_Powers_U64")
        if self._has_powers_u64:
            self._lib.FStat_Compute_Powers_U64.argtypes = [
                ctypes.c_void_p,  # handle
                ctypes.c_void_p,  # weights pointer
                ctypes.c_void_p,  # uint64 device output pointer
            ]
            self._lib.FStat_Compute_Powers_U64.restype = None
        self._has_matched_filter_row_projections_i32 = _has_symbol(
            self._lib, "FStat_Compute_RowSums_I32"
        )
        if self._has_matched_filter_row_projections_i32:
            self._lib.FStat_Compute_RowSums_I32.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._lib.FStat_Compute_RowSums_I32.restype = None
        self._has_supports_row_projections = _has_symbol(
            self._lib, "FStat_Supports_RowSums"
        )
        if self._has_supports_row_projections:
            self._lib.FStat_Supports_RowSums.argtypes = []
            self._lib.FStat_Supports_RowSums.restype = ctypes.c_int
        self._has_fine_powers_u64 = _has_symbol(
            self._lib, "FStat_Compute_FinePowers_U64"
        )
        if self._has_fine_powers_u64:
            self._lib.FStat_Compute_FinePowers_U64.argtypes = [
                ctypes.c_void_p,  # device row-sum input pointer
                ctypes.c_int,     # num_weight_terms
                ctypes.c_int,     # num_streams
                ctypes.c_int,     # windows_per_stream (frozen: 128)
                ctypes.c_int,     # batch
                ctypes.c_void_p,  # device uint64 fine-power output pointer
            ]
            self._lib.FStat_Compute_FinePowers_U64.restype = None
        self._has_supports_fine_powers = _has_symbol(
            self._lib, "FStat_Supports_FinePowers"
        )
        if self._has_supports_fine_powers:
            self._lib.FStat_Supports_FinePowers.argtypes = []
            self._lib.FStat_Supports_FinePowers.restype = ctypes.c_int
        self._has_fused_fine_u64 = _has_symbol(
            self._lib, "FStat_Compute_FusedFine_U64"
        )
        if self._has_fused_fine_u64:
            self._lib.FStat_Compute_FusedFine_U64.argtypes = [
                ctypes.c_void_p,  # handle
                ctypes.c_void_p,  # weights pointer (host)
                ctypes.c_void_p,  # device uint64 fine-power output pointer
                ctypes.c_void_p,  # device uint64 marginal-power output pointer
                ctypes.c_void_p,  # optional device int32 row-sum tap (or NULL)
            ]
            self._lib.FStat_Compute_FusedFine_U64.restype = None
        self._has_supports_fused_fine = _has_symbol(
            self._lib, "FStat_Supports_FusedFine"
        )
        if self._has_supports_fused_fine:
            self._lib.FStat_Supports_FusedFine.argtypes = []
            self._lib.FStat_Supports_FusedFine.restype = ctypes.c_int
        self._has_fused_fine_mask_u64 = _has_symbol(
            self._lib, "FStat_Compute_FusedFineMask_U64"
        )
        if self._has_fused_fine_mask_u64:
            self._lib.FStat_Compute_FusedFineMask_U64.argtypes = [
                ctypes.c_void_p,   # handle
                ctypes.c_void_p,   # weights pointer (host)
                ctypes.c_int,      # anchor_bin
                ctypes.c_int,      # designated_half_width
                ctypes.POINTER(ctypes.c_uint64),  # bulk mask words (host, 4)
                ctypes.c_int,      # cfar_rank
                ctypes.c_ulonglong,  # multiplier_q16
                ctypes.c_void_p,   # device uint64 fine-power accumulator
                ctypes.c_void_p,   # device int32 mask output
                ctypes.c_void_p,   # optional device uint64 marginals (or NULL)
                ctypes.c_void_p,   # optional device int32 row-sum tap (or NULL)
            ]
            self._lib.FStat_Compute_FusedFineMask_U64.restype = None
        self._has_supports_fused_fine_mask = _has_symbol(
            self._lib, "FStat_Supports_FusedFineMask"
        )
        if self._has_supports_fused_fine_mask:
            self._lib.FStat_Supports_FusedFineMask.argtypes = []
            self._lib.FStat_Supports_FusedFineMask.restype = ctypes.c_int
        self._has_get_fine_specs = _has_symbol(self._lib, "FStat_GetFineSpecs")
        if self._has_get_fine_specs:
            self._lib.FStat_GetFineSpecs.argtypes = [
                ctypes.POINTER(ctypes.c_int)
            ] * 3
            self._lib.FStat_GetFineSpecs.restype = None

        self._has_numden_mask_rational_half = _has_symbol(
            self._lib, "FStat_Compute_NumDen_Mask_RationalHalf"
        )
        if self._has_numden_mask_rational_half:
            self._lib.FStat_Compute_NumDen_Mask_RationalHalf.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._lib.FStat_Compute_NumDen_Mask_RationalHalf.restype = None
        self._has_numden_mask_rational_half_checked = _has_symbol(
            self._lib, "FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount"
        )
        if self._has_numden_mask_rational_half_checked:
            checked = (
                self._lib.FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount
            )
            checked.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            checked.restype = None

    def _get_specs(self) -> KernelSpecs:
        """Query compile-time kernel parameters."""
        c_k = ctypes.c_int()
        c_n = ctypes.c_int()
        c_b = ctypes.c_int()
        c_reference_offset_bins = ctypes.c_int()

        self._lib.FStat_GetSpecs(
            ctypes.byref(c_k),
            ctypes.byref(c_n),
            ctypes.byref(c_b),
            ctypes.byref(c_reference_offset_bins),
        )
        _raise_library_error(self._lib, "FStat_GetSpecs")

        return KernelSpecs(
            K=c_k.value,
            N=c_n.value,
            bits=c_b.value,
            reference_offset_bins=c_reference_offset_bins.value,
        )

    def _get_features(self) -> KernelFeatures:
        """Query compiled implementation features when supported by the library."""
        if not getattr(self, "_has_features", False):
            return KernelFeatures(
                use_dp4a=False,
                use_uint64_power_accumulation=False,
                block_threads=0,
                use_constant_weight_lanes=False,
                use_shared_weight_lanes=False,
                grid_max_blocks=0,
            )

        c_dp4a = ctypes.c_int()
        c_uint64_power = ctypes.c_int()
        c_threads = ctypes.c_int()
        c_constant_weight_lanes = ctypes.c_int()
        c_shared_weight_lanes = ctypes.c_int()
        c_grid_max_blocks = ctypes.c_int()
        self._lib.FStat_GetFeatures(
            ctypes.byref(c_dp4a),
            ctypes.byref(c_uint64_power),
            ctypes.byref(c_threads),
        )
        _raise_library_error(self._lib, "FStat_GetFeatures")
        if getattr(self, "_has_optimization_features", False):
            self._lib.FStat_GetOptimizationFeatures(
                ctypes.byref(c_constant_weight_lanes),
                ctypes.byref(c_shared_weight_lanes),
                ctypes.byref(c_grid_max_blocks),
            )
            _raise_library_error(self._lib, "FStat_GetOptimizationFeatures")
        return KernelFeatures(
            use_dp4a=bool(c_dp4a.value),
            use_uint64_power_accumulation=bool(c_uint64_power.value),
            block_threads=int(c_threads.value),
            use_constant_weight_lanes=bool(c_constant_weight_lanes.value),
            use_shared_weight_lanes=bool(c_shared_weight_lanes.value),
            grid_max_blocks=int(c_grid_max_blocks.value),
        )

    def _get_version(self) -> KernelVersion:
        """Query the locked core version."""
        c_major = ctypes.c_int()
        c_minor = ctypes.c_int()
        c_patch = ctypes.c_int()

        self._lib.FStat_GetVersion(
            ctypes.byref(c_major),
            ctypes.byref(c_minor),
            ctypes.byref(c_patch),
        )
        _raise_library_error(self._lib, "FStat_GetVersion")

        return KernelVersion(
            major=int(c_major.value),
            minor=int(c_minor.value),
            patch=int(c_patch.value),
        )

    def create_handle(self, M: int, d_in: Any, d_out: Any):
        """Create a kernel handle (use as a context manager)."""
        rows = _checked_c_int(
            M,
            field="detector_rows_per_block",
            minimum=1,
        )
        return _KernelHandle(self._lib, rows, d_in, d_out)

    def create_detector_matrix_handle(
        self, detector_rows_per_block: int, d_in: Any, d_out: Any
    ):
        """Create a handle using descriptive detector-matrix terminology."""
        return self.create_handle(detector_rows_per_block, d_in, d_out)

    def create_batch_handle(self, M: int, batch: int, d_in: Any, d_out: Any):
        """Create a batched kernel handle (use as a context manager)."""
        rows = _checked_c_int(
            M,
            field="detector_rows_per_block",
            minimum=1,
        )
        batch_count = _checked_c_int(batch, field="batch", minimum=1)
        return _KernelHandle(
            self._lib,
            rows,
            d_in,
            d_out,
            batch=batch_count,
        )

    def create_detector_matrix_batch_handle(
        self,
        detector_rows_per_block: int,
        batch: int,
        d_in: Any,
        d_out: Any,
    ):
        """Create a batched handle using descriptive detector-matrix terminology."""
        return self.create_batch_handle(
            detector_rows_per_block, batch, d_in, d_out
        )

    def create_raw(self, M: int, in_ptr: int, out_ptr: int):
        """Create a handle from raw pointers."""
        rows = _checked_c_int(
            M,
            field="detector_rows_per_block",
            minimum=1,
        )
        handle = self._lib.FStat_Create(in_ptr, out_ptr, rows)
        if not handle:
            raise RuntimeError(
                self.last_error() or "FStat_Create returned NULL."
            )
        return handle

    def create_raw_batch(self, M: int, batch: int, in_ptr: int, out_ptr: int):
        """Create a batched handle from raw pointers."""
        if not getattr(self, "_has_batch_create", False):
            raise RuntimeError("Kernel library does not expose FStat_Create_Batch.")
        rows = _checked_c_int(
            M,
            field="detector_rows_per_block",
            minimum=1,
        )
        batch_count = _checked_c_int(batch, field="batch", minimum=1)
        handle = self._lib.FStat_Create_Batch(
            in_ptr,
            out_ptr,
            rows,
            batch_count,
        )
        if not handle:
            raise RuntimeError(
                self.last_error() or "FStat_Create_Batch returned NULL."
            )
        return handle

    def last_error(self) -> str:
        """Return the last CUDA/API error reported by the kernel library."""
        if not getattr(self, "_has_last_error", False):
            return ""
        return _last_error_from_library(self._lib)

    def _check_call(self, operation: str) -> None:
        if getattr(self, "_has_last_error", False):
            _raise_library_error(self._lib, operation)

    @property
    def supports_batch(self) -> bool:
        """Return True if the kernel library supports batched handles."""
        return bool(getattr(self, "_has_batch_create", False))

    def compute_diagnostic_float(self, handle, weights_ptr: int):
        """Execute the diagnostic floating-point F-statistic path."""
        self._lib.FStat_Compute_DiagnosticFloat(handle, weights_ptr)
        self._check_call("FStat_Compute_DiagnosticFloat")

    def compute_powers(self, handle, weights_ptr: int):
        """Execute the kernel and write per-weight power terms to d_out."""
        if not getattr(self, "_has_powers", False):
            raise RuntimeError("Kernel library does not expose FStat_Compute_Powers.")
        self._lib.FStat_Compute_Powers(handle, weights_ptr)
        self._check_call("FStat_Compute_Powers")

    def compute_powers_u64(self, handle, weights_ptr: int, powers_ptr: int):
        """Execute the kernel and write exact uint64 power terms to powers_ptr."""
        if not getattr(self, "_has_powers_u64", False):
            raise RuntimeError(
                "Kernel library does not expose FStat_Compute_Powers_U64."
            )
        self._lib.FStat_Compute_Powers_U64(handle, weights_ptr, powers_ptr)
        self._check_call("FStat_Compute_Powers_U64")

    def supports_row_projections(self) -> bool:
        """Return True when the v2 exact row-sum front end is available."""
        if not getattr(self, "_has_matched_filter_row_projections_i32", False):
            return False
        if getattr(self, "_has_supports_row_projections", False):
            return bool(int(self._lib.FStat_Supports_RowSums()) == 1)
        return True

    def compute_row_projections_i32(
        self, handle, weights_ptr: int, row_projections_ptr: int
    ):
        """Write exact int32 complex row sums (v2 front end).

        row_projections_ptr must address a device buffer of
        batch * num_weight_terms * detector_rows_per_block * 2 int32
        elements; layout per the FStat_Compute_RowSums_I32 contract
        (term-major, stream-major rows, interleaved re/im).
        """
        if not getattr(self, "_has_matched_filter_row_projections_i32", False):
            raise RuntimeError(
                "Kernel library does not expose FStat_Compute_RowSums_I32."
            )
        self._lib.FStat_Compute_RowSums_I32(
            handle, weights_ptr, row_projections_ptr
        )
        self._check_call("FStat_Compute_RowSums_I32")

    def supports_fine_powers(self) -> bool:
        """Return True when the on-device fine-power stage is available."""
        if not getattr(self, "_has_fine_powers_u64", False):
            return False
        if getattr(self, "_has_supports_fine_powers", False):
            return bool(int(self._lib.FStat_Supports_FinePowers()) == 1)
        return True

    def get_fine_specs(self) -> dict[str, int]:
        """Frozen fine-reduction geometry compiled into the library."""
        if not getattr(self, "_has_get_fine_specs", False):
            raise RuntimeError(
                "Kernel library does not expose FStat_GetFineSpecs."
            )
        w = ctypes.c_int(0)
        p = ctypes.c_int(0)
        b = ctypes.c_int(0)
        self._lib.FStat_GetFineSpecs(
            ctypes.byref(w), ctypes.byref(p), ctypes.byref(b)
        )
        return {
            "windows_per_stream": int(w.value),
            "pad_factor": int(p.value),
            "fine_bins": int(b.value),
        }

    def compute_fine_powers_u64(
        self,
        row_projections_ptr: int,
        num_streams: int,
        windows_per_stream: int,
        batch: int,
        fine_powers_ptr: int,
    ):
        """On-device fine-reduction power stage (fxfft256 v1).

        Composes directly with compute_row_projections_i32: row_projections_ptr is the
        same device buffer that call filled (term-major, stream-major rows,
        interleaved re/im int32), and fine_powers_ptr must address
        batch * num_weight_terms * 256 uint64 on the device (zeroed by the
        library before accumulation). Exact integers throughout; the
        transform is the frozen fxfft256 v1 specification. This entry
        takes no handle. windows_per_stream must be 128.
        """
        if not getattr(self, "_has_fine_powers_u64", False):
            raise RuntimeError(
                "Kernel library does not expose FStat_Compute_FinePowers_U64."
            )
        terms = _checked_c_int(
            self.specs.num_weight_terms,
            field="num_weight_terms",
            minimum=1,
        )
        stream_capacity: int | None = None
        features = getattr(self, "features", None)
        if features is not None:
            block_threads = int(getattr(features, "block_threads", 0))
            grid_max_blocks = int(getattr(features, "grid_max_blocks", 0))
            if block_threads > 0 and grid_max_blocks > 0:
                stream_capacity = block_threads * grid_max_blocks
        streams = _checked_c_int(
            num_streams,
            field="num_streams",
            minimum=1,
            maximum=stream_capacity,
        )
        windows = _checked_c_int(
            windows_per_stream,
            field="windows_per_stream",
            minimum=FINE_WINDOWS_PER_STREAM,
            maximum=FINE_WINDOWS_PER_STREAM,
        )
        batch_count = _checked_c_int(batch, field="batch", minimum=1)
        self._lib.FStat_Compute_FinePowers_U64(
            row_projections_ptr,
            terms,
            streams,
            windows,
            batch_count,
            fine_powers_ptr,
        )
        self._check_call("FStat_Compute_FinePowers_U64")

    def supports_fused_fine(self) -> bool:
        """Return True when the fused fine kernel (core 2.2.0) is available."""
        if not getattr(self, "_has_fused_fine_u64", False):
            return False
        if getattr(self, "_has_supports_fused_fine", False):
            return bool(int(self._lib.FStat_Supports_FusedFine()) == 1)
        return True

    def compute_fused_fine_u64(
        self,
        handle,
        weights_ptr: int,
        fine_powers_ptr: int,
        powers_ptr: int,
        row_projections_tap_ptr: int = 0,
    ):
        """One launch from packed samples to exact fine and coarse powers.

        The fused form of the deployed fine path (kernel core 2.2.0):
        row sums live in shared memory, the exact uint64 coarse marginals
        and the frozen fxfft256 v1 fine powers are accumulated in the same
        launch, and the bit-exact marginal identity is internal to the
        kernel. Output layouts match the composed path exactly:
        fine_powers_ptr addresses batch * num_weight_terms * 256 uint64
        and powers_ptr addresses batch * num_weight_terms uint64 (both
        zeroed by the library before accumulation). row_projections_tap_ptr, when
        nonzero, addresses the same device buffer layout
        compute_row_projections_i32 fills and receives the identical bits; leave
        it 0 in production so row sums never touch global memory.
        """
        if not getattr(self, "_has_fused_fine_u64", False):
            raise RuntimeError(
                "Kernel library does not expose FStat_Compute_FusedFine_U64."
            )
        self._lib.FStat_Compute_FusedFine_U64(
            handle,
            weights_ptr,
            fine_powers_ptr,
            powers_ptr,
            row_projections_tap_ptr if row_projections_tap_ptr else None,
        )
        self._check_call("FStat_Compute_FusedFine_U64")

    def supports_fused_fine_mask(self) -> bool:
        """Return True when the decision epilogue (core 2.3.0) is available."""
        if not getattr(self, "_has_fused_fine_mask_u64", False):
            return False
        if getattr(self, "_has_supports_fused_fine_mask", False):
            return bool(int(self._lib.FStat_Supports_FusedFineMask()) == 1)
        return True

    def compute_fused_fine_mask_u64(
        self,
        handle,
        weights_ptr: int,
        anchor_bin: int,
        designated_half_width: int,
        bulk_mask_words,
        cfar_rank: int,
        multiplier_q16: int,
        fine_powers_ptr: int,
        mask_ptr: int,
        powers_ptr: int = 0,
        row_projections_tap_ptr: int = 0,
    ):
        """Deployed form: packed samples and bundle constants in, one
        mask bit per aligned frame out (kernel core 2.3.0).

        The fused datapath plus the frozen fine decision v1 epilogue
        (bit-identical to ``pilot_proxy.fine_decision.fine_mask_decision``
        over the same exact fine powers). ``bulk_mask_words`` is the
        4-word packed 256-bit bulk mask (``fine_decision.pack_bulk_mask``).
        ``mask_ptr`` must address ``batch`` int32 on the device (zeroed
        by the library; doubles as the completion counter during the
        launch). ``fine_powers_ptr`` is the required exact accumulator;
        ``powers_ptr`` and ``row_projections_tap_ptr`` are optional debug taps
        (0 in production). anchor/width/rank/multiplier are runtime
        bundle data.
        """
        if not getattr(self, "_has_fused_fine_mask_u64", False):
            raise RuntimeError(
                "Kernel library does not expose "
                "FStat_Compute_FusedFineMask_U64."
            )
        raw_words = list(bulk_mask_words)
        if len(raw_words) != 4:
            raise ValueError("bulk_mask_words must contain 4 uint64 words.")
        words = [
            _checked_uint64(word, field=f"bulk_mask_words[{index}]")
            for index, word in enumerate(raw_words)
        ]
        multiplier = _checked_uint64(
            multiplier_q16,
            field="multiplier_q16",
            positive=True,
        )
        anchor = _checked_c_int(
            anchor_bin,
            field="anchor_bin",
            minimum=0,
            maximum=255,
        )
        half_width = _checked_c_int(
            designated_half_width,
            field="designated_half_width",
            minimum=0,
            maximum=127,
        )
        rank = _checked_c_int(
            cfar_rank,
            field="cfar_rank",
            minimum=0,
            maximum=255,
        )
        arr = (ctypes.c_uint64 * 4)(*words)
        self._lib.FStat_Compute_FusedFineMask_U64(
            handle,
            weights_ptr,
            anchor,
            half_width,
            arr,
            rank,
            multiplier,
            fine_powers_ptr,
            mask_ptr,
            powers_ptr if powers_ptr else None,
            row_projections_tap_ptr if row_projections_tap_ptr else None,
        )
        self._check_call("FStat_Compute_FusedFineMask_U64")

    def compute_numden_mask_rational_half(
        self,
        handle,
        weights_ptr: int,
        threshold_half_numerator: int,
        threshold_half_denominator: int,
        numerator_ptr: int,
        denominator_ptr: int,
        mask_ptr: int,
    ):
        """Execute the half-threshold path and write numerator/denominator/mask."""
        if not getattr(self, "_has_numden_mask_rational_half", False):
            raise RuntimeError(
                "Kernel library does not expose "
                "FStat_Compute_NumDen_Mask_RationalHalf."
            )
        numerator = _checked_uint64(
            threshold_half_numerator,
            field="threshold_half_numerator",
        )
        denominator = _checked_uint64(
            threshold_half_denominator,
            field="threshold_half_denominator",
            positive=True,
        )
        self._lib.FStat_Compute_NumDen_Mask_RationalHalf(
            handle,
            weights_ptr,
            ctypes.c_ulonglong(numerator),
            ctypes.c_ulonglong(denominator),
            numerator_ptr,
            denominator_ptr,
            mask_ptr,
        )
        self._check_call("FStat_Compute_NumDen_Mask_RationalHalf")

    def compute_numden_mask_rational_half_checked(
        self,
        handle,
        weights_ptr: int,
        threshold_half_numerator: int,
        threshold_half_denominator: int,
        numerator_ptr: int,
        denominator_ptr: int,
        mask_ptr: int,
        rational_overflow_count_ptr: int,
    ):
        """Execute half-threshold num/den/mask and write an overflow counter."""
        if not getattr(self, "_has_numden_mask_rational_half_checked", False):
            raise RuntimeError(
                "Kernel library does not expose "
                "FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount."
            )
        numerator = _checked_uint64(
            threshold_half_numerator,
            field="threshold_half_numerator",
        )
        denominator = _checked_uint64(
            threshold_half_denominator,
            field="threshold_half_denominator",
            positive=True,
        )
        self._lib.FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount(
            handle,
            weights_ptr,
            ctypes.c_ulonglong(numerator),
            ctypes.c_ulonglong(denominator),
            numerator_ptr,
            denominator_ptr,
            mask_ptr,
            rational_overflow_count_ptr,
        )
        self._check_call(
            "FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount"
        )

    def destroy(self, handle):
        """Destroy a kernel handle."""
        self._lib.FStat_Destroy(handle)
        self._check_call("FStat_Destroy")


class _KernelHandle:
    """Context manager for kernel handles."""

    def __init__(
        self,
        lib,
        M: int,
        d_in: Any,
        d_out: Any,
        *,
        batch: int = 1,
    ):
        """Create a kernel handle for the provided device buffers."""
        self._lib = lib
        if batch < 1:
            raise ValueError("batch must be >= 1.")
        if batch > 1:
            create_batch = getattr(lib, "FStat_Create_Batch", None)
            if create_batch is None:
                raise RuntimeError("Kernel library does not expose FStat_Create_Batch.")
            self._handle = create_batch(d_in.data.ptr, d_out.data.ptr, M, batch)
        else:
            self._handle = lib.FStat_Create(
                d_in.data.ptr,
                d_out.data.ptr,
                M,
            )
        if not self._handle:
            last_error = _last_error_from_library(lib)
            raise RuntimeError(
                last_error or "F-statistic kernel handle creation failed."
            )

    def __enter__(self):
        """Return the underlying handle for use in a context manager."""
        return self._handle

    def __exit__(self, *args):
        """Destroy the underlying handle when exiting the context."""
        self._lib.FStat_Destroy(self._handle)
        _raise_library_error(self._lib, "FStat_Destroy")
