"""Transient transport failures are retried, and no exception's own
formatting can end a run (docs/POST_RUN_DEFERRED.md@30e50c2 items 17 and 18)."""
from __future__ import annotations

import http.client

import pytest

from pilot_proxy.archive.sources import cadc_transport


def test_truncated_transfers_are_expected_errors():
    urllib3 = pytest.importorskip("urllib3")
    errors = cadc_transport.expected_errors()
    assert http.client.IncompleteRead in errors
    # Raw urllib3 failures escape requests when the streamed body breaks
    # (truncation) or stalls (read timeout); both must be retryable.
    assert issubclass(urllib3.exceptions.ProtocolError, errors)
    assert issubclass(urllib3.exceptions.ReadTimeoutError, errors)
    assert OSError in errors


class _BadStr(ConnectionError):
    """An exception whose __str__ returns a non-string, like cadcutils'."""

    def __str__(self):                                   # noqa: D105
        return 42                                         # type: ignore[return-value]


class _Wrapped(Exception):
    def __init__(self):
        super().__init__()
        self.orig_exception = ValueError("inner detail")

    def __str__(self):                                   # noqa: D105
        return None                                       # type: ignore[return-value]


def test_describe_exception_never_raises():
    text = cadc_transport.describe_exception(_BadStr())
    assert text.startswith("_BadStr")
    assert cadc_transport.describe_exception(_Wrapped()) == "_Wrapped: inner detail"
    assert cadc_transport.describe_exception(RuntimeError("plain")) == "RuntimeError: plain"
