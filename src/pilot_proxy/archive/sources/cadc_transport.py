"""Bounded network policy for the optional CADC client stack.

``StorageInventoryClient.cadcget`` has no timeout argument. Its cadcutils
transport uses a Requests session and supplies its own timeout, so a process
``socket.setdefaulttimeout`` does not control supported CADC clients. This
module makes the archive connect/read policy explicit on each thread-owned
session and provides a serialized, restored socket-default fallback for
simple test clients or older transports.
"""
from __future__ import annotations

import http.client
import socket
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps

CADC_CONNECT_TIMEOUT_SECONDS = 30.0
CADC_READ_TIMEOUT_SECONDS = 120.0
CADC_REQUEST_TIMEOUT = (
    CADC_CONNECT_TIMEOUT_SECONDS,
    CADC_READ_TIMEOUT_SECONDS,
)

_SOCKET_DEFAULT_LOCK = threading.RLock()
_REQUEST_DEADLINE = ContextVar("pilot_proxy_cadc_request_deadline", default=None)


@lru_cache(maxsize=1)
def expected_errors() -> tuple:
    """Archive, network, and local I/O errors a source may retry/report.

    Optional dependencies stay lazy so importing the source still works in a
    local-only installation. Exceptions outside these families are programming
    or environment-contract failures and must abort the run.
    """
    errors = [OSError]
    try:
        from cadcutils import exceptions as cadc_exceptions
    except ImportError:
        pass
    else:
        errors.append(cadc_exceptions.HttpException)
    try:
        from requests.exceptions import RequestException
    except ImportError:
        pass
    else:
        errors.append(RequestException)
    try:
        from urllib3.exceptions import HTTPError as Urllib3HTTPError
    except ImportError:
        pass
    else:
        # A truncated transfer surfaces from requests as a raw urllib3
        # ProtocolError ("Connection broken: IncompleteRead(...)"), outside
        # RequestException. It is as transient as any other broken connection
        # and killed both CANFAR shards within one minute on 2026-09-02. A
        # read that stalls inside the streamed body surfaces the same way as a
        # raw urllib3 ReadTimeoutError (2026-09-17). Both derive from urllib3's
        # HTTPError, so accept that family.
        errors.append(Urllib3HTTPError)
    errors.append(http.client.IncompleteRead)
    return tuple(errors)


@contextmanager
def socket_timeout(seconds: float, *, deadline=None):
    """Set and restore the serialized socket default within a deadline."""
    with _SOCKET_DEFAULT_LOCK:
        timeout = float(seconds)
        if deadline is not None:
            # Compute this only after acquiring the process-wide fallback lock:
            # another legacy request may have consumed the remaining budget
            # while this caller waited for serialized access.
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("survey outage deadline exceeded")
            timeout = min(timeout, remaining)
        previous = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            yield
        finally:
            socket.setdefaulttimeout(previous)


@contextmanager
def request_deadline(deadline):
    """Cap a supported Requests call at an optional monotonic deadline.

    Context variables are local to the probe thread, so concurrent archive
    requests may carry different survey deadlines without mutating process
    network defaults.
    """
    token = _REQUEST_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _REQUEST_DEADLINE.reset(token)


def configure_request_timeout(client) -> bool:
    """Apply the timeout policy to one thread-owned cadcutils session.

    Returns False when the client does not expose the supported transport
    boundary; callers then use :func:`socket_timeout` as a compatibility path.
    A read timeout is an inactivity limit, so a healthy large transfer may run
    longer while bytes continue to arrive.
    """
    transport = getattr(client, "_cadc_client", None)
    get_session = getattr(transport, "_get_session", None)
    if not callable(get_session):
        return False
    try:
        session = get_session()
        original_send = session.send
    except (AttributeError, TypeError):
        return False
    if getattr(original_send, "_pilot_proxy_bounded_timeout", False):
        return True

    @wraps(original_send)
    def bounded_send(request, **kwargs):
        timeout = CADC_REQUEST_TIMEOUT
        deadline = _REQUEST_DEADLINE.get()
        if deadline is not None:
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("survey outage deadline exceeded")
            timeout = (
                min(CADC_CONNECT_TIMEOUT_SECONDS, remaining),
                min(CADC_READ_TIMEOUT_SECONDS, remaining),
            )
        kwargs["timeout"] = timeout
        return original_send(request, **kwargs)

    bounded_send._pilot_proxy_bounded_timeout = True
    try:
        session.send = bounded_send
    except (AttributeError, TypeError):
        return False
    return True


def describe_exception(exc: BaseException) -> str:
    """Name and message of an exception, never raising.

    Some library exceptions' ``__str__`` returns a non-string (cadcutils'
    HttpException wraps an HTTPError that way), so formatting them raises
    TypeError. Inside a retry loop that turns a retryable error into the one
    that ends the run: both CANFAR shards died of it on 2026-09-03. The
    wrapped exception's own message is preferred to a bare repr.
    """
    name = type(exc).__name__
    for candidate in (exc, getattr(exc, "orig_exception", None)):
        if candidate is None:
            continue
        try:
            text = str(candidate)
        except Exception:                                  # noqa: BLE001
            continue
        if isinstance(text, str) and text:
            return f"{name}: {text}"
    try:
        text = repr(exc)
    except Exception:                                      # noqa: BLE001
        return name
    return f"{name}: {text}" if isinstance(text, str) and text else name
