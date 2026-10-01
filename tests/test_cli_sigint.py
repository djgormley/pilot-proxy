"""The CLI restores a usable SIGINT when a launcher left it ignored
(docs/POST_RUN_DEFERRED.md@30e50c2 item 14)."""
from __future__ import annotations

import signal

import pytest

from pilot_proxy import cli


def test_main_takes_sigint_back_from_an_ignoring_parent():
    previous = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, previous)


def test_main_leaves_a_custom_handler_alone():
    previous = signal.getsignal(signal.SIGINT)
    handler = lambda *_: None                              # noqa: E731
    try:
        signal.signal(signal.SIGINT, handler)
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        assert signal.getsignal(signal.SIGINT) is handler
    finally:
        signal.signal(signal.SIGINT, previous)
