"""Pytest setup and CLI-option integration tests."""

from types import SimpleNamespace

import pytest

import ggml_ot

from .utils.setup_dataset import _get_data_source


@pytest.mark.dev
def test_threads_option_applied(pytestconfig):
    """Dev-only test to validate that pytest ``--threads`` sets ggml_ot.settings.n_threads."""

    threads = pytestconfig.getoption("threads")
    if threads is None:
        pytest.skip("dev test: run with --threads to validate")

    assert ggml_ot.settings.n_threads == int(threads)


@pytest.mark.dev
def test_data_source_option_applied(pytestconfig):
    """Dev-only test to validate that ``--data-source`` is parsed and normalized."""
    raw_source = pytestconfig.getoption("data_source")
    assert raw_source in {"synthetic", "network", "all"}

    # Mirror the path used by setup_dataset helpers without triggering data loading.
    request_like = SimpleNamespace(config=pytestconfig)
    if raw_source == "all":
        with pytest.raises(ValueError, match="--data-source=all"):
            _get_data_source(request_like)
    else:
        resolved_source = _get_data_source(request_like)
        assert resolved_source == raw_source
