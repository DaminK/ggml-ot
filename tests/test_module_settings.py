"""Settings module tests for mutation, validation, and threading behavior."""

import importlib
import pytest
import torch


def test_get_set_settings():
    """Settings object should expose mutable fields and update() support."""
    settings = importlib.import_module("ggml_ot").settings

    # basic expectations about types and sensible defaults
    assert isinstance(settings.n_threads, int)
    assert settings.n_threads >= 1
    assert hasattr(settings, "device")
    assert hasattr(settings, "verbose") and isinstance(settings.verbose, bool)

    original = settings.verbose
    settings.verbose = not original
    assert settings.verbose == (not original)

    settings.update(verbose=original)
    assert settings.verbose == original


def test_update_rejects_unknown_keys():
    """update() should reject unknown setting names."""
    settings = importlib.import_module("ggml_ot").settings
    with pytest.raises(AttributeError, match=r"Unknown setting"):
        settings.update(this_setting_does_not_exist=True)


def test_n_threads_updates_torch_thread_pool():
    """Changing n_threads should update torch's thread pool accordingly."""
    settings = importlib.import_module("ggml_ot").settings
    original_threads = settings.n_threads

    try:
        settings.n_threads = 3
        assert settings.n_threads == 3
        assert torch.get_num_threads() == 3
        settings.update(n_threads=4)
        assert settings.n_threads == 4
        assert torch.get_num_threads() == 4
    finally:
        settings.update(n_threads=original_threads)


@pytest.mark.parametrize("strategy", ["orthonormal", "orthogonal", "xavier", "random"])
def test_init_strategy_dispatch(strategy, monkeypatch):
    """Every documented init_strategy yields a trainable (n_comps, dim) map."""
    from types import SimpleNamespace

    from ggml_ot.optimization._ggml import _ggml_init

    settings = importlib.import_module("ggml_ot").settings
    monkeypatch.setattr(settings, "init_strategy", strategy)
    map_A = _ggml_init(SimpleNamespace(dataset=SimpleNamespace(dim=10)), 3)
    assert map_A.shape == (3, 10) and map_A.requires_grad
