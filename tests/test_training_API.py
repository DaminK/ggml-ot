"""Training API validation tests for user-facing solver constraints and warnings."""

import numpy as np
import pytest
import torch

import ggml_ot
import ggml_ot.optimization.api as interface_mod
from ggml_ot.data.generic import TripletDataset
from .utils.config import get_synth_config


def _make_covariance_dataset() -> TripletDataset:
    """Minimal dataset **with covariances** for GMM / Sinkhorn API tests.

    .. todo::
        Switch to the synthetic GMM dataset generation function once
        implemented (see ``ggml_ot.data``), so that this hand-rolled
        factory can be removed.
    """
    rng = np.random.default_rng(0)
    n_dists, n_points, dim = 6, 4, 3
    supports = [rng.normal(size=(n_points, dim)).astype(np.float32) for _ in range(n_dists)]
    labels = np.array([0, 0, 0, 1, 1, 1])
    covariances = [np.stack([np.eye(dim, dtype=np.float32) for _ in range(n_points)], axis=0) for _ in range(n_dists)]
    return TripletDataset(supports=supports, covariances=covariances, distribution_labels=labels)


def test_train_rejects_gmm_params_without_fitted_gmm():
    """GMM-specific options must not be accepted for datasets without covariances."""
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    with pytest.raises(ValueError, match="requires a dataset with covariances"):
        ggml_ot.train(ds, mi_reg=1.0, plot_iter=0, return_dataset=False, max_iter=1)


def test_train_sinkhorn_rejects_nonpositive_entropic_reg():
    """Solver-specific Sinkhorn API should reject nonpositive entropic regularization."""
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    with pytest.raises(ValueError, match="entropic_reg > 0"):
        interface_mod.train_sinkhorn(
            dataset=ds,
            alpha=10.0,
            reg_type="fro",
            n_comps=2,
            entropic_reg=0.0,
            plot_iter=0,
            max_iter=1,
        )


def test_train_sinkhorn_accepts_covariance_dataset():
    """Sinkhorn route should run on covariance-bearing datasets."""
    ds = _make_covariance_dataset()

    map_A, epoch_time = interface_mod.train_sinkhorn(
        dataset=ds,
        alpha=10.0,
        reg_type="fro",
        n_comps=2,
        entropic_reg=0.5,
        plot_iter=0,
        max_iter=1,
        verbose=False,
        batch_size=6,
        train_size=None,
    )

    assert isinstance(map_A, np.ndarray)
    assert map_A.ndim == 2
    assert np.isfinite(epoch_time)


def test_train_history_returns_map_snapshots_for_debug_plotting():
    """train_history() returns map_A snapshots from init through each epoch."""
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)

    history = ggml_ot.train_history(
        ds,
        n_comps=2,
        max_iter=2,
        stop_thr=None,
        plot_iter=0,
        verbose=False,
    )

    assert isinstance(history, list)
    assert len(history) == 3
    assert all(isinstance(map_A, np.ndarray) for map_A in history)
    assert all(map_A.shape == history[0].shape for map_A in history)
    assert np.isfinite(history[-1]).all()
    assert not np.shares_memory(history[0], history[1])


def test_train_gmm_requires_covariances():
    """GMM EMD2 convenience API should require covariance tensors."""
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    with pytest.raises(ValueError, match="requires a dataset with covariances"):
        interface_mod.train_gmm(
            dataset=ds,
            alpha=10.0,
            reg_type="fro",
            n_comps=2,
            lr=0.1,
            max_iter=1,
            verbose=False,
            plot_iter=0,
        )


def test_train_with_covariances_should_forward_squared_ground_cost_to_ot_params(monkeypatch):
    """Generic train() should thread squared_ground_cost into Gaussian OT params."""
    ds = _make_covariance_dataset()
    captured = {}

    def _fake_ggml_generic(**kwargs):
        captured["ot_params"] = kwargs["ot_params"]
        return np.eye(2), 0.1

    monkeypatch.setattr(interface_mod, "_ggml_generic", _fake_ggml_generic)

    interface_mod.train(
        dataset=ds,
        alpha=10.0,
        reg_type="fro",
        n_comps=2,
        max_iter=1,
        plot_iter=0,
        return_dataset=False,
        squared_ground_cost=False,
    )

    assert getattr(captured["ot_params"], "squared_ground_cost", None) is False


def test_train_gmm_should_forward_squared_ground_cost_to_ot_params(monkeypatch):
    """GMM training helper should forward squared_ground_cost into Wasserstein-Bures params."""
    ds = _make_covariance_dataset()
    captured = {}

    def _fake_ggml_generic(**kwargs):
        captured["ot_params"] = kwargs["ot_params"]
        return np.eye(2), 0.1

    monkeypatch.setattr(interface_mod, "_ggml_generic", _fake_ggml_generic)

    interface_mod.train_gmm(
        dataset=ds,
        alpha=10.0,
        reg_type="fro",
        n_comps=2,
        lr=0.1,
        max_iter=1,
        verbose=False,
        plot_iter=0,
        squared_ground_cost=False,
    )

    assert captured["ot_params"].squared_ground_cost is False


def test_train_emd2_warns_for_cuda_emd2(monkeypatch):
    """Exact EMD2 should warn when CUDA is selected as OT solve is CPU-bound."""
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    monkeypatch.setattr(interface_mod.settings, "_device", torch.device("cuda:0"), raising=False)
    monkeypatch.setattr(interface_mod, "_ggml_generic", lambda **kwargs: (np.eye(2), 0.1))

    with pytest.warns(UserWarning, match="CPU-bound"):
        interface_mod.train_emd2(
            dataset=ds,
            alpha=10.0,
            reg_type="fro",
            n_comps=2,
            plot_iter=0,
            max_iter=1,
        )


@pytest.mark.parametrize(
    ("method_name", "api_name", "call_kwargs", "expected_kwargs", "return_time"),
    [
        pytest.param(
            "train_emd2",
            "train_emd2",
            {"alpha": 7.0, "max_iter": 2, "plot_iter": 0},
            {"alpha": 7.0, "max_iter": 2},
            0.1,
            id="emd2",
        ),
        pytest.param(
            "train_sinkhorn",
            "train_sinkhorn",
            {"alpha": 5.0, "entropic_reg": 0.5, "max_iter": 2, "plot_iter": 0},
            {"alpha": 5.0, "entropic_reg": 0.5},
            0.2,
            id="sinkhorn",
        ),
    ],
)
def test_dataset_solver_wrappers_dispatch_to_api(
    monkeypatch, method_name, api_name, call_kwargs, expected_kwargs, return_time
):
    """Dataset solver wrapper methods should forward to the corresponding API function."""
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    calls = {}

    def _fake_solver(dataset, **kwargs):
        calls["dataset"] = dataset
        calls["kwargs"] = kwargs
        return np.eye(2), return_time

    monkeypatch.setattr(interface_mod, api_name, _fake_solver)

    result = getattr(ds, method_name)(**call_kwargs)
    assert calls["dataset"] is ds
    for key, value in expected_kwargs.items():
        assert calls["kwargs"][key] == value
    assert isinstance(result, tuple)
