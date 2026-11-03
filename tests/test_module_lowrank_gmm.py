"""Tests for the low-rank (MFA) GMM covariance type.

Covers the two constructor modes, the Woodbury E-step vs the materialized
full-covariance path, MFA fitting, persistence roundtrip, sampling, AIC/BIC
free-parameter counting, restart robustness, and an AnnData end-to-end path
that exercises the storage canonicalization. Also guards that ``diag``/``full``
behaviour is unchanged.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

import ggml_ot
from ggml_ot.gmm._GaussianMixture import GaussianMixture
from ggml_ot.gmm._fit_core import GMMFitConfig, free_parameter_count

pytestmark = [pytest.mark.gmm]


def _lowrank_clusters(*, d=12, k=3, q=2, n_per=200, seed=0):
    """Synthetic data with k low-rank-plus-noise Gaussian clusters."""
    rng = np.random.default_rng(seed)
    mus = rng.normal(0.0, 6.0, size=(k, d))
    blocks = []
    for j in range(k):
        loadings = rng.normal(0.0, 1.0, size=(d, q))
        z = rng.normal(0.0, 1.0, size=(n_per, q))
        noise = rng.normal(0.0, 0.3, size=(n_per, d))
        blocks.append(mus[j] + z @ loadings.T + noise)
    return np.concatenate(blocks).astype(np.float64)


def _fitted_lowrank_model(*, d=12, k=3, q=2, **kw):
    x = _lowrank_clusters(d=d, k=k, q=q, **kw)
    cfg = GMMFitConfig(
        n_components=k,
        covariance_type="lowrank",
        factor_rank=q,
        max_iter=50,
        n_init=1,
        eps=1e-4,
        psi_floor=1e-3,
    )
    return GaussianMixture.fit_from_numpy(x=x, config=cfg), x


# ---------------------------------------------------------------------------
# Constructor two-mode rule
# ---------------------------------------------------------------------------


def test_lowrank_fitting_registers_live_factors():
    model = GaussianMixture(n_components=3, n_features=8, covariance_type="lowrank", factor_rank=2)
    assert model.factor_rank == 2
    assert model.loadings is not None and tuple(model.loadings.shape) == (1, 3, 8, 2)
    assert model.psi is not None and tuple(model.psi.shape) == (1, 3, 8)
    assert tuple(model.var.shape) == (1, 3, 8, 8)


def test_lowrank_loaded_has_no_live_factors():
    var = np.eye(4)[None, None].repeat(2, axis=1) * 0.5  # (1, 2, 4, 4)
    var_t = torch.as_tensor(var, dtype=torch.float32)
    model = GaussianMixture(
        n_components=2,
        n_features=4,
        covariance_type="lowrank",
        var_init=var_t,
    )
    assert model.factor_rank is None
    assert model.loadings is None and model.psi is None
    assert tuple(model.var.shape) == (1, 2, 4, 4)


def test_lowrank_requires_factor_rank_or_var_init():
    with pytest.raises(ValueError, match="factor_rank.*fit.*var_init.*load"):
        GaussianMixture(n_components=2, n_features=4, covariance_type="lowrank")


@pytest.mark.parametrize("bad_q", [0, -1, 8, 9])
def test_lowrank_rejects_out_of_range_factor_rank(bad_q):
    with pytest.raises(ValueError, match="0 < factor_rank < n_features"):
        GaussianMixture(n_components=2, n_features=8, covariance_type="lowrank", factor_rank=bad_q)


# ---------------------------------------------------------------------------
# Fitting + materialized covariance properties
# ---------------------------------------------------------------------------


def test_lowrank_fit_materializes_finite_symmetric_spd():
    result, _ = _fitted_lowrank_model()
    var = np.asarray(result.var, dtype=np.float64)
    assert var.shape == (1, 3, 12, 12)
    mats = var[0]
    assert np.all(np.isfinite(mats))
    assert np.allclose(mats, mats.transpose(0, 2, 1), atol=1e-5)
    min_eig = min(float(np.linalg.eigvalsh(m).min()) for m in mats)
    assert min_eig > 0.0
    assert np.isfinite(result.bic)


def test_lowrank_fit_recovers_cluster_assignments():
    result, x = _fitted_lowrank_model(n_per=200)
    hard = result.model.predict_hard_components_numpy(x)
    # Each of the 3 contiguous 200-sample blocks should be dominated by one component.
    for block in range(3):
        labels = hard[block * 200 : (block + 1) * 200]
        majority = np.bincount(labels, minlength=3).max()
        assert majority > 150


# ---------------------------------------------------------------------------
# Woodbury E-step correctness
# ---------------------------------------------------------------------------


def test_woodbury_log_prob_matches_exact_gaussian():
    # The Woodbury log-density must equal the exact log-density of
    # Sigma_k = Lambda_k Lambda_k^T + diag(Psi_k) for each component. Use
    # float64 so the comparison is tight (the materialized-var scoring path,
    # tested separately, adds an O(eps) jitter and is only approximate).
    torch.manual_seed(0)
    d, k, q = 10, 3, 3
    model = GaussianMixture(n_components=k, n_features=d, covariance_type="lowrank", factor_rank=q).double()
    model.mu.data = torch.randn(1, k, d, dtype=torch.float64)
    model.loadings.data = 0.5 * torch.randn(1, k, d, q, dtype=torch.float64)
    model.psi.data = torch.rand(1, k, d, dtype=torch.float64) + 0.2

    x = torch.randn(40, d, dtype=torch.float64)
    woodbury = model._woodbury_log_prob(model.check_size(x)).squeeze(-1)  # (N, K)

    eye = torch.eye(d, dtype=torch.float64)
    reference = torch.empty(x.shape[0], k, dtype=torch.float64)
    for j in range(k):
        sigma = model.loadings.data[0, j] @ model.loadings.data[0, j].T + torch.diag(model.psi.data[0, j])
        mvn = torch.distributions.MultivariateNormal(model.mu.data[0, j], covariance_matrix=sigma + 0.0 * eye)
        reference[:, j] = mvn.log_prob(x)
    assert torch.allclose(woodbury, reference, atol=1e-8, rtol=1e-6)


def test_loaded_scoring_approximates_live_woodbury():
    # A loaded model scores via the stored full covariance, whose inversion adds
    # an O(eps) jitter, so it only approximates the live Woodbury scoring. Hard
    # assignments must be identical; scores close to eps.
    result, x = _fitted_lowrank_model()
    loaded = GaussianMixture.from_dict(result.model.to_dict())
    xt = torch.as_tensor(x, dtype=torch.float32)

    live = result.model.score_samples(xt).detach().numpy()
    reloaded = loaded.score_samples(xt).detach().numpy()
    assert np.allclose(live, reloaded, atol=5e-2, rtol=1e-3)
    np.testing.assert_array_equal(
        result.model.predict_hard_components_numpy(x),
        loaded.predict_hard_components_numpy(x),
    )


# ---------------------------------------------------------------------------
# Free-parameter count (AIC/BIC)
# ---------------------------------------------------------------------------


def test_free_parameter_count_lowrank_formula():
    k, d, q = 4, 20, 3
    expected = (k - 1) + 2 * k * d + k * d * q - k * q * (q - 1) // 2
    assert free_parameter_count(k, d, "lowrank", factor_rank=q) == expected


def test_free_parameter_count_lowrank_requires_rank():
    with pytest.raises(ValueError, match="factor_rank"):
        free_parameter_count(3, 10, "lowrank")


def test_lowrank_bic_is_finite_and_uses_lowrank_count():
    result, x = _fitted_lowrank_model()
    n = x.shape[0]
    ll = result.model.score_samples(torch.as_tensor(x, dtype=torch.float32)).mean().item()
    p = free_parameter_count(3, 12, "lowrank", factor_rank=2)
    expected_bic = -2.0 * ll * n + p * np.log(n)
    got = float(result.model.bic(torch.as_tensor(x, dtype=torch.float32)).item())
    assert np.isfinite(got)
    assert got == pytest.approx(expected_bic, rel=1e-4)


# ---------------------------------------------------------------------------
# Persistence + sampling
# ---------------------------------------------------------------------------


def test_lowrank_to_from_dict_roundtrip_drops_factors():
    result, _ = _fitted_lowrank_model()
    payload = result.model.to_dict()
    assert payload["covariance_type"] == "lowrank"
    assert np.asarray(payload["var"]).shape == (3, 12, 12)

    loaded = GaussianMixture.from_dict(payload)
    assert loaded.factor_rank is None  # loaded from stored covariance, no factors
    assert loaded.loadings is None and loaded.psi is None
    assert tuple(loaded.var.shape) == (1, 3, 12, 12)


def test_lowrank_sample_returns_expected_shape():
    result, _ = _fitted_lowrank_model()
    sampled, labels = result.model.sample(25)
    assert tuple(sampled.shape) == (25, 12)
    assert int(labels.shape[0]) == 25


# ---------------------------------------------------------------------------
# Restart robustness
# ---------------------------------------------------------------------------


def test_reinitialize_after_failure_preserves_lowrank_config():
    model = GaussianMixture(n_components=3, n_features=12, covariance_type="lowrank", factor_rank=2, psi_floor=1e-3)
    x = torch.as_tensor(_lowrank_clusters(), dtype=torch.float32)
    model._global_var = x.var(dim=0, unbiased=False)

    model._reinitialize_after_failure(x)

    assert model.factor_rank == 2
    assert model.psi_floor == 1e-3
    assert model._global_var is not None
    assert model.loadings is not None and tuple(model.loadings.shape) == (1, 3, 12, 2)
    assert tuple(model.var.shape) == (1, 3, 12, 12)


# ---------------------------------------------------------------------------
# AnnData end-to-end (exercises storage canonicalization)
# ---------------------------------------------------------------------------


def _lowrank_adata(seed=0):
    rng = np.random.default_rng(seed)
    blocks, samples = [], []
    for j, loc in enumerate((0.0, 4.0, -4.0)):
        loadings = rng.normal(0.0, 1.0, size=(5, 2))
        z = rng.normal(0.0, 1.0, size=(80, 2))
        blocks.append(loc + z @ loadings.T + rng.normal(0.0, 0.2, size=(80, 5)))
        samples += [f"s{j + 1}"] * 80
    x = np.vstack(blocks).astype(np.float64)
    obs = pd.DataFrame({"sample": samples, "patient_group": ["A"] * len(samples)})
    obs.index = obs.index.astype(str)
    return AnnData(X=x, obs=obs)


def test_parse_factor_rank():
    from ggml_ot.gmm._select_k_comps import parse_factor_rank

    assert parse_factor_rank(10) == (10, None)
    assert parse_factor_rank(None) == (None, None)
    assert parse_factor_rank([7]) == (7, None)  # single-element list collapses to fixed
    assert parse_factor_rank([5, 10, 15]) == (None, [5, 10, 15])
    with pytest.raises(ValueError, match="positive integers"):
        parse_factor_rank([0, 2])


@pytest.mark.anndata
def test_lowrank_factor_rank_grid_selects_q():
    adata = _lowrank_adata()
    dataset = ggml_ot.from_anndata(adata, patient_col="sample", label_col="patient_group", n_cells=80)
    dataset.fit_gmm(
        k_comps=2,
        component_sharing="global",
        covariance_type="lowrank",
        factor_rank=[1, 2, 3],
        psi_floor=1e-3,
        k_selection_metric="heldout_nll",
        gmm_key="gmm_lowrank_q",
    )
    cfg = adata.uns["gmm_lowrank_q"]
    selection = cfg["selection"]
    assert selection["factor_rank_candidates"] == [1, 2, 3]
    best_q = int(selection["best_factor_rank"])
    assert best_q in (1, 2, 3)
    assert len(selection["factor_rank_scores"]["score"]) == 3
    # The persisted scalar rank matches the selected one.
    assert int(cfg["fit_params"]["factor_rank"]) == best_q


def test_factor_rank_list_requires_lowrank():
    adata = _lowrank_adata()
    dataset = ggml_ot.from_anndata(adata, patient_col="sample", label_col="patient_group", n_cells=80)
    with pytest.raises(ValueError, match="factor_rank may only be a list"):
        dataset.fit_gmm(
            k_comps=2,
            component_sharing="global",
            covariance_type="full",
            factor_rank=[1, 2, 3],
            gmm_key="gmm_bad",
        )


@pytest.mark.anndata
def test_lowrank_fit_gmm_anndata_end_to_end():
    adata = _lowrank_adata()
    dataset = ggml_ot.from_anndata(adata, patient_col="sample", label_col="patient_group", n_cells=80)
    dataset.fit_gmm(
        k_comps=1,
        component_sharing="global",
        covariance_type="lowrank",
        factor_rank=2,
        psi_floor=1e-3,
        refit="full",
        gmm_key="gmm_lowrank",
    )
    assert "gmm_lowrank" in adata.uns
    cfg = adata.uns["gmm_lowrank"]
    assert cfg["covariance_type"] == "lowrank"
    assert int(cfg["fit_params"]["factor_rank"]) == 2
    assert float(cfg["fit_params"]["psi_floor"]) == 1e-3
    var = np.asarray(cfg["model"]["var"], dtype=np.float64)
    # materialized full covariance (K, D, D) was stored and canonicalized
    assert var.shape[-1] == var.shape[-2] == adata.n_vars
    assert np.all(np.isfinite(var))

    # Reload roundtrip through the AnnData schema.
    reloaded = ggml_ot.from_anndata(adata, patient_col="sample", label_col="patient_group", gmm_key="gmm_lowrank")
    assert reloaded is not None


# ---------------------------------------------------------------------------
# Regression guard: diag / full unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cov_type", ["diag", "full"])
def test_diag_and_full_still_fit(cov_type):
    x = _lowrank_clusters(d=6, k=3, q=2, n_per=120)
    cfg = GMMFitConfig(n_components=3, covariance_type=cov_type, max_iter=40, n_init=1)
    result = GaussianMixture.fit_from_numpy(x=x, config=cfg)
    assert np.all(np.isfinite(np.asarray(result.var)))
    assert np.isfinite(result.bic)
    if cov_type == "full":
        assert np.asarray(result.var).shape == (1, 3, 6, 6)
    else:
        assert np.asarray(result.var).shape == (1, 3, 6)
