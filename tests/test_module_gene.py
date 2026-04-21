"""Gene analysis utility tests.

Scope
-----
Tests `ggml_ot.gene.ranking._ranking` input validation behavior.

"""

import numpy as np
import pytest
import importlib

from anndata import AnnData

ranking_mod = importlib.import_module("ggml_ot.gene.ranking")
_ranking = ranking_mod._ranking


@pytest.mark.anndata
def test_ranking_validates_component_indices():
    """Invalid component indices should raise a ValueError."""
    adata = AnnData(X=np.zeros((3, 4), dtype=np.float32))
    adata.varm["W_ggml"] = np.ones((4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"greater than zero"):
        _ranking(adata, components="0")


@pytest.mark.anndata
def test_ranking_validates_n_genes():
    """Requesting too many genes should raise a ValueError."""
    adata = AnnData(X=np.zeros((3, 4), dtype=np.float32))
    adata.varm["W_ggml"] = np.ones((4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"Tried to plot"):
        _ranking(adata, components="1", n_genes=999)


@pytest.mark.anndata
def test_ranking_validates_polarity():
    """Invalid polarity names should raise a ValueError."""
    adata = AnnData(X=np.zeros((3, 4), dtype=np.float32))
    adata.varm["W_ggml"] = np.ones((4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"polarity"):
        _ranking(adata, components="1", polarity="minimum")


@pytest.mark.anndata
def test_ranking_returns_shown_genes(monkeypatch):
    """Returned genes should match the ranking plot order."""
    adata = AnnData(X=np.zeros((3, 5), dtype=np.float32))
    adata.var_names = ["pos_1", "neg_1", "pos_2", "neg_2", "mid"]
    adata.varm["W_ggml"] = np.array([[4.0], [-3.0], [2.0], [-5.0], [1.0]], dtype=np.float32)

    monkeypatch.setattr(ranking_mod, "ranking", lambda *args, **kwargs: None)
    monkeypatch.setattr(ranking_mod, "savefig_or_show", lambda *args, **kwargs: None)

    top_genes = _ranking(adata, components="1", polarity="both", n_genes=4, show=False)

    np.testing.assert_array_equal(top_genes, [["pos_1", "pos_2", "neg_1", "neg_2"]])
