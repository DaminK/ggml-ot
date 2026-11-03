"""Tests for curated dataset shortcut wrappers."""

from __future__ import annotations

import scanpy as sc

from ggml_ot.data import datasets


class _FakeDataset:
    def __init__(self):
        self.normalized = False

    def normalize(self):
        self.normalized = True
        return self


def test_curated_shortcut_applies_hvg_and_normalize(monkeypatch):
    """Shortcut wrappers should keep loading simple while exposing common prep knobs."""
    adata = object()
    fake_dataset = _FakeDataset()
    calls = {}

    def fake_load_cellxgene(dataset_id, *, url, path):
        calls["load"] = {"dataset_id": dataset_id, "url": url, "path": path}
        return adata

    def fake_highly_variable_genes(received_adata, *, n_top_genes, subset):
        calls["hvg"] = {
            "adata": received_adata,
            "n_top_genes": n_top_genes,
            "subset": subset,
        }

    def fake_from_anndata(received_adata, **kwargs):
        calls["from_anndata"] = {"adata": received_adata, "kwargs": kwargs}
        return fake_dataset

    monkeypatch.setattr(datasets, "load_cellxgene", fake_load_cellxgene)
    monkeypatch.setattr(datasets, "from_anndata", fake_from_anndata)
    monkeypatch.setattr(sc.pp, "highly_variable_genes", fake_highly_variable_genes)

    result = datasets.myocardial_infarction(
        path="/tmp/cache",
        n_top_genes=123,
        normalize=True,
        n_cells=50,
    )

    assert result is fake_dataset
    assert fake_dataset.normalized
    assert calls["load"]["dataset_id"] == "c1f6034b-7973-45e1-85e7-16933d0550bc.h5ad"
    assert calls["load"]["path"] == "/tmp/cache"
    assert calls["hvg"] == {"adata": adata, "n_top_genes": 123, "subset": True}
    assert calls["from_anndata"]["adata"] is adata
    assert calls["from_anndata"]["kwargs"] == {
        "patient_col": "sample",
        "label_col": "patient_group",
        "label_type": "categorical",
        "n_cells": 50,
    }


def test_curated_shortcut_defaults_do_not_preprocess_or_normalize(monkeypatch):
    """Existing shortcut behavior remains unchanged by default."""
    adata = object()
    fake_dataset = _FakeDataset()
    calls = {"hvg": 0}

    monkeypatch.setattr(datasets, "load_cellxgene", lambda *args, **kwargs: adata)
    monkeypatch.setattr(datasets, "from_anndata", lambda *args, **kwargs: fake_dataset)
    monkeypatch.setattr(
        sc.pp,
        "highly_variable_genes",
        lambda *args, **kwargs: calls.__setitem__("hvg", 1),
    )

    result = datasets.covid()

    assert result is fake_dataset
    assert not fake_dataset.normalized
    assert calls["hvg"] == 0
