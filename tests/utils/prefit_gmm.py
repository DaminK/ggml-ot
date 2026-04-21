"""Reusable pre-fitted GMM fixtures for performance tests."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

import torch

import ggml_ot
from .setup_dataset import get_gmm_k_candidates, make_perf_anndata

_PREFIT_GMM_CACHE: dict[str, dict[str, Any]] = {}


def _adapt_k_comps(k_candidates: list[int], *, max_k: int) -> int | list[int]:
    """Adapt component candidates to available cells-per-distribution."""
    valid = sorted({int(k) for k in k_candidates if 1 <= int(k) <= int(max_k)})
    if len(valid) == 0:
        return 1
    if len(valid) == 1:
        return int(valid[0])
    return valid


def _cache_key(
    *,
    data_source: str,
    device: str,
    component_sharing: str,
    max_cells_per_patient: int,
    use_rep: str | None,
    fit_gmm_kwargs: dict[str, Any],
) -> str:
    payload = {
        "data_source": data_source,
        "device": device,
        "component_sharing": component_sharing,
        "max_cells_per_patient": int(max_cells_per_patient),
        "use_rep": use_rep,
        "fit_gmm_kwargs": fit_gmm_kwargs,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def get_prefit_gmm_payload(
    *,
    data_source: str,
    device: str,
    component_sharing: str,
    max_cells_per_patient: int,
    use_rep: str | None,
    fit_gmm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return cached pre-fitted GMM dataset payload for one test matrix setup."""
    key = _cache_key(
        data_source=data_source,
        device=device,
        component_sharing=component_sharing,
        max_cells_per_patient=max_cells_per_patient,
        use_rep=use_rep,
        fit_gmm_kwargs=fit_gmm_kwargs,
    )
    if key not in _PREFIT_GMM_CACHE:
        ggml_ot.settings.device = "cuda:0" if device == "gpu" else "cpu"
        adata = make_perf_anndata(data_source=data_source)
        if device == "gpu":
            _ = torch.zeros(1, device="cuda:0")  # warm-up

        min_cells = int(adata.obs.groupby("sample").size().min())
        n_cells = min(min_cells, int(max_cells_per_patient))
        k_candidates = get_gmm_k_candidates(data_source)
        k_comps = _adapt_k_comps(k_candidates, max_k=n_cells)

        dataset = ggml_ot.from_anndata(
            adata.copy(),
            patient_col="sample",
            label_col="patient_group",
            use_rep=use_rep,
            n_cells=n_cells,
        )

        t0 = time.perf_counter()
        dataset = dataset.fit_gmm(
            component_sharing=component_sharing,
            k_comps=k_comps,
            gmm_key=f"gmm_perf_{component_sharing}_shared",
            **fit_gmm_kwargs,
        )
        gmm_fit_seconds = time.perf_counter() - t0
        _PREFIT_GMM_CACHE[key] = {
            "dataset": dataset,
            "n_cells": n_cells,
            "k_comps": k_comps,
            "use_rep": use_rep,
            "gmm_fit_seconds": gmm_fit_seconds,
        }

    cached = _PREFIT_GMM_CACHE[key]
    return {
        "dataset": copy.deepcopy(cached["dataset"]),
        "n_cells": int(cached["n_cells"]),
        "k_comps": cached["k_comps"],
        "use_rep": cached["use_rep"],
        "gmm_fit_seconds": float(cached["gmm_fit_seconds"]),
    }


__all__ = [
    "get_prefit_gmm_payload",
]
