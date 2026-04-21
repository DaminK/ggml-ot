"""Shared test dataset setup utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, cast

import scanpy as sc

import ggml_ot
from .config import get_dataset_params

_DATASET_PARAMS = get_dataset_params()
_DATASET_SETTINGS = _DATASET_PARAMS["datasets"]
_GMM_FITTING_SETTINGS = _DATASET_PARAMS["gmm_fitting"]
_ANNDATA_PROFILES = ("smoke", "perf")

PERF_DATA_SOURCES = tuple(_DATASET_PARAMS["data_sources"])
_GMM_K_CANDIDATES_BY_SOURCE = _GMM_FITTING_SETTINGS["k_comps_candidates_by_data_source"]
SYNTH_GMM_K_CANDIDATES = [int(k) for k in _GMM_K_CANDIDATES_BY_SOURCE["synthetic"]]
NETWORK_GMM_K_CANDIDATES = [int(k) for k in _GMM_K_CANDIDATES_BY_SOURCE["network"]]
_SYNTH_SMOKE_BASE_CONFIG: dict[str, Any] = dict(_DATASET_SETTINGS["synthetic"]["smoke"]["generator_kwargs"])
_SYNTH_PERF_BASE_CONFIG: dict[str, Any] = dict(_DATASET_SETTINGS["synthetic"]["perf"]["generator_kwargs"])
_NETWORK_ANNDATA_CACHE: dict[tuple[str, bool], Any] = {}
_PERF_ANNDATA_CACHE: dict[str, Any] = {}


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _get_data_source(request) -> str:
    source = request.config.getoption("--data-source")
    if source == "all":
        raise ValueError(
            "For --data-source=all, pass an explicit source to setup helpers "
            "(e.g. via indirect fixture parametrization)."
        )
    if source not in PERF_DATA_SOURCES:
        raise ValueError(f"Unknown data source: {source}. Expected one of {PERF_DATA_SOURCES}.")
    return cast(str, source)


def _validate_profile(profile: str) -> Literal["smoke", "perf"]:
    if profile not in _ANNDATA_PROFILES:
        raise ValueError(f"Unknown AnnData profile: {profile}. Expected one of {_ANNDATA_PROFILES}.")
    return cast(Literal["smoke", "perf"], profile)


def _profile_settings(data_source: str, *, profile: Literal["smoke", "perf"]) -> dict[str, Any]:
    source_settings = _DATASET_SETTINGS.get(data_source)
    if not isinstance(source_settings, dict):
        raise ValueError(f"Missing dataset config for data_source={data_source}")
    profile_key = profile
    profile_settings = source_settings.get(profile_key)
    if not isinstance(profile_settings, dict):
        raise ValueError(f"Missing profile config for data_source={data_source}, profile={profile_key}")
    return profile_settings


def _n_cells_for_profile(data_source: str, *, profile: Literal["smoke", "perf"]) -> int:
    profile_settings = _profile_settings(data_source, profile=profile)
    n_cells = profile_settings.get("n_cells")
    if not isinstance(n_cells, int):
        raise ValueError(f"Missing integer n_cells for data_source={data_source}, profile={profile}")
    return int(n_cells)


def _network_dataset_id() -> str:
    network_settings = _DATASET_SETTINGS.get("network")
    dataset_id = network_settings.get("dataset_id") if isinstance(network_settings, dict) else None
    if not isinstance(dataset_id, str):
        raise ValueError("Missing global network dataset_id in tests/dataset_params.yml")
    return dataset_id


def _synth_smoke_config(*, n_cells: int) -> dict[str, Any]:
    return {**_SYNTH_SMOKE_BASE_CONFIG, "distribution_size": int(n_cells)}


def _synth_perf_config(*, n_cells: int) -> dict[str, Any]:
    return {**_SYNTH_PERF_BASE_CONFIG, "distribution_size": int(n_cells)}


def _load_network_anndata(*, dataset_id: str, apply_hvg: bool) -> Any:
    cache_key = (dataset_id, apply_hvg)
    if cache_key not in _NETWORK_ANNDATA_CACHE:
        adata = ggml_ot.data.load_cellxgene(dataset_id)
        if apply_hvg:
            sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True)
        _NETWORK_ANNDATA_CACHE[cache_key] = adata
    return _NETWORK_ANNDATA_CACHE[cache_key].copy()


def _build_smoke_datasets(adata, *, n_cells: int) -> dict[str, Any]:
    """Build standard AnnData dataset variants for smoke/integration tests."""
    return {
        "default": ggml_ot.from_anndata(adata, n_cells=n_cells),
        "use_rep": ggml_ot.from_anndata(adata, use_rep="X_pca", n_cells=n_cells),
        "group_by": ggml_ot.from_anndata(adata, group_by="cell_type"),
    }


def _build_anndata(data_source: str, *, profile: Literal["smoke", "perf"]) -> tuple[Any, dict[str, str]]:
    """Build shared AnnData payload for a given profile/source pair."""
    if data_source == "network":
        dataset_id = _network_dataset_id()
        if profile == "smoke":
            print("Loading dataset from CellxGene...")
        else:
            print("Loading full AnnData dataset from CellxGene for perf tests...")
        adata = _load_network_anndata(dataset_id=dataset_id, apply_hvg=(profile == "smoke"))
        metadata = {"data_source": "network", "dataset_ref": dataset_id}
        return adata, metadata

    if profile == "smoke":
        print("Using synthetic anndata test data...")
        synth_config = _synth_smoke_config(n_cells=_n_cells_for_profile("synthetic", profile="smoke"))
        synth_generator = "synth_anndata_smoke"
    else:
        print("Using synthetic AnnData perf data (from_synth-aligned config)...")
        synth_config = _synth_perf_config(n_cells=_n_cells_for_profile("synthetic", profile="perf"))
        synth_generator = "synth_anndata_perf"

    adata = ggml_ot.data.synth_anndata(**synth_config)
    metadata = {
        "data_source": "synthetic",
        "dataset_ref": _stable_hash({"generator": synth_generator, **synth_config}),
    }
    return adata, metadata


def get_anndata_registry(
    request,
    *,
    profile: Literal["smoke", "perf"] = "smoke",
    data_source: str | None = None,
) -> dict[str, Any]:
    """Return shared AnnData registry for the requested test profile.

    `smoke` profile returns both AnnData and prebuilt dataset variants.
    `perf` profile returns only AnnData + metadata.
    """
    profile = _validate_profile(profile)
    resolved_source = _get_data_source(request) if data_source is None else data_source
    if resolved_source not in PERF_DATA_SOURCES:
        raise ValueError(f"Unknown data source: {resolved_source}. Expected one of {PERF_DATA_SOURCES}.")
    adata, metadata = _build_anndata(resolved_source, profile=profile)
    registry: dict[str, Any] = {"adata": adata, **metadata}
    if profile == "smoke":
        n_cells = _n_cells_for_profile(resolved_source, profile="smoke")
        registry["datasets"] = _build_smoke_datasets(adata, n_cells=n_cells)
    return registry


def get_perf_setup_identity(request, n_cells: int | None = None, data_source: str | None = None) -> tuple[str, str]:
    """Return ``(data_source, dataset_ref)`` used for perf snapshot keys."""
    resolved_source = _get_data_source(request) if data_source is None else data_source
    if resolved_source not in PERF_DATA_SOURCES:
        raise ValueError(f"Unknown data source: {resolved_source}. Expected one of {PERF_DATA_SOURCES}.")
    resolved_n_cells = _n_cells_for_profile(resolved_source, profile="perf") if n_cells is None else int(n_cells)
    if resolved_source == "network":
        return "network", _network_dataset_id()
    payload = {**_synth_perf_config(n_cells=resolved_n_cells), "generator": "from_synth_default"}
    return "synthetic", _stable_hash(payload)


def make_perf_dataset(data_source: str, n_cells: int | None = None):
    """Build a perf dataset for the requested source."""
    resolved_n_cells = _n_cells_for_profile(data_source, profile="perf") if n_cells is None else int(n_cells)
    if data_source == "network":
        dataset_id = _network_dataset_id()
        adata = _load_network_anndata(dataset_id=dataset_id, apply_hvg=True)
        return ggml_ot.from_anndata(adata, n_cells=resolved_n_cells)
    synth_kwargs = _synth_perf_config(n_cells=resolved_n_cells)
    return ggml_ot.data.from_synth(**synth_kwargs, show=False)


def make_perf_anndata(data_source: str):
    """Build an AnnData payload for GMM perf tests using shared source setup."""
    if data_source not in _PERF_ANNDATA_CACHE:
        adata, _ = _build_anndata(data_source, profile="perf")
        _PERF_ANNDATA_CACHE[data_source] = adata
    return _PERF_ANNDATA_CACHE[data_source].copy()


def get_gmm_k_candidates(data_source: str) -> list[int]:
    """Return the agreed K-candidate presets for each perf data source."""
    if data_source == "synthetic":
        return list(SYNTH_GMM_K_CANDIDATES)
    if data_source == "network":
        return list(NETWORK_GMM_K_CANDIDATES)
    raise ValueError(f"Unknown data_source: {data_source}")
