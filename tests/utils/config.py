"""Shared YAML settings loaders and kwargs builders for test configuration.

Low-level YAML access
---------------------
``get_dataset_params``  – dataset construction / source-selection / GMM fitting.
``get_ggml_params``     – GGML train/test/tune + perf-benchmark guard settings.

Convenience kwargs builders
---------------------------
``get_synth_config``  – synthetic dataset kwargs (``profile="smoke"`` or ``"perf"``).
``get_kwargs``        – GGML API kwargs via 3-layer merge (defaults → profile → suite → overrides).
``get_thresholds``    – performance regression thresholds (global defaults + suite overrides).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml


_TESTS_ROOT = Path(__file__).resolve().parent.parent
_DATASET_PARAMS_PATH = _TESTS_ROOT / "dataset_params.yml"
_GGML_PARAMS_PATH = _TESTS_ROOT / "ggml_params.yml"

_Profile = Literal["smoke", "perf"]
_Api = Literal["train", "test", "train_test", "tune"]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


@lru_cache(maxsize=1)
def get_dataset_params() -> dict[str, Any]:
    """Load dataset/fixture parameters (including GMM fitting presets)."""
    return _load_yaml(_DATASET_PARAMS_PATH)


@lru_cache(maxsize=1)
def get_ggml_params() -> dict[str, Any]:
    """Load GGML train/test/tune/perf parameters."""
    return _load_yaml(_GGML_PARAMS_PATH)


# ---------------------------------------------------------------------------
# Convenience kwargs builders – single source of truth for test calls.
# ---------------------------------------------------------------------------


def get_synth_config(profile: _Profile = "smoke") -> dict[str, Any]:
    """Synthetic dataset construction kwargs from ``dataset_params.yml``.

    Parameters
    ----------
    profile
        ``"smoke"`` for lightweight CI tests, ``"perf"`` for benchmark tests.
    """
    settings = get_dataset_params()["datasets"]["synthetic"][profile]
    return {
        **dict(settings["generator_kwargs"]),
        "distribution_size": int(settings["n_cells"]),
    }


def get_kwargs(
    profile: _Profile = "smoke",
    api: _Api = "train",
    *,
    suite: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build GGML API kwargs via 3-layer merge.

    Merge order::

        defaults.<api>  ←  <profile>.<api>  ←  <profile>.<suite>.<api>  ←  **overrides
             L0                  L1                      L2                      L3

    Parameters
    ----------
    profile
        ``"smoke"`` for lightweight CI tests, ``"perf"`` for benchmark tests.
    api
        Which GGML API the kwargs target: ``"train"``, ``"test"``,
        ``"train_test"``, or ``"tune"``.
    suite
        Perf-benchmark suite name (e.g. ``"empirical"``, ``"gmm"``,
        ``"minibatch"``).  Required when *profile* is ``"perf"``.
    **overrides
        Ad-hoc overrides applied as layer 3 (e.g. ``entropic_reg=0``).

    Raises
    ------
    ValueError
        If *profile* is ``"perf"`` and *api* is not ``"train_test"``, or if
        *suite* is missing for the ``"perf"`` profile.
    """
    if profile == "perf" and api != "train_test":
        raise ValueError(f"perf profile only supports api='train_test', got {api!r}")
    if profile == "perf" and suite is None:
        raise ValueError("perf profile requires suite=")

    params = get_ggml_params()

    # L0: defaults.<api>
    kw: dict[str, Any] = dict(params["defaults"].get(api) or {})
    # L1: <profile>.<api>
    kw.update(params[profile].get(api) or {})
    # L2: <profile>.<suite>.<api>
    if suite is not None:
        kw.update(params[profile][suite].get(api) or {})
    # L3: caller overrides
    kw.update(overrides)
    return kw


def get_thresholds(suite: str | None = None) -> dict[str, Any]:
    """Performance regression thresholds (global defaults + suite overrides).

    Parameters
    ----------
    suite
        If given, the suite-specific thresholds override the global defaults.
    """
    params = get_ggml_params()
    base: dict[str, Any] = dict(params["defaults"].get("thresholds") or {})
    if suite is not None:
        base.update(params["perf"][suite].get("thresholds") or {})
    return base


def get_variants(suite: str) -> dict[str, dict[str, Any]]:
    """GGML variant kwargs from ``ggml_params.yml → perf.<suite>.variants``.

    Parameters
    ----------
    suite
        Perf-benchmark suite name (e.g. ``"gmm"``).

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of variant name → GGML override kwargs.
    """
    params = get_ggml_params()
    raw = params["perf"][suite].get("variants") or {}
    return {name: dict(kw) for name, kw in raw.items()}


def build_snapshot_params(
    suite: str,
    *,
    solver: str,
    device: str,
    data_source: str,
    dataset_ref: str,
    n_threads: int,
    **extra: Any,
) -> dict[str, Any]:
    """Build ``snapshot_params`` dict for performance regression snapshots.

    Merges GGML ``train_test`` kwargs from the YAML config automatically,
    handles solver-specific ``entropic_reg``, and accepts additional keys
    (e.g. GMM-specific ``fit_*`` params) via ``**extra``.

    Parameters
    ----------
    suite
        Config suite name (``"empirical"``, ``"gmm"``, ``"minibatch"``).
    solver
        ``"emd2"`` or ``"sinkhorn"``.
    device
        ``"cpu"`` or ``"gpu"``.
    data_source
        ``"synthetic"`` or ``"network"``.
    dataset_ref
        Stable hash / identifier for the dataset.
    n_threads
        Thread count at test time.
    **extra
        Additional keys merged last (e.g. ``sharing``, ``variant``,
        ``fit_*`` keys for GMM snapshots).
    """
    kw = get_kwargs("perf", "train_test", suite=suite)
    entropic_reg = 0.0 if solver == "emd2" else float(kw.get("entropic_reg", 10.0))

    params: dict[str, Any] = dict(
        data_source=data_source,
        dataset_ref=dataset_ref,
        device=device,
        solver=solver,
        n_threads=n_threads,
        entropic_reg=entropic_reg,
        **{k: v for k, v in kw.items() if k != "entropic_reg"},
    )
    params.update(extra)
    return params
