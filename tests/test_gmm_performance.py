"""Performance/timing benchmarks for downstream GMM + GGML workflows."""

from __future__ import annotations

from itertools import product
import warnings

import pytest
import torch

import ggml_ot

from .utils.config import get_dataset_params, get_kwargs, get_thresholds, get_variants, build_snapshot_params
from .utils.prefit_gmm import get_prefit_gmm_payload
from .utils.setup_dataset import (
    PERF_DATA_SOURCES,
    get_perf_setup_identity,
)
from .utils.performance_snapshot import evaluate_perf_case


pytestmark = [pytest.mark.gmm, pytest.mark.perf, pytest.mark.anndata]

_DATASET_PARAMS = get_dataset_params()
_GMM_DATASET_SETTINGS = _DATASET_PARAMS["gmm_fitting"]

MAX_CELLS_PER_PATIENT = int(_GMM_DATASET_SETTINGS["max_cells_per_patient"])
GMM_SHARING = str(_GMM_DATASET_SETTINGS["component_sharing"])
GMM_USE_REP_BY_SOURCE = dict(_GMM_DATASET_SETTINGS["use_rep_by_data_source"])
GMM_VARIANTS = get_variants("gmm")
GMM_FIT_KWARGS = dict(_GMM_DATASET_SETTINGS["fit_gmm_kwargs"])
TRAIN_TEST_KWARGS = get_kwargs("perf", "train_test", suite="gmm")
THRESHOLDS = get_thresholds("gmm")
SQUARED_GROUND_COST_OPTIONS = (True, False)


def _selected_data_sources(config) -> list[str]:
    selected = config.getoption("--data-source")
    return list(PERF_DATA_SOURCES) if selected == "all" else [selected]


def pytest_generate_tests(metafunc):
    if "gmm_case" not in metafunc.fixturenames:
        return

    data_sources = _selected_data_sources(metafunc.config)
    cases = list(
        product(
            data_sources,
            tuple(GMM_VARIANTS.keys()),
            SQUARED_GROUND_COST_OPTIONS,
            ("emd2", "sinkhorn"),
            ("cpu", "gpu"),
        )
    )
    ids = [
        f"data: {src}, variant: {variant}, mean_dist: {'squared' if squared_ground_cost else 'unsquared'}, "
        f"solver: {solver}, device: {device}"
        for src, variant, squared_ground_cost, solver, device in cases
    ]
    metafunc.parametrize("gmm_case", cases, ids=ids)


def _use_rep_for_source(data_source: str) -> str | None:
    if data_source not in GMM_USE_REP_BY_SOURCE:
        raise ValueError(f"Missing gmm_fitting.use_rep_by_data_source entry for data_source={data_source!r}")
    return GMM_USE_REP_BY_SOURCE[data_source]


@pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    match=r"__array__ implementation doesn't accept a copy keyword, so passing copy=False failed.",
)
def test_performance_on_gmm(capsys, request, gmm_case: tuple[str, str, bool, str, str]):
    """Run one GMM perf matrix case with snapshot regression checks."""
    data_source, variant_name, squared_ground_cost, solver, device = gmm_case
    variant_kwargs = GMM_VARIANTS[variant_name]
    mean_dist_name = "squared" if squared_ground_cost else "unsquared"

    if device == "gpu" and not torch.cuda.is_available():
        with capsys.disabled():
            print(
                f"\n[gmm perf] data={data_source}, variant={variant_name}, mean_dist={mean_dist_name}, "
                f"solver={solver}, device=gpu: CUDA not available, skipping"
            )
        pytest.skip("CUDA not available")

    # NOTE: intentionally do not skip unstable solver/variant combinations.
    # Perf tests should expose instability instead of hiding it.

    cached = get_prefit_gmm_payload(
        data_source=data_source,
        device=device,
        component_sharing=GMM_SHARING,
        max_cells_per_patient=MAX_CELLS_PER_PATIENT,
        use_rep=_use_rep_for_source(data_source),
        fit_gmm_kwargs=GMM_FIT_KWARGS,
    )
    dataset = cached["dataset"]
    n_cells = cached["n_cells"]
    k_comps = cached["k_comps"]
    use_rep = cached["use_rep"]
    gmm_fit_seconds = cached["gmm_fit_seconds"]

    kw = {**TRAIN_TEST_KWARGS, **variant_kwargs, "squared_ground_cost": squared_ground_cost}
    if solver == "emd2":
        kw["entropic_reg"] = 0.0

    if device == "gpu" and solver == "emd2":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Exact EMD2 solver is CPU-bound; no GPU acceleration is used for OT solves\.",
                category=UserWarning,
            )
            _, score_df = dataset.train_test(**kw)
    else:
        _, score_df = dataset.train_test(**kw)

    inf_time_ms = float(score_df.iloc[0][("inf_time(ms)", "mean")])
    # Synthetic GMM baselines may have modest absolute knn_acc; regressions are baseline-relative in A2.
    knn_acc = float(score_df.iloc[0][("knn", "mean")])
    epoch_time_seconds = float(score_df.iloc[0][("epoch_time(s)", "mean")])

    _, dataset_ref = get_perf_setup_identity(request, data_source=data_source)
    snapshot_params = build_snapshot_params(
        "gmm",
        solver=solver,
        device=device,
        data_source=data_source,
        dataset_ref=dataset_ref,
        n_threads=ggml_ot.settings.n_threads,
        # GMM-specific extra keys (dataset-side, not GGML API kwargs).
        sharing=GMM_SHARING,
        variant=variant_name,
        squared_ground_cost=squared_ground_cost,
        n_cells=n_cells,
        k_comps=k_comps,
        fit_k_selection_metric=GMM_FIT_KWARGS["k_selection_metric"],
        fit_refit=GMM_FIT_KWARGS.get("refit", "full"),
        fit_covariance_type=GMM_FIT_KWARGS["covariance_type"],
        # Preserve the historical snapshot field name for backward compatibility.
        fit_subsample_frac=GMM_FIT_KWARGS["train_size"],
        fit_max_iter=GMM_FIT_KWARGS["max_iter"],
        fit_use_rep=use_rep,
    )

    current_scores = {"knn": knn_acc}
    current_times = {
        "gmm_fit_time(s)": gmm_fit_seconds,
        "epoch_time(s)": epoch_time_seconds,
        "inf_time(s)": inf_time_ms / 1000.0,
    }

    with capsys.disabled():
        evaluate_perf_case(
            request=request,
            suite="gmm_perf",
            model="gmm",
            context={
                "data": data_source,
                "variant": variant_name,
                "mean_dist": mean_dist_name,
                "solver": solver,
                "device": device,
            },
            log_metrics={"epoch_time(s)": epoch_time_seconds, "knn": knn_acc},
            optional_log_metrics={"gmm_fit_time(s)": gmm_fit_seconds, "inf_time(ms)": inf_time_ms},
            snapshot_params=snapshot_params,
            current_scores=current_scores,
            current_times=current_times,
            thresholds=THRESHOLDS,
        )
