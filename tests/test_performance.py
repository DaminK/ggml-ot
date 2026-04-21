"""Performance regression benchmark for GGML train_test.

Benchmarks CPU/GPU across EMD2 and Sinkhorn on synthetic/network sources.
Dataset source is selected via ``--data-source`` and surfaced as parametrized cases.
"""

from __future__ import annotations

from itertools import product
import warnings

import pytest
import torch

import ggml_ot

from .utils.config import get_kwargs, get_thresholds, build_snapshot_params
from .utils.setup_dataset import PERF_DATA_SOURCES, get_perf_setup_identity, make_perf_dataset
from .utils.performance_snapshot import evaluate_perf_case

TRAIN_TEST_KWARGS = get_kwargs("perf", "train_test", suite="empirical")
THRESHOLDS = get_thresholds("empirical")


def _selected_data_sources(config) -> list[str]:
    selected = config.getoption("--data-source")
    return list(PERF_DATA_SOURCES) if selected == "all" else [selected]


def pytest_generate_tests(metafunc):
    if "empirical_case" not in metafunc.fixturenames:
        return

    data_sources = _selected_data_sources(metafunc.config)
    cases = list(product(data_sources, ("emd2", "sinkhorn"), ("cpu", "gpu")))
    ids = [f"data: {src}, solver: {solver}, device: {device}" for src, solver, device in cases]
    metafunc.parametrize("empirical_case", cases, ids=ids)


def _run_benchmark(ds, *, solver: str) -> tuple[float, float]:
    """Run train_test and return (elapsed_time, knn_accuracy)."""
    if solver == "sinkhorn":
        ds.normalize()

    kw = dict(TRAIN_TEST_KWARGS)
    if solver == "emd2":
        kw["entropic_reg"] = 0.0

    _, score_df = ggml_ot.train_test(ds, **kw)
    acc = float(score_df["knn"]["mean"].iloc[0])
    elapsed = float(score_df["epoch_time(s)"]["mean"].iloc[0])
    return elapsed, acc


@pytest.mark.perf
@pytest.mark.filterwarnings("ignore::DeprecationWarning", match=r"Triggered internally at /pytorch/")
def test_performance_on_empirical(capsys, request, empirical_case: tuple[str, str, str]):
    """Run one matrix case and compare against its snapshot baseline."""
    data_source, solver, device = empirical_case

    if device == "gpu" and not torch.cuda.is_available():
        with capsys.disabled():
            print(f"\n[perf] data={data_source} solver={solver} device=gpu: CUDA not available, skipping")
        pytest.skip("CUDA not available")

    ggml_ot.settings.device = "cuda:0" if device == "gpu" else "cpu"
    ds = make_perf_dataset(data_source=data_source)
    if device == "gpu":
        _ = torch.zeros(1, device="cuda:0")  # warm-up

    # Exact EMD2 on CUDA is expected to warn that OT solve is CPU-bound.
    if device == "gpu" and solver == "emd2":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Exact EMD2 solver is CPU-bound; no GPU acceleration is used for OT solves\.",
                category=UserWarning,
            )
            elapsed, acc = _run_benchmark(ds, solver=solver)
    else:
        elapsed, acc = _run_benchmark(ds, solver=solver)

    _, dataset_ref = get_perf_setup_identity(request, data_source=data_source)
    snapshot_params = build_snapshot_params(
        "empirical",
        solver=solver,
        device=device,
        data_source=data_source,
        dataset_ref=dataset_ref,
        n_threads=ggml_ot.settings.n_threads,
    )

    current_scores = {"knn": acc}
    current_times = {"epoch_time(s)": elapsed}

    with capsys.disabled():
        evaluate_perf_case(
            request=request,
            suite="core_perf",
            model="empirical",
            context={"data": data_source, "solver": solver, "device": device},
            log_metrics={"epoch_time(s)": elapsed, "knn": acc},
            snapshot_params=snapshot_params,
            current_scores=current_scores,
            current_times=current_times,
            thresholds=THRESHOLDS,
        )
