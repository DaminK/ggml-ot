"""Mini-batch performance regression benchmark.

Validates that training with ``batch_size < N`` (stratified batching) still
produces comparable accuracy and timing relative to the full-batch baseline.

Dataset source is selected via ``--data-source`` (same as other perf tests).
Configuration lives in ``ggml_params.yml → perf.minibatch``.
"""

from __future__ import annotations

import pytest

import ggml_ot

from .utils.config import get_kwargs, get_thresholds, build_snapshot_params
from .utils.setup_dataset import PERF_DATA_SOURCES, get_perf_setup_identity, make_perf_dataset
from .utils.performance_snapshot import evaluate_perf_case

TRAIN_TEST_KWARGS = get_kwargs("perf", "train_test", suite="minibatch")
THRESHOLDS = get_thresholds("minibatch")


def _selected_data_sources(config) -> list[str]:
    selected = config.getoption("--data-source")
    return list(PERF_DATA_SOURCES) if selected == "all" else [selected]


def pytest_generate_tests(metafunc):
    if "minibatch_case" not in metafunc.fixturenames:
        return

    data_sources = _selected_data_sources(metafunc.config)
    ids = [f"data: {src}" for src in data_sources]
    metafunc.parametrize("minibatch_case", data_sources, ids=ids)


@pytest.mark.perf
@pytest.mark.filterwarnings("ignore::DeprecationWarning", match=r"Triggered internally at /pytorch/")
def test_minibatch_perf_on_empirical(capsys, request, minibatch_case: str):
    """Run one mini-batch case and compare against its snapshot baseline."""
    data_source = minibatch_case

    ggml_ot.settings.device = "cpu"
    ds = make_perf_dataset(data_source=data_source)

    # emd2: override entropic_reg to 0
    kw = {**TRAIN_TEST_KWARGS, "entropic_reg": 0.0}
    _, score_df = ggml_ot.train_test(ds, **kw)
    acc = float(score_df["knn"]["mean"].iloc[0])
    elapsed = float(score_df["epoch_time(s)"]["mean"].iloc[0])

    _, dataset_ref = get_perf_setup_identity(request, data_source=data_source)
    snapshot_params = build_snapshot_params(
        "minibatch",
        solver="emd2",
        device="cpu",
        data_source=data_source,
        dataset_ref=dataset_ref,
        n_threads=ggml_ot.settings.n_threads,
    )

    current_scores = {"knn": acc}
    current_times = {"epoch_time(s)": elapsed}

    with capsys.disabled():
        evaluate_perf_case(
            request=request,
            suite="minibatch_perf",
            model="minibatch",
            context={"data": data_source, "solver": "emd2", "batch_size": str(kw["batch_size"])},
            log_metrics={"epoch_time(s)": elapsed, "knn": acc},
            snapshot_params=snapshot_params,
            current_scores=current_scores,
            current_times=current_times,
            thresholds=THRESHOLDS,
        )
