"""Unit tests for performance snapshot helper behavior."""

from __future__ import annotations

import json

from .utils import performance_snapshot as ps
from .utils.config import get_dataset_params, get_kwargs, build_snapshot_params


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def test_compare_and_update_does_not_overwrite_on_degradation(tmp_path, monkeypatch):
    """Baseline files should not be rewritten when current metrics degrade."""
    cls_file = tmp_path / "classification.json"
    time_file = tmp_path / "time.json"
    monkeypatch.setattr(ps, "CLASSIFICATION_FILE", cls_file)
    monkeypatch.setattr(ps, "TIME_FILE", time_file)

    key = "suite=core_perf|data_source=synthetic|solver=emd2|device=cpu"
    params = {"suite": "core_perf", "data_source": "synthetic", "solver": "emd2", "device": "cpu"}
    baseline_cls = {key: {"params": params, "metrics": {"knn": 0.95}, "updated_at": "2026-02-20T00:00:00+00:00"}}
    baseline_time = {
        key: {"params": params, "metrics": {"epoch_time(s)": 1.0}, "updated_at": "2026-02-20T00:00:00+00:00"}
    }
    _write_json(cls_file, baseline_cls)
    _write_json(time_file, baseline_time)

    _, issues = ps.compare_and_update_snapshots(
        key=key,
        params=params,
        current_scores={"knn": 0.7},
        current_times={"epoch_time(s)": 2.0},
        update_baseline=True,
        accuracy_rel_drop_max=0.01,
        time_rel_increase_max=0.01,
        time_abs_increase_max_s=0.0,
    )

    assert issues
    assert json.loads(cls_file.read_text(encoding="utf-8")) == baseline_cls
    assert json.loads(time_file.read_text(encoding="utf-8")) == baseline_time


def test_build_snapshot_rows_filters_to_current_config_and_deduplicates(tmp_path, monkeypatch):
    """Overview rows should include only current-config setups and keep newest duplicate."""
    cls_file = tmp_path / "classification.json"
    time_file = tmp_path / "time.json"
    monkeypatch.setattr(ps, "CLASSIFICATION_FILE", cls_file)
    monkeypatch.setattr(ps, "TIME_FILE", time_file)

    dataset_params = get_dataset_params()
    expected_ref = ps._expected_dataset_ref_by_source()["synthetic"]  # noqa: SLF001 - internal helper under test
    train_kwargs = get_kwargs("perf", "train_test", suite="empirical")

    params_base = build_snapshot_params(
        "empirical",
        solver="emd2",
        device="cpu",
        data_source="synthetic",
        dataset_ref=expected_ref,
        n_threads=4,  # placeholder, overridden below
    )
    # evaluate_perf_case injects 'suite' at get_snapshot_key level; replicate here.
    params_base["suite"] = "core_perf"
    # Remove n_threads so we can parametrize it below.
    del params_base["n_threads"]
    key_old, _ = ps.get_snapshot_key(**{**params_base, "n_threads": 4})
    key_new, _ = ps.get_snapshot_key(**{**params_base, "n_threads": 16})

    bad_params = dict(params_base)
    bad_params["lr"] = float(train_kwargs["lr"]) * 10.0
    key_bad, _ = ps.get_snapshot_key(**bad_params)

    _write_json(
        cls_file,
        {
            key_old: {
                "params": {**params_base, "n_threads": 4},
                "metrics": {"knn": 0.8},
                "updated_at": "2026-02-20T00:00:00+00:00",
            },
            key_new: {
                "params": {**params_base, "n_threads": 16},
                "metrics": {"knn": 0.9},
                "updated_at": "2026-02-21T00:00:00+00:00",
            },
            key_bad: {"params": bad_params, "metrics": {"knn": 0.1}, "updated_at": "2026-02-22T00:00:00+00:00"},
        },
    )
    _write_json(
        time_file,
        {
            key_old: {
                "params": {**params_base, "n_threads": 4},
                "metrics": {"epoch_time(s)": 2.0},
                "updated_at": "2026-02-20T00:00:00+00:00",
            },
            key_new: {
                "params": {**params_base, "n_threads": 16},
                "metrics": {"epoch_time(s)": 1.5},
                "updated_at": "2026-02-21T00:00:00+00:00",
            },
            key_bad: {
                "params": bad_params,
                "metrics": {"epoch_time(s)": 99.0},
                "updated_at": "2026-02-22T00:00:00+00:00",
            },
        },
    )

    rows = ps.build_snapshot_matrix_rows()
    rows = [
        r
        for r in rows
        if r["data_source"] == "synthetic"
        and r["model"] == "empirical"
        and r["solver"] == "emd2"
        and r["device"] == "cpu"
    ]

    assert len(rows) == 1
    assert rows[0]["mean_dist"] == ""
    assert rows[0]["knn"] == 0.9
    assert rows[0]["epoch_time(s)"] == 1.5
    assert rows[0]["updated_at"] == "2026-02-21T00:00:00+00:00"
    assert set(dataset_params["data_sources"]) >= {"synthetic", "network"}


def test_gmm_snapshot_rows_keep_squared_ground_cost_variants_separate(tmp_path, monkeypatch):
    """GMM overview rows should keep squared/unsquared mean-distance baselines distinct."""
    cls_file = tmp_path / "classification.json"
    time_file = tmp_path / "time.json"
    monkeypatch.setattr(ps, "CLASSIFICATION_FILE", cls_file)
    monkeypatch.setattr(ps, "TIME_FILE", time_file)

    gmm_cfg = get_dataset_params()["gmm_fitting"]
    expected_ref = ps._expected_dataset_ref_by_source()["synthetic"]  # noqa: SLF001 - internal helper under test

    params_base = build_snapshot_params(
        "gmm",
        solver="emd2",
        device="cpu",
        data_source="synthetic",
        dataset_ref=expected_ref,
        n_threads=4,
        sharing=str(gmm_cfg["component_sharing"]),
        variant="none",
        n_cells=int(gmm_cfg["max_cells_per_patient"]),
        k_comps=list(gmm_cfg["k_comps_candidates_by_data_source"]["synthetic"]),
        fit_k_selection_metric=gmm_cfg["fit_gmm_kwargs"]["k_selection_metric"],
        fit_refit=gmm_cfg["fit_gmm_kwargs"].get("refit", "full"),
        fit_covariance_type=gmm_cfg["fit_gmm_kwargs"]["covariance_type"],
        fit_subsample_frac=float(gmm_cfg["fit_gmm_kwargs"]["train_size"]),
        fit_max_iter=int(gmm_cfg["fit_gmm_kwargs"]["max_iter"]),
        fit_use_rep=gmm_cfg["use_rep_by_data_source"]["synthetic"],
    )
    params_base["suite"] = "gmm_perf"

    params_squared = {**params_base, "squared_ground_cost": True}
    params_unsquared = {**params_base, "squared_ground_cost": False}
    key_squared, _ = ps.get_snapshot_key(**params_squared)
    key_unsquared, _ = ps.get_snapshot_key(**params_unsquared)

    _write_json(
        cls_file,
        {
            key_squared: {
                "params": params_squared,
                "metrics": {"knn": 0.81},
                "updated_at": "2026-02-20T00:00:00+00:00",
            },
            key_unsquared: {
                "params": params_unsquared,
                "metrics": {"knn": 0.78},
                "updated_at": "2026-02-21T00:00:00+00:00",
            },
        },
    )
    _write_json(
        time_file,
        {
            key_squared: {
                "params": params_squared,
                "metrics": {"epoch_time(s)": 2.0},
                "updated_at": "2026-02-20T00:00:00+00:00",
            },
            key_unsquared: {
                "params": params_unsquared,
                "metrics": {"epoch_time(s)": 2.2},
                "updated_at": "2026-02-21T00:00:00+00:00",
            },
        },
    )

    rows = ps.build_snapshot_matrix_rows(suite="gmm_perf")
    rows = [
        r
        for r in rows
        if r["data_source"] == "synthetic"
        and r["model"] == "gmm"
        and r["variant"] == "none"
        and r["solver"] == "emd2"
        and r["device"] == "cpu"
    ]

    assert len(rows) == 2
    assert {row["mean_dist"] for row in rows} == {"squared", "unsquared"}

    markdown = ps.build_snapshot_overview_markdown(data_sources=["synthetic"], rows=rows)
    assert "| model | variant | mean_dist | solver | device |" in markdown
    assert "squared" in markdown
    assert "unsquared" in markdown

    html = ps.build_snapshot_overview_html(data_sources=["synthetic"], rows=rows)
    assert "<table" in html
    assert "<th>mean_dist</th>" in html
    assert "squared" in html
    assert "unsquared" in html


def test_prune_snapshot_files_keeps_gmm_mean_dist_variants_separate(tmp_path, monkeypatch):
    """Pruning should not collapse squared/unsquared GMM rows into one baseline entry."""
    cls_file = tmp_path / "classification.json"
    time_file = tmp_path / "time.json"
    monkeypatch.setattr(ps, "CLASSIFICATION_FILE", cls_file)
    monkeypatch.setattr(ps, "TIME_FILE", time_file)

    gmm_cfg = get_dataset_params()["gmm_fitting"]
    expected_ref = ps._expected_dataset_ref_by_source()["synthetic"]  # noqa: SLF001 - internal helper under test

    params_base = build_snapshot_params(
        "gmm",
        solver="emd2",
        device="cpu",
        data_source="synthetic",
        dataset_ref=expected_ref,
        n_threads=4,
        sharing=str(gmm_cfg["component_sharing"]),
        variant="none",
        n_cells=int(gmm_cfg["max_cells_per_patient"]),
        k_comps=list(gmm_cfg["k_comps_candidates_by_data_source"]["synthetic"]),
        fit_k_selection_metric=gmm_cfg["fit_gmm_kwargs"]["k_selection_metric"],
        fit_refit=gmm_cfg["fit_gmm_kwargs"].get("refit", "full"),
        fit_covariance_type=gmm_cfg["fit_gmm_kwargs"]["covariance_type"],
        fit_subsample_frac=float(gmm_cfg["fit_gmm_kwargs"]["train_size"]),
        fit_max_iter=int(gmm_cfg["fit_gmm_kwargs"]["max_iter"]),
        fit_use_rep=gmm_cfg["use_rep_by_data_source"]["synthetic"],
    )
    params_base["suite"] = "gmm_perf"

    params_squared_old = {**params_base, "squared_ground_cost": True}
    params_squared_new = {**params_base, "squared_ground_cost": True, "n_threads": 16}
    params_unsquared = {**params_base, "squared_ground_cost": False}
    key_squared_old, _ = ps.get_snapshot_key(**params_squared_old)
    key_squared_new, _ = ps.get_snapshot_key(**params_squared_new)
    key_unsquared, _ = ps.get_snapshot_key(**params_unsquared)

    _write_json(
        cls_file,
        {
            key_squared_old: {
                "params": params_squared_old,
                "metrics": {"knn": 0.70},
                "updated_at": "2026-02-20T00:00:00+00:00",
            },
            key_squared_new: {
                "params": params_squared_new,
                "metrics": {"knn": 0.80},
                "updated_at": "2026-02-21T00:00:00+00:00",
            },
            key_unsquared: {
                "params": params_unsquared,
                "metrics": {"knn": 0.75},
                "updated_at": "2026-02-22T00:00:00+00:00",
            },
        },
    )
    _write_json(
        time_file,
        {
            key_squared_old: {
                "params": params_squared_old,
                "metrics": {"epoch_time(s)": 2.4},
                "updated_at": "2026-02-20T00:00:00+00:00",
            },
            key_squared_new: {
                "params": params_squared_new,
                "metrics": {"epoch_time(s)": 2.0},
                "updated_at": "2026-02-21T00:00:00+00:00",
            },
            key_unsquared: {
                "params": params_unsquared,
                "metrics": {"epoch_time(s)": 2.2},
                "updated_at": "2026-02-22T00:00:00+00:00",
            },
        },
    )

    results = ps.prune_snapshot_files_to_current_config()

    cls_pruned = json.loads(cls_file.read_text(encoding="utf-8"))
    time_pruned = json.loads(time_file.read_text(encoding="utf-8"))

    assert results[str(cls_file)] == (3, 2)
    assert results[str(time_file)] == (3, 2)
    assert set(cls_pruned) == {key_squared_new, key_unsquared}
    assert set(time_pruned) == {key_squared_new, key_unsquared}


def test_write_snapshot_overview_file_skip_if_empty_avoids_empty_output(tmp_path, monkeypatch):
    """Session-time overview writes should be skipped when no current snapshot entries exist."""
    cls_file = tmp_path / "classification.json"
    time_file = tmp_path / "time.json"
    overview_file = tmp_path / "performance_overview.md"
    overview_html_file = tmp_path / "performance_overview.html"
    monkeypatch.setattr(ps, "CLASSIFICATION_FILE", cls_file)
    monkeypatch.setattr(ps, "TIME_FILE", time_file)
    monkeypatch.setattr(ps, "OVERVIEW_FILE", overview_file)
    monkeypatch.setattr(ps, "OVERVIEW_HTML_FILE", overview_html_file)

    output = ps.write_snapshot_overview_file(skip_if_empty=True)
    html_output = ps.write_snapshot_overview_html_file(skip_if_empty=True)

    assert output is None
    assert html_output is None
    assert not overview_file.exists()
    assert not overview_html_file.exists()
