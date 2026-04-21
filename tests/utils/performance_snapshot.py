"""Performance snapshot utilities for tracking regression in test scores."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import get_dataset_params, get_kwargs, get_variants

# Snapshots stored under tests/data/performance_snapshots/
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "performance_snapshots"
CLASSIFICATION_FILE = SNAPSHOT_DIR / "classification_performance.json"
TIME_FILE = SNAPSHOT_DIR / "computation_time.json"
OVERVIEW_FILE = SNAPSHOT_DIR / "performance_overview.md"
OVERVIEW_HTML_FILE = SNAPSHOT_DIR / "performance_overview.html"
SUITE_TO_MODEL = {
    "core_perf": "empirical",
    "gmm_perf": "gmm",
    "minibatch_perf": "minibatch",
}
_DEVICE_OPTIONS = ("cpu", "gpu")


def _utc_timestamp() -> str:
    """Return ISO8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_hash(values: dict[str, Any]) -> str:
    text = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _coerce_param_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    txt = value.strip()
    if txt in {"None", "null"}:
        return None
    if txt in {"True", "False"}:
        return txt == "True"
    try:
        return ast.literal_eval(txt)
    except (ValueError, SyntaxError):
        return txt


def _value_matches(params: dict[str, Any], key: str, expected: Any) -> bool:
    if key not in params:
        return False
    actual = _coerce_param_value(params[key])
    if isinstance(expected, float):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _expected_dataset_ref_by_source() -> dict[str, str]:
    dataset_params = get_dataset_params()
    datasets = dataset_params["datasets"]

    # synthetic ref mirrors tests.utils.setup_dataset.get_perf_setup_identity
    synth_perf = datasets["synthetic"]["perf"]
    synth_config = dict(synth_perf["generator_kwargs"])
    synth_config["distribution_size"] = int(synth_perf["n_cells"])
    synthetic_ref = _stable_hash({**synth_config, "generator": "from_synth_default"})

    network_ref = str(datasets["network"]["dataset_id"])
    return {"synthetic": synthetic_ref, "network": network_ref}


def _matches_current_perf_config(params: dict[str, Any]) -> bool:
    suite = str(params.get("suite", ""))
    if suite not in SUITE_TO_MODEL:
        return False

    dataset_params = get_dataset_params()
    data_sources = set(dataset_params["data_sources"])
    data_source = str(params.get("data_source", ""))
    if data_source not in data_sources:
        return False

    expected_refs = _expected_dataset_ref_by_source()
    if not _value_matches(params, "dataset_ref", expected_refs[data_source]):
        return False

    solver = str(params.get("solver", ""))
    device = str(params.get("device", ""))
    if device not in _DEVICE_OPTIONS:
        return False

    suite_name = SUITE_TO_MODEL[suite]  # "empirical", "gmm", "minibatch"

    # Check train_test kwargs (merged defaults → perf → suite).
    # entropic_reg in YAML is the sinkhorn default; emd2 snapshots store 0.0.
    merged_kw = get_kwargs("perf", "train_test", suite=suite_name)
    expected_entropic = 0.0 if solver == "emd2" else float(merged_kw.get("entropic_reg", 10.0))
    if not _value_matches(params, "entropic_reg", expected_entropic):
        return False
    for key, expected in merged_kw.items():
        if key == "entropic_reg":
            continue  # already checked solver-specific above
        if not _value_matches(params, key, expected):
            return False

    if suite == "core_perf":
        return True

    if suite == "minibatch_perf":
        return True

    # gmm_perf — additional dataset-fitting checks
    cfg_dataset = dataset_params["gmm_fitting"]
    variants = get_variants("gmm")
    variant = str(params.get("variant", "")).strip() or "none"
    if variant not in variants:
        return False
    if not _value_matches(params, "sharing", str(cfg_dataset["component_sharing"])):
        return False
    if not _value_matches(params, "k_comps", list(cfg_dataset["k_comps_candidates_by_data_source"][data_source])):
        return False

    fit_map = {
        "fit_k_selection_metric": cfg_dataset["fit_gmm_kwargs"]["k_selection_metric"],
        "fit_refit": cfg_dataset["fit_gmm_kwargs"].get("refit", "full"),
        "fit_covariance_type": cfg_dataset["fit_gmm_kwargs"]["covariance_type"],
        "fit_max_iter": int(cfg_dataset["fit_gmm_kwargs"]["max_iter"]),
        "fit_use_rep": cfg_dataset["use_rep_by_data_source"][data_source],
    }
    for key, expected in fit_map.items():
        if not _value_matches(params, key, expected):
            return False

    # Historical snapshots stored the GMM holdout ratio under
    # `fit_subsample_frac`; newer tests may emit `fit_train_size`.
    expected_fit_train_size = float(cfg_dataset["fit_gmm_kwargs"]["train_size"])
    if not (
        _value_matches(params, "fit_train_size", expected_fit_train_size)
        or _value_matches(params, "fit_subsample_frac", expected_fit_train_size)
    ):
        return False

    return True


def get_snapshot_key(**kwargs) -> tuple[str, dict[str, Any]]:
    """Generate a unique key from test configuration."""
    ordered_params = dict(sorted(kwargs.items(), key=lambda kv: kv[0]))
    key = "|".join(f"{k}={v}" for k, v in ordered_params.items())
    return key, ordered_params


def load_baseline(file_path: Path, key: str) -> dict[str, float]:
    """Load baseline metrics for a given key from snapshot file."""
    if file_path.exists():
        with open(file_path) as f:
            data = json.load(f)
        entry = data.get(key, {})
        if isinstance(entry, dict) and "metrics" in entry:
            metrics = entry["metrics"]
            if isinstance(metrics, dict):
                return {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        return {}
    return {}


def save_baseline(
    file_path: Path,
    key: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    updated_at: str | None = None,
) -> None:
    """Save baseline metrics for a key, merging with any existing metrics."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if file_path.exists():
        with open(file_path) as f:
            data = json.load(f)

    existing = data.get(key, {})
    existing_metrics: dict[str, float] = {}
    if isinstance(existing, dict) and isinstance(existing.get("metrics"), dict):
        existing_metrics = {str(k): float(v) for k, v in existing["metrics"].items() if isinstance(v, (int, float))}

    merged_metrics = {**existing_metrics, **{k: float(v) for k, v in metrics.items()}}
    data[key] = {"params": params, "metrics": merged_metrics, "updated_at": updated_at or _utc_timestamp()}
    with open(file_path, "w") as f:
        json.dump({k: data[k] for k in sorted(data)}, f, indent=2, sort_keys=True)


def log_perf_case(
    *,
    suite: str,
    context: dict[str, str],
    metrics: dict[str, float],
    optional_metrics: dict[str, float] | None = None,
    metric_annotations: dict[str, str] | None = None,
) -> None:
    """Print standardized performance metrics for one benchmark case."""
    ctx = ", ".join(f"{k}: {v}" for k, v in context.items())
    print(f"\n[{suite} perf] {ctx}")
    annotations = metric_annotations or {}
    for name, value in metrics.items():
        suffix = f" {annotations[name]}" if name in annotations else ""
        print(f"  {name}: {value:.3f}{suffix}")
    if optional_metrics:
        for name, value in optional_metrics.items():
            suffix = f" {annotations[name]}" if name in annotations else ""
            print(f"  {name}: {value:.3f}{suffix}")


def compare_and_update_snapshots(
    *,
    key: str,
    params: dict[str, Any],
    current_scores: dict[str, float],
    current_times: dict[str, float],
    update_baseline: bool,
    accuracy_rel_drop_max: float,
    time_rel_increase_max: float,
    time_abs_increase_max_s: float,
) -> tuple[dict[str, str], list[str]]:
    """Run snapshot comparison and optional baseline update.

    Returns metric annotations for inline perf output. Only degraded metrics are annotated.
    """
    baseline_scores = load_baseline(CLASSIFICATION_FILE, key)
    baseline_times = load_baseline(TIME_FILE, key)
    metric_annotations: dict[str, str] = {}
    issues: list[str] = []

    if baseline_scores:
        for metric, cur_val in current_scores.items():
            if metric not in baseline_scores:
                continue
            base_val = float(baseline_scores[metric])
            if base_val <= 0:
                continue
            min_allowed = base_val * (1.0 - accuracy_rel_drop_max)
            cur = float(cur_val)
            if cur < min_allowed:
                drop_rel = (base_val - cur) / base_val
                annotation = f"(DEGRADATION: baseline={base_val:.4f}, drop={drop_rel * 100:.1f}%)"
                metric_annotations[metric] = annotation
                issues.append(
                    f"{metric}: {cur:.4f} < {min_allowed:.4f} (baseline={base_val:.4f}, drop={drop_rel * 100:.1f}%)"
                )

    if baseline_times:
        for metric, cur_val in current_times.items():
            if metric not in baseline_times:
                continue
            base_val = float(baseline_times[metric])
            if base_val < 0:
                continue
            max_allowed = base_val * (1.0 + time_rel_increase_max) + time_abs_increase_max_s
            cur = float(cur_val)
            if cur > max_allowed:
                inc_rel = ((cur - base_val) / base_val) if base_val > 0 else float("inf")
                annotation = f"(TIME DEGRADATION: baseline={base_val:.4f}s, increase={inc_rel * 100:.1f}%)"
                metric_annotations[metric] = annotation
                issues.append(
                    f"{metric}: {cur:.4f}s > {max_allowed:.4f}s "
                    f"(baseline={base_val:.4f}s, increase={inc_rel * 100:.1f}%)"
                )

    if update_baseline and not issues:
        save_baseline(CLASSIFICATION_FILE, key, params, current_scores)
        save_baseline(TIME_FILE, key, params, current_times)

    return metric_annotations, issues


def get_perf_cli_flags(request: Any) -> tuple[bool, bool]:
    """Return ``(--update-baseline, --fail-on-degradation)`` flags for current test run."""
    config = request.config
    update_baseline = bool(config.getoption("--update-baseline", default=False))
    fail_on_degradation = bool(config.getoption("--fail-on-degradation", default=False))
    return update_baseline, fail_on_degradation


def evaluate_perf_case(
    *,
    request: Any,
    suite: str,
    model: str,
    context: dict[str, str],
    log_metrics: dict[str, float],
    snapshot_params: dict[str, Any],
    current_scores: dict[str, float],
    current_times: dict[str, float],
    thresholds: dict[str, float],
    optional_log_metrics: dict[str, float] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Log one perf case, compare against snapshot baseline, and optionally update baseline."""
    key, params = get_snapshot_key(suite=suite, **snapshot_params)
    update_baseline, fail_on_degradation = get_perf_cli_flags(request)
    metric_annotations, issues = compare_and_update_snapshots(
        key=key,
        params=params,
        current_scores=current_scores,
        current_times=current_times,
        update_baseline=update_baseline,
        accuracy_rel_drop_max=float(thresholds["accuracy_rel_drop_max"]),
        time_rel_increase_max=float(thresholds["time_rel_increase_max"]),
        time_abs_increase_max_s=float(thresholds["time_abs_increase_max_s"]),
    )
    log_perf_case(
        suite=model,
        context=context,
        metrics=log_metrics,
        optional_metrics=optional_log_metrics,
        metric_annotations=metric_annotations,
    )
    if fail_on_degradation and issues:
        msg = f"SNAPSHOT DEGRADATION for setup_key={key}:\n  " + "\n  ".join(issues)
        raise AssertionError(msg)
    return key, params


def _parse_key_params(key: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for chunk in key.split("|"):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        params[k] = v
    return params


def _load_snapshot_entries(file_path: Path) -> dict[str, dict[str, Any]]:
    if not file_path.exists():
        return {}
    with open(file_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}

    entries: dict[str, dict[str, Any]] = {}
    for key, raw in data.items():
        if not isinstance(raw, dict):
            continue
        params = raw.get("params", {})
        metrics = raw.get("metrics", {})
        updated_at = raw.get("updated_at", "")
        if not isinstance(params, dict) or not isinstance(metrics, dict):
            continue
        merged_params = {**_parse_key_params(key), **{str(k): v for k, v in params.items()}}
        entries[key] = {
            "params": merged_params,
            "metrics": {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            "updated_at": str(updated_at) if updated_at is not None else "",
        }
    return entries


def _suite_to_model(suite_name: str) -> str:
    return SUITE_TO_MODEL.get(suite_name, suite_name or "unknown")


def _mean_dist_label_from_params(params: dict[str, Any]) -> str:
    """Return display label for squared/non-squared GMM mean-distance settings."""
    if str(params.get("suite", "")) != "gmm_perf":
        return ""
    squared_ground_cost = _coerce_param_value(params.get("squared_ground_cost", False))
    return "squared" if bool(squared_ground_cost) else "unsquared"


def _is_newer_timestamp(candidate: str, current: str) -> bool:
    if not current:
        return bool(candidate)
    if not candidate:
        return False
    return candidate > current


def build_snapshot_matrix_rows(*, suite: str | None = None) -> list[dict[str, Any]]:
    """Build merged rows from classification/time snapshots for display."""
    cls_entries = _load_snapshot_entries(CLASSIFICATION_FILE)
    time_entries = _load_snapshot_entries(TIME_FILE)
    by_display_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}

    for key in sorted(set(cls_entries) | set(time_entries)):
        cls_entry = cls_entries.get(key, {})
        time_entry = time_entries.get(key, {})

        params = {}
        params.update(time_entry.get("params", {}))
        params.update(cls_entry.get("params", {}))
        suite_name = str(params.get("suite", ""))
        if suite is None and suite_name not in SUITE_TO_MODEL:
            continue
        if suite is not None and suite_name != suite:
            continue
        if not _matches_current_perf_config(params):
            continue
        model_name = _suite_to_model(suite_name)

        metrics = {}
        metrics.update(time_entry.get("metrics", {}))
        metrics.update(cls_entry.get("metrics", {}))
        updated_at = cls_entry.get("updated_at") or time_entry.get("updated_at") or ""

        row = {
            "model": model_name,
            "data_source": str(params.get("data_source", "")),
            "variant": str(params.get("variant", "")).strip() or "none",
            "mean_dist": _mean_dist_label_from_params(params),
            "solver": str(params.get("solver", "")),
            "device": str(params.get("device", "")),
            "knn": metrics.get("knn"),
            "epoch_time(s)": metrics.get("epoch_time(s)"),
            "gmm_fit_time(s)": metrics.get("gmm_fit_time(s)"),
            "inf_time(s)": metrics.get("inf_time(s)"),
            "updated_at": str(updated_at),
        }
        display_key = _display_key_from_params(params)
        existing = by_display_key.get(display_key)
        if existing is None or _is_newer_timestamp(row["updated_at"], str(existing.get("updated_at", ""))):
            by_display_key[display_key] = row

    rows = list(by_display_key.values())
    rows.sort(
        key=lambda row: (
            row["data_source"],
            row["model"],
            row["variant"],
            row["mean_dist"],
            row["solver"],
            row["device"],
        )
    )
    return rows


def format_snapshot_matrix_markdown(rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> str:
    """Format rows as a compact Markdown table."""
    if columns is None:
        columns = [
            "data_source",
            "model",
            "variant",
            "mean_dist",
            "solver",
            "device",
            "knn",
            "epoch_time(s)",
            "gmm_fit_time(s)",
            "inf_time(s)",
            "updated_at",
        ]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            elif val is None:
                values.append("")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_snapshot_matrix_html(rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> str:
    """Format rows as a compact HTML table."""
    if columns is None:
        columns = [
            "data_source",
            "model",
            "variant",
            "mean_dist",
            "solver",
            "device",
            "knn",
            "epoch_time(s)",
            "gmm_fit_time(s)",
            "inf_time(s)",
            "updated_at",
        ]

    parts = [
        '<table class="perf-table">',
        "<thead>",
        "<tr>",
        *(f"<th>{escape(col)}</th>" for col in columns),
        "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for row in rows:
        parts.append("<tr>")
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                cell = f"{val:.4f}"
            elif val is None:
                cell = ""
            else:
                cell = str(val)
            parts.append(f"<td>{escape(cell)}</td>")
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "\n".join(parts)


def build_snapshot_overview_markdown(
    *,
    data_sources: list[str] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> str:
    """Build a human-readable Markdown overview split by data source."""
    selected_sources = data_sources or ["synthetic", "network"]
    all_rows = build_snapshot_matrix_rows() if rows is None else rows
    table_columns = [
        "model",
        "variant",
        "mean_dist",
        "solver",
        "device",
        "knn",
        "epoch_time(s)",
        "gmm_fit_time(s)",
        "inf_time(s)",
        "updated_at",
    ]
    lines: list[str] = [
        "# Performance Snapshot Overview",
        "",
        f"Generated: {_utc_timestamp()}",
        "",
    ]

    found_any = False
    for source_name in selected_sources:
        rows = [row for row in all_rows if row.get("data_source") == source_name]
        lines.append(f"## `{source_name}`")
        if rows:
            lines.append(format_snapshot_matrix_markdown(rows, columns=table_columns))
            found_any = True
        else:
            lines.append("_No snapshot entries found._")
        lines.append("")

    if not found_any:
        lines.append("_No performance snapshot entries available yet._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_snapshot_overview_html(
    *,
    data_sources: list[str] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> str:
    """Build a human-readable HTML overview split by data source."""
    selected_sources = data_sources or ["synthetic", "network"]
    all_rows = build_snapshot_matrix_rows() if rows is None else rows
    table_columns = [
        "model",
        "variant",
        "mean_dist",
        "solver",
        "device",
        "knn",
        "epoch_time(s)",
        "gmm_fit_time(s)",
        "inf_time(s)",
        "updated_at",
    ]

    sections: list[str] = []
    found_any = False
    for source_name in selected_sources:
        source_rows = [row for row in all_rows if row.get("data_source") == source_name]
        sections.append(f"<section><h2>{escape(source_name)}</h2>")
        if source_rows:
            sections.append(format_snapshot_matrix_html(source_rows, columns=table_columns))
            found_any = True
        else:
            sections.append('<p class="empty">No snapshot entries found.</p>')
        sections.append("</section>")

    if not found_any:
        sections.append('<p class="empty">No performance snapshot entries available yet.</p>')

    timestamp = escape(_utc_timestamp())
    body = "\n".join(sections)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "  <title>Performance Snapshot Overview</title>\n"
        "  <style>\n"
        "    :root { color-scheme: light; }\n"
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; color: #18212b; }\n"
        "    h1, h2 { margin-bottom: 0.5rem; }\n"
        "    .meta { color: #51606f; margin-bottom: 1.5rem; }\n"
        "    section { margin-top: 2rem; }\n"
        "    .perf-table { border-collapse: collapse; width: 100%; font-size: 0.95rem; }\n"
        "    .perf-table th, .perf-table td { border: 1px solid #d7dee6; padding: 0.45rem 0.6rem; text-align: left; }\n"
        "    .perf-table th { background: #eef3f8; }\n"
        "    .perf-table tbody tr:nth-child(even) { background: #f9fbfd; }\n"
        "    .empty { color: #6b7785; font-style: italic; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>Performance Snapshot Overview</h1>\n"
        f'  <p class="meta">Generated: {timestamp}</p>\n'
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def write_snapshot_overview_file(output_path: Path | None = None, *, skip_if_empty: bool = False) -> Path | None:
    """Write Markdown overview file for perf snapshot suites and return its path.

    When ``skip_if_empty=True``, no file is written and ``None`` is returned if
    there are no current-config snapshot rows to summarize.
    """
    rows = build_snapshot_matrix_rows()
    if skip_if_empty and not rows:
        return None

    path = output_path or OVERVIEW_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    content = build_snapshot_overview_markdown(rows=rows)
    path.write_text(content, encoding="utf-8")
    return path


def write_snapshot_overview_html_file(output_path: Path | None = None, *, skip_if_empty: bool = False) -> Path | None:
    """Write HTML overview file for perf snapshot suites and return its path."""
    rows = build_snapshot_matrix_rows()
    if skip_if_empty and not rows:
        return None

    path = output_path or OVERVIEW_HTML_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    content = build_snapshot_overview_html(rows=rows)
    path.write_text(content, encoding="utf-8")
    return path


def _display_key_from_params(params: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    suite_name = str(params.get("suite", ""))
    model_name = _suite_to_model(suite_name)
    return (
        str(params.get("data_source", "")),
        model_name,
        str(params.get("variant", "")).strip() or "none",
        _mean_dist_label_from_params(params),
        str(params.get("solver", "")),
        str(params.get("device", "")),
    )


def _prune_snapshot_file(file_path: Path) -> tuple[int, int]:
    if not file_path.exists():
        return 0, 0

    with open(file_path) as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, dict):
        raw_data = {}

    selected: dict[tuple[str, str, str, str, str, str], tuple[str, dict[str, Any]]] = {}
    for key, entry in raw_data.items():
        if not isinstance(entry, dict):
            continue
        params_data = entry.get("params", {})
        metrics_data = entry.get("metrics", {})
        updated_at = str(entry.get("updated_at", "") or "")
        if not isinstance(params_data, dict) or not isinstance(metrics_data, dict):
            continue

        merged_params = {**_parse_key_params(key), **{str(k): v for k, v in params_data.items()}}
        if not _matches_current_perf_config(merged_params):
            continue

        display_key = _display_key_from_params(merged_params)
        current = selected.get(display_key)
        if current is None or _is_newer_timestamp(updated_at, current[1].get("updated_at", "")):
            selected[display_key] = (
                key,
                {
                    "params": {str(k): v for k, v in params_data.items()},
                    "metrics": {str(k): float(v) for k, v in metrics_data.items() if isinstance(v, (int, float))},
                    "updated_at": updated_at,
                },
            )

    pruned_data = {key: value for key, value in sorted((item[0], item[1]) for item in selected.values())}
    with open(file_path, "w") as f:
        json.dump(pruned_data, f, indent=2, sort_keys=True)
    return len(raw_data), len(pruned_data)


def prune_snapshot_files_to_current_config() -> dict[str, tuple[int, int]]:
    """Keep only current-config snapshot entries (one per display row) in baseline files."""
    return {
        str(CLASSIFICATION_FILE): _prune_snapshot_file(CLASSIFICATION_FILE),
        str(TIME_FILE): _prune_snapshot_file(TIME_FILE),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Print a Markdown overview table of perf snapshot baselines.")
    parser.add_argument("--suite", default=None, help="Optional raw suite filter (e.g. core_perf, gmm_perf).")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows (0 = all rows).")
    parser.add_argument("--write-overview", action="store_true", help="Write overview Markdown file to snapshot dir.")
    parser.add_argument("--write-overview-html", action="store_true", help="Write overview HTML file to snapshot dir.")
    parser.add_argument(
        "--prune-to-config",
        action="store_true",
        help="Prune snapshot JSON files to active YAML-configured setups before printing/writing.",
    )
    args = parser.parse_args()

    if args.prune_to_config:
        results = prune_snapshot_files_to_current_config()
        for path, (before, after) in results.items():
            print(f"Pruned {path}: {before} -> {after} entries")

    rows = build_snapshot_matrix_rows(suite=args.suite)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(format_snapshot_matrix_markdown(rows))
    if args.write_overview:
        output_path = write_snapshot_overview_file()
        print(f"\nWrote overview file: {output_path}")
    if args.write_overview_html:
        output_path = write_snapshot_overview_html_file()
        print(f"\nWrote HTML overview file: {output_path}")


if __name__ == "__main__":
    _main()
