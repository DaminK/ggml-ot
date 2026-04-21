"""Validation helpers for fitted GMM representations."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from ggml_ot._utils._anndata_utils import (
    get_distribution_ids,
    get_distribution_index,
)

_SUMMARY_COLUMN_ORDER = [
    "count",
    "mean_train_nll",
    "sd_train_nll",
    "mean_validation_nll",
    "sd_validation_nll",
    "mean_nll_gap",
    "sd_nll_gap",
]


@dataclass(frozen=True)
class GMMValidationReport:
    """Bundle detailed hold-out diagnostics, a compact summary, and an optional plot."""

    results: pd.DataFrame
    summary: pd.DataFrame
    ax: object | None = None


def _resolve_distribution_ids(adata, *, gmm_cfg: dict, patient_col: str) -> np.ndarray:
    """Resolve patient/distribution ids in persisted GMM order when available."""
    stored_distribution_ids = gmm_cfg.get("weight_inference", {}).get("distribution_ids")
    if stored_distribution_ids is None:
        return np.asarray(get_distribution_ids(adata, patient_col), dtype=object)
    return np.asarray([str(dist_id) for dist_id in stored_distribution_ids], dtype=object)


def _resolve_distribution_n_components(gmm_cfg: dict, *, n_distributions: int) -> np.ndarray:
    """Return one component-count entry per persisted distribution."""
    distribution_n_components = np.asarray(
        gmm_cfg.get("distribution_n_components", np.full(n_distributions, int(gmm_cfg["n_components"]))),
        dtype=int,
    )
    if distribution_n_components.shape != (n_distributions,):
        raise ValueError(
            "distribution_n_components must contain one entry per distribution. "
            f"Got shape={distribution_n_components.shape}, expected ({n_distributions},)."
        )
    return distribution_n_components


def _columnar_to_records(columnar: Mapping[str, Sequence[object]]) -> list[dict[str, object]]:
    """Convert a columnar selection trace back into per-k records."""
    if not columnar:
        return []

    keys = list(columnar.keys())
    lengths = {len(columnar[key]) for key in keys}
    if len(lengths) != 1:
        raise ValueError("Stored selection diagnostics must use aligned column lengths.")

    n_rows = lengths.pop()
    records: list[dict[str, object]] = []
    for row_idx in range(n_rows):
        record = {key: columnar[key][row_idx] for key in keys}
        records.append(record)
    return records


def _record_for_best_k(
    columnar_scores: Mapping[str, Sequence[object]],
    *,
    best_k: int,
    context: str,
) -> dict[str, object]:
    """Return the stored heldout-NLL diagnostics for the selected K."""
    records = _columnar_to_records(columnar_scores)
    matches = [record for record in records if int(record["k"]) == int(best_k)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one stored score record for best_k={best_k} in {context}, found {len(matches)}."
        )

    record = dict(matches[0])
    required = ["train_nll", "validation_nll", "nll_gap", "n_train_cells", "n_val_cells"]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(
            "Stored selection metadata does not contain hold-out diagnostics "
            f"({missing}). Please rerun fit_gmm with the current ggml_ot version."
        )
    return record


def _label_map(adata, *, patient_col: str, label_col: str) -> dict[str, str]:
    """Return one label string per distribution, warning on mixed labels."""
    patient_values = adata.obs[patient_col].astype(str)
    label_values = adata.obs[label_col].astype(str)

    label_by_distribution: dict[str, str] = {}
    for distribution_id, frame in pd.DataFrame(
        {
            "distribution_id": patient_values.to_numpy(),
            "label": label_values.to_numpy(),
        }
    ).groupby("distribution_id", sort=False):
        labels = np.unique(frame["label"].to_numpy())
        if len(labels) == 0:
            raise ValueError(f"No labels found for distribution {distribution_id!r}.")
        if len(labels) > 1:
            warnings.warn(
                f"Cells from one sample {patient_col}={distribution_id!r} contain multiple labels {label_col}={labels}"
            )
        label_by_distribution[str(distribution_id)] = str(labels[0])
    return label_by_distribution


def _resolve_validation_target(data, *, gmm_key: str | None, patient_col: str | None, label_col: str | None):
    """Resolve the AnnData object and metadata needed for hold-out validation."""
    if hasattr(data, "adata"):
        use_rep = getattr(data, "use_rep", None)
        resolved_gmm_key = gmm_key or f"gmm_{'X' if use_rep is None else use_rep}"
        resolved_patient_col = patient_col or getattr(data, "patient_col", None)
        resolved_label_col = label_col if label_col is not None else getattr(data, "label_col", None)
        if resolved_patient_col is None:
            raise ValueError("Could not resolve `patient_col` from the dataset. Please pass it explicitly.")
        return data.adata, resolved_gmm_key, resolved_patient_col, resolved_label_col

    if hasattr(data, "obs") and hasattr(data, "uns"):
        if gmm_key is None:
            raise ValueError("`gmm_key` is required when validating a raw AnnData object.")
        if patient_col is None:
            raise ValueError("`patient_col` is required when validating a raw AnnData object.")
        return data, gmm_key, patient_col, label_col

    raise TypeError(
        "GMM fit validation requires an AnnData object or an AnnData-backed dataset. "
        "Plain TripletDataset instances do not retain the cell-level matrix needed for hold-out NLL validation."
    )


def evaluate_holdout_nll(
    adata,
    *,
    gmm_key: str,
    patient_col: str,
    label_col: str | None = None,
    train_frac: float | None = None,
    val_frac: float = 0.25,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Return stored patient-wise train/validation NLL diagnostics for a fitted GMM schema.

    Parameters
    ----------
    adata
        AnnData object containing the fitted GMM schema.
    gmm_key
        Key under ``adata.uns`` that stores the fitted GMM.
    patient_col
        Column in ``adata.obs`` identifying the patient-level distributions.
    label_col
        Optional label column to carry into the output table.
    train_frac
        Retained for backward compatibility. Stored diagnostics are returned;
        no new split is generated here.
    val_frac
        Retained for backward compatibility. Stored diagnostics are returned;
        no new split is generated here.
    random_seed
        Retained for backward compatibility. Stored diagnostics are returned;
        no randomization is performed here.

    Returns
    -------
    pandas.DataFrame
        One row per patient/distribution with fit-time train/validation NLL
        diagnostics from held-out-k selection.
    """
    if gmm_key not in adata.uns:
        raise KeyError(f"GMM key {gmm_key!r} not found in adata.uns.")

    gmm_cfg = dict(adata.uns[gmm_key])
    selection = dict(gmm_cfg.get("selection") or {})
    if selection.get("metric") != "heldout_nll":
        raise ValueError(
            f"GMM key {gmm_key!r} does not store heldout_nll selection metadata. "
            "Fit the GMM with k_selection_metric='heldout_nll' first."
        )

    distribution_ids = _resolve_distribution_ids(adata, gmm_cfg=gmm_cfg, patient_col=patient_col)
    distribution_index = get_distribution_index(
        adata,
        distribution_col=patient_col,
        distribution_ids=distribution_ids,
    )
    distribution_n_components = _resolve_distribution_n_components(gmm_cfg, n_distributions=len(distribution_ids))
    label_by_distribution = (
        None if label_col is None else _label_map(adata, patient_col=patient_col, label_col=label_col)
    )
    distribution_counts = np.bincount(distribution_index, minlength=len(distribution_ids))
    stored_train_frac = selection.get("train_frac")
    stored_val_frac = selection.get("val_frac")

    records: list[dict[str, object]] = []
    scores_by_distribution = selection.get("scores_by_distribution")
    if scores_by_distribution is not None:
        best_k_by_distribution = {
            str(dist_id): int(best_k) for dist_id, best_k in dict(selection.get("best_k_by_distribution", {})).items()
        }
        for dist_idx, distribution_id in enumerate(distribution_ids.tolist()):
            distribution_id = str(distribution_id)
            if distribution_id not in scores_by_distribution:
                raise ValueError(
                    f"Missing stored score trace for distribution {distribution_id!r}. "
                    "Please rerun fit_gmm with the current ggml_ot version."
                )
            best_k = best_k_by_distribution.get(distribution_id, int(distribution_n_components[dist_idx]))
            selected_record = _record_for_best_k(
                scores_by_distribution[distribution_id],
                best_k=best_k,
                context=f"distribution {distribution_id!r}",
            )
            record = {
                "distribution_id": distribution_id,
                "distribution_index": int(dist_idx),
                "n_components": int(best_k),
                "n_cells_total": int(distribution_counts[dist_idx]),
                "n_train_cells": int(selected_record["n_train_cells"]),
                "n_val_cells": int(selected_record["n_val_cells"]),
                "train_nll": float(selected_record["train_nll"]),
                "validation_nll": float(selected_record["validation_nll"]),
                "nll_gap": float(selected_record["nll_gap"]),
                "train_frac": np.nan if stored_train_frac is None else float(stored_train_frac),
                "val_frac": np.nan if stored_val_frac is None else float(stored_val_frac),
                "component_sharing": str(gmm_cfg.get("component_sharing", "sample_specific")),
                "covariance_type": str(gmm_cfg.get("covariance_type", "full")),
                "use_rep": gmm_cfg.get("use_rep"),
                "gmm_key": gmm_key,
            }
            if label_by_distribution is not None:
                record["label"] = label_by_distribution[distribution_id]
            records.append(record)
    else:
        selected_record = _record_for_best_k(
            selection.get("scores", {}),
            best_k=int(selection["best_k"]),
            context="global selection trace",
        )
        records.append(
            {
                "distribution_id": "__global__",
                "distribution_index": 0,
                "n_components": int(selection["best_k"]),
                "n_cells_total": int(adata.n_obs),
                "n_train_cells": int(selected_record["n_train_cells"]),
                "n_val_cells": int(selected_record["n_val_cells"]),
                "train_nll": float(selected_record["train_nll"]),
                "validation_nll": float(selected_record["validation_nll"]),
                "nll_gap": float(selected_record["nll_gap"]),
                "train_frac": np.nan if stored_train_frac is None else float(stored_train_frac),
                "val_frac": np.nan if stored_val_frac is None else float(stored_val_frac),
                "component_sharing": str(gmm_cfg.get("component_sharing", "global")),
                "covariance_type": str(gmm_cfg.get("covariance_type", "full")),
                "use_rep": gmm_cfg.get("use_rep"),
                "gmm_key": gmm_key,
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_holdout_nll(
    results: pd.DataFrame,
    *,
    groupby_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize patient-wise hold-out NLL diagnostics.

    Parameters
    ----------
    results
        Output from :func:`evaluate_holdout_nll`.
    groupby_cols
        Optional columns used to create per-group summaries.

    Returns
    -------
    pandas.DataFrame
        Summary dataframe with flat, user-facing columns:
        ``count``, ``mean_train_nll``, ``sd_train_nll``,
        ``mean_validation_nll``, ``sd_validation_nll``,
        ``mean_nll_gap``, and ``sd_nll_gap``.
    """
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame(columns=_SUMMARY_COLUMN_ORDER)

    metrics = ["train_nll", "validation_nll", "nll_gap"]
    group_cols = list(groupby_cols or ["group"])
    if groupby_cols is None:
        results_df = results_df.assign(group="All")

    grouped = results_df.groupby(group_cols, dropna=False, sort=False)
    summary = grouped[metrics].agg(["mean", "std"]).fillna(0.0)
    summary.columns = [
        "mean_train_nll",
        "sd_train_nll",
        "mean_validation_nll",
        "sd_validation_nll",
        "mean_nll_gap",
        "sd_nll_gap",
    ]
    summary.insert(0, "count", grouped.size().astype(int))
    summary = summary.reindex(columns=_SUMMARY_COLUMN_ORDER)
    if len(group_cols) == 1:
        summary.index = pd.Index(summary.index.tolist(), name=group_cols[0])
    return summary


def validate_gmm(
    data,
    *,
    gmm_key: str | None = None,
    patient_col: str | None = None,
    label_col: str | None = None,
    train_frac: float | None = None,
    val_frac: float = 0.25,
    random_seed: int | None = None,
    groupby_cols: Sequence[str] | None = None,
    plot: bool = True,
    plot_kwargs: Mapping[str, object] | None = None,
) -> GMMValidationReport:
    """Run a one-call hold-out validation pass for a fitted GMM.

    Parameters
    ----------
    data
        AnnData object or AnnData-backed dataset containing a fitted GMM schema.
    gmm_key, patient_col, label_col
        Validation metadata. When ``data`` is an AnnData-backed dataset,
        ``gmm_key`` defaults to the same key used by :func:`ggml_ot.gmm.fit_gmm`
        and ``patient_col`` / ``label_col`` default to the dataset attributes.
    train_frac, val_frac, random_seed
        Retained for backward compatibility. Validation now reads the stored
        fit-time held-out diagnostics rather than generating a fresh split.
    groupby_cols
        Optional columns used to group the summary table.
    plot
        Whether to create the appendix-style validation boxplot.
    plot_kwargs
        Optional keyword arguments forwarded to
        :func:`ggml_ot.plot.eval.gmm_fit_validation_boxplot`.

    Returns
    -------
    GMMValidationReport
        Bundle with patient-level rows, summary table, and optional plot axis.
    """
    adata, resolved_gmm_key, resolved_patient_col, resolved_label_col = _resolve_validation_target(
        data,
        gmm_key=gmm_key,
        patient_col=patient_col,
        label_col=label_col,
    )
    results = evaluate_holdout_nll(
        adata,
        gmm_key=resolved_gmm_key,
        patient_col=resolved_patient_col,
        label_col=resolved_label_col,
        train_frac=train_frac,
        val_frac=val_frac,
        random_seed=random_seed,
    )
    summary = summarize_holdout_nll(results, groupby_cols=groupby_cols)

    ax = None
    if plot:
        from ggml_ot.plot.eval import gmm_fit_validation_boxplot

        ax = gmm_fit_validation_boxplot(results, **dict(plot_kwargs or {}))

    return GMMValidationReport(results=results, summary=summary, ax=ax)


__all__ = [
    "evaluate_holdout_nll",
    "summarize_holdout_nll",
    "validate_gmm",
    "GMMValidationReport",
]
