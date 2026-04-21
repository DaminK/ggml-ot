"""Decoupler enrichment backend.

Wraps ``decoupler.mt.gsea`` for axis-level and component-level
gene score enrichment. Imported lazily — ``decoupler`` is an
optional dependency.
"""

from __future__ import annotations

import pandas as pd

from ggml_ot.settings import settings


def _format_pathway_label(pathway: str, *, resource: str, collection: str | None) -> str:
    """Create a human-readable label while preserving the raw pathway id."""
    label = str(pathway)

    if resource.lower() == "msigdb" and collection == "hallmark" and label.startswith("HALLMARK_"):
        label = label.removeprefix("HALLMARK_")
        label = label.replace("_", " ")

    return label


def run_enrichment(
    score_df: pd.DataFrame,
    *,
    group_col: str,
    gene_col: str = "gene",
    score_col: str = "score",
    resource: str = "MSigDB",
    collection: str = "hallmark",
    method: str = "gsea",
    organism: str = "human",
    min_n: int = 5,
) -> pd.DataFrame:
    """Run enrichment on pre-computed gene scores using decoupler.

    Parameters
    ----------
    score_df
        Long-format DataFrame with at least ``group_col``, ``gene_col``,
        and ``score_col`` columns. Each unique value of ``group_col``
        (an axis index or component id) defines one ranked gene list.
    group_col
        Column that identifies the group (``"axis"`` or ``"component"``).
    gene_col
        Column with gene names.
    score_col
        Column with signed gene scores.
    resource
        Resource name passed to ``dc.op.resource()``.
    collection
        Collection filter within the resource (e.g. ``"hallmark"``).
    method
        Enrichment method. Currently only ``"gsea"`` is supported.
    organism
        Organism for ``dc.op.resource()``.
    min_n
        Minimum number of targets per gene set.

    Returns
    -------
    pandas.DataFrame
        Long-format enrichment results with columns:
        ``<group_col>``, ``pathway``, ``pathway_label``, ``score``, ``pvalue``.
    """
    if method != "gsea":
        raise ValueError(f"Unsupported enrichment method: {method!r}. Use 'gsea'.")

    try:
        import decoupler as dc
    except ImportError:
        raise ImportError(
            "The 'decoupler' package is required for GSEA enrichment but is not installed. "
            "Install it with: pip install decoupler>=2.1,<3"
        ) from None

    # Fetch and filter gene set network using the decoupler 2.x API.
    net = dc.op.resource(resource, organism=organism)
    if collection is not None:
        if "collection" not in net.columns:
            raise ValueError(
                f"Resource {resource!r} has no 'collection' column. Available columns: {list(net.columns)}"
            )
        net = net[net["collection"] == collection].copy()
        if net.empty:
            raise ValueError(f"No gene sets found for collection={collection!r} in resource={resource!r}.")

    rename_map = {}
    if "geneset" in net.columns:
        rename_map["geneset"] = "source"
    if "genesymbol" in net.columns:
        rename_map["genesymbol"] = "target"
    if rename_map:
        net = net.rename(columns=rename_map)
    missing = {"source", "target"} - set(net.columns)
    if missing:
        raise ValueError(
            "Enrichment resource must provide source/target gene-set columns after normalization; "
            f"missing {sorted(missing)} from columns {list(net.columns)}"
        )

    # MSigDB resources can contain duplicate (source, target) rows after
    # collection filtering.  decoupler validates uniqueness, so deduplicate.
    net = net.drop_duplicates(subset=["source", "target"])
    pathway_label_map = {
        pathway: _format_pathway_label(pathway, resource=resource, collection=collection)
        for pathway in net["source"].astype(str).unique()
    }

    # Build a score matrix: rows = groups, columns = genes.
    # Use string labels for the index to avoid dtype mismatches with decoupler.
    groups = score_df[group_col].unique()
    group_labels = [str(g) for g in groups]
    genes = score_df[gene_col].unique()
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    import numpy as np

    mat = np.zeros((len(groups), len(genes)), dtype=np.float64)
    for row_i, grp in enumerate(groups):
        grp_df = score_df[score_df[group_col] == grp]
        for _, r in grp_df.iterrows():
            g_idx = gene_to_idx.get(r[gene_col])
            if g_idx is not None:
                mat[row_i, g_idx] = r[score_col]

    mat_df = pd.DataFrame(mat, index=group_labels, columns=genes)

    # decoupler 2.x returns (estimate_df, pvals_df) for DataFrame input.
    estimate, pvals = dc.mt.gsea(
        mat_df,
        net,
        tmin=min_n,
        seed=settings.random_seed,
    )

    # Assemble long-format results, mapping string labels back to originals.
    result_rows: list[dict] = []
    for grp, label in zip(groups, group_labels):
        for pathway in estimate.columns:
            result_rows.append(
                {
                    group_col: grp,
                    "pathway": pathway,
                    "pathway_label": pathway_label_map.get(str(pathway), str(pathway)),
                    "score": float(estimate.loc[label, pathway]),
                    "norm_score": float(estimate.loc[label, pathway]),
                    "pvalue": float(pvals.loc[label, pathway]),
                }
            )

    return pd.DataFrame(result_rows)
