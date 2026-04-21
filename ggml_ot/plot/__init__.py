from .plotting import (
    distribution,
    clustermap_embedding,
)

from .embedding import emb as embedding

from .clustermap import clustermap

from .contour import contour_hyperparams

from .eval import table, confusion_matrix

from .subspace import (
    scatter_subspace,
    scatter_3d,
    ellipse_overlay,
    panel_subspaces,
    plot_gmm_panel,
    latent_gmm,
    panel_synth_dataset,
)

from .latent_axes import latent_axes
from .enrichment import latent_enrichment, gmm_component_enrichment
from .gmm import gmm_components

__all__ = [
    "embedding",
    "clustermap",
    "clustermap_embedding",
    "contour_hyperparams",
    "distribution",
    "table",
    "confusion_matrix",
    "scatter_subspace",
    "scatter_3d",
    "ellipse_overlay",
    "panel_subspaces",
    "plot_gmm_panel",
    "latent_gmm",
    "panel_synth_dataset",
    "latent_axes",
    "latent_enrichment",
    "gmm_component_enrichment",
    "gmm_components",
]
