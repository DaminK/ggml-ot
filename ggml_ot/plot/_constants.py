"""Plotting constants shared across :mod:`ggml_ot.plot` modules.

These dicts are the single source of truth for label-type-aware plotting
defaults (embedding method, palette, method display name). Clustermap,
embedding, and composite plots all read from here so the colormap on the
embedding side matches the colormap used for the clustermap row/column
annotations in the joined ``clustermap_embedding`` view.
"""

from __future__ import annotations


# Default embedding method per label_type.
EMB_LABELTYPE = {
    "categorical": "umap",
}

# Default colormap per label_type. Shared between the embedding scatter
# palette and the clustermap row/column annotation colors so the joined
# clustermap_embedding plot reads consistently.
PALETTE_LABELTYPE = {
    "categorical": "Set2",
}

# Display labels for embedding methods, used in subplot titles like
# "Embedding (UMAP)" / "Embedding (Diffusion)".
EMB_DISPLAY = {
    "umap": "UMAP",
    "tsne": "t-SNE",
    "mds": "MDS",
    "diffusion": "Diffusion",
    "fast_diffusion": "Fast Diffusion",
}
