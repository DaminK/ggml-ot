=============================
Analysis tools: :code:`.gene`
=============================

.. module:: ggml_ot.gene

.. currentmodule:: ggml_ot

The `ggml_ot.gene` module provides tools for identifying and visualizing the most important genes
in low-dimensional embeddings of single-cell data. It includes functions to compute gene importance
from an AnnData object, perform enrichment analysis on top-ranked genes, and visualize enriched
biological terms. These functions help you understand the learned embeddings and identify the genes
that influence cell differences or key components in your data.

+++++++++++++++++++++++++++++++++++++++
Rank genes in Loadings/Components
+++++++++++++++++++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   gene.ranking

+++++++++++++++++++++++++++++++++++++++
Gene Enrichment per Loadings/Components
+++++++++++++++++++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   gene.enrichment
