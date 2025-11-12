=====================
Plotting: :code:`.pl`
=====================

.. module:: ggml_ot.pl

.. currentmodule:: ggml_ot


The `ggml_ot.pl` module provides a collection of functions to visualize distance matrices,
embeddings, distributions, and classification results. These plotting functions are designed
to work with `ggml_ot` datasets and outputs, offering options for dimensionality reduction,
plotting techniques, and customization of the visualizations.

+++++++++++++++++++
Patient-level plots
+++++++++++++++++++
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.clustermap
   pl.embedding
   pl.clustermap_embedding

+++++++++++++++++++
Evaluation
+++++++++++++++++++
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.table
   pl.confusion_matrix
   pl.contour_hyperparams
