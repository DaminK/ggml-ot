=====================
Plotting: :code:`.pl`
=====================

.. module:: ggml_ot.pl

.. currentmodule:: ggml_ot


The `ggml_ot.pl` module provides a collection of functions to visualize distance matrices,
embeddings, and classification results. These plotting functions are designed
to work with `ggml_ot` datasets and outputs, offering options for dimensionality reduction,
plotting techniques, and customization of the visualizations.

+++++++++++++++++++++++++++++++
Displaying and saving plots
+++++++++++++++++++++++++++++++
Plots are displayed automatically in interactive environments such as Jupyter
notebooks and IPython. In scripts, pass ``show=True`` to display a plot.

Pass ``show=False`` to suppress immediate display and return the Matplotlib or
seaborn plot object. This lets you customize the plot and display it later with
``plt.show()``.

To save a plot using its default filename in the directory configured by
``ggml_ot.settings.figdir`` (see :doc:`settings`), pass ``save=True``. Provide
a different filename with ``save="my_plot.png"``.

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

+++++++++++++++++++
Subspace / GMM
+++++++++++++++++++
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.scatter_subspace
   pl.scatter_3d
   pl.ellipse_overlay
   pl.panel_subspaces
   pl.plot_gmm_panel
   pl.panel_synth_dataset
