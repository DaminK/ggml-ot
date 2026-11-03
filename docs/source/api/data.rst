==============================
Dataset classes: :code:`.data`
==============================

.. module:: ggml_ot.data

.. currentmodule:: ggml_ot

The `ggml_ot.data` module provides classes and functions for creating, handling, and
processing datasets for `ggml-ot`. These classes can be created from both AnnData objects and synthetic data and
are compatible with PyTorch.
The module also provides methods to split datasets, train and test models and download datasets.

+++++++++++++++++++++++++++++++++++++++
TripletDataset
+++++++++++++++++++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   data.AnnData_TripletDataset
   data.TripletDataset

++++++++++++++++++++
CELLxGENE interface
++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   data.load_cellxgene


++++++++++++++++++++++++++
Included example datasets
++++++++++++++++++++++++++

Curated CELLxGENE cohort shortcuts that download the data and return a
ready-to-train :class:`~ggml_ot.data.AnnData_TripletDataset`. See
:doc:`../references` for the source publication of each cohort.

.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   data.breastcancer
   data.covid
   data.myocardial_infarction
   data.kidney_injury


+++++++++++++++++++++++
Generate synthetic data
+++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   data.from_synth
   data.from_synth_gmm
