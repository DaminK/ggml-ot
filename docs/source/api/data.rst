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


+++++++++++++++++++++++
Generate synthetic data
+++++++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   data.from_synth
