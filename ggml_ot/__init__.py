import sys

from .ggml import train

from .interface import from_anndata, from_numpy

from . import data as data

from . import plot as pl

from . import gene as gene

from .benchmark import train_test, tune, test

__all__ = [
    "train",
    "from_anndata",
    "from_numpy",
    "test",
    "train_test",
    "tune",
    "data",
    "pl",
    "gene",
]

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pl"]})
