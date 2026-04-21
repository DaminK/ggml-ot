import sys

from .settings import settings

from .optimization.api import train, train_emd2, train_sinkhorn, train_gmm, train_history

from .data.interface import from_anndata, from_numpy

from . import data as data

from . import plot as pl

from . import gene as gene

from . import gmm as gmm

from .benchmark import train_test, tune, test

__all__ = [
    "train",
    "train_emd2",
    "train_sinkhorn",
    "train_gmm",
    "train_history",
    "from_anndata",
    "from_numpy",
    "test",
    "train_test",
    "tune",
    "data",
    "pl",
    "gmm",
    "gene",
    "settings",
]

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pl"]})
