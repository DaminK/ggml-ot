from . import TripletDataset, AnnData_TripletDataset
from ggml_ot._utils._docs import wraps
from anndata import AnnData

import inspect


@wraps(AnnData_TripletDataset)
def from_anndata(*args, **kwargs) -> AnnData_TripletDataset:
    return AnnData_TripletDataset(*args, **kwargs)


from_anndata.__signature__ = inspect.signature(AnnData_TripletDataset)


@wraps(AnnData_TripletDataset.to_anndata)
def to_anndata(dataset: AnnData_TripletDataset) -> AnnData:
    return dataset.to_anndata()


@wraps(TripletDataset)
def from_numpy(*args, **kwargs) -> TripletDataset:
    return TripletDataset(*args, **kwargs)


from_numpy.__signature__ = inspect.signature(TripletDataset)
