from ggml_ot.data import TripletDataset, AnnData_TripletDataset
from ggml_ot.data.util import wraps

import inspect


@wraps(AnnData_TripletDataset)
def from_anndata(*args, **kwargs) -> AnnData_TripletDataset:
    return AnnData_TripletDataset(*args, **kwargs)


from_anndata.__signature__ = inspect.signature(AnnData_TripletDataset)


@wraps(TripletDataset)
def from_numpy(*args, **kwargs) -> TripletDataset:
    return TripletDataset(*args, **kwargs)


from_numpy.__signature__ = inspect.signature(TripletDataset)
