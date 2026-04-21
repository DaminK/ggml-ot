from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ggml_ot.data import TripletDataset, AnnData_TripletDataset


# TripletLoss and Linear Map parameters
@dataclass
class GGMLParams:
    alpha: float
    n_comps: int


# Regularization parameters
RegType = Literal[1, "1", 2, "2", "fro", "nuc"]
CanonicalRegType = Literal[1, "fro", "nuc"]


def canonicalize_reg_type(reg_type: RegType) -> CanonicalRegType:
    """Map accepted `reg_type` aliases to canonical norm identifiers."""
    if isinstance(reg_type, Integral):
        reg_int = int(reg_type)
        if reg_int == 1:
            return 1
        if reg_int == 2:
            return "fro"

    if isinstance(reg_type, str):
        reg_key = reg_type.strip().lower()
        if reg_key == "1":
            return 1
        if reg_key in {"2", "fro"}:
            return "fro"
        if reg_key == "nuc":
            return "nuc"

    raise ValueError(
        f"Unsupported reg_type={reg_type!r}. Use one of: 1/'1' (L1), 2/'2'/'fro' (L2/Frobenius), 'nuc' (nuclear norm)."
    )


@dataclass
class RegularizationParams:
    reg: float
    reg_type: RegType


# MI Regularization parameters
@dataclass
class MIRegularizationParams(RegularizationParams):
    mi_reg: float | bool
    mi_reg_weighting: str | None
    eps: float = 1e-4
    mi_sqrt: bool = False


# Optimization parameters (e.g. Adam)
@dataclass
class OptimizationParams:
    lr: float
    max_iter: int
    stop_thr: float | None = None


# DataLoader parameters
@dataclass
class DataLoaderParams:
    dataset: TripletDataset | AnnData_TripletDataset
    batch_size: int
    train_size: float | None = None


# Generic OT parameters
@dataclass
class OTParams:
    identical_supports: bool = False
    eps: float = 1e-4
    squared_ground_cost: bool = False


# Sinkhorn parameters for entropic regularization (used in batched triplet_loss)
@dataclass
class SinkhornParams(OTParams):
    entropic_reg: float = 1
    sinkhorn_max_iter: int = 100
    sinkhorn_stop_thr: float | None = 1e-6


# Wasserstein-Bures parameters for Gaussian distributions
@dataclass
class WassersteinBuresParams(SinkhornParams):
    entropic_reg: float = 0
    diag_bures_approx: bool = False
