from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class GMMFitConfig:
    """Fit configuration for the torch-native backend."""

    n_components: int
    covariance_type: Literal["diag", "full"] = "full"
    max_iter: int = 100
    tol: float = 1e-3
    n_init: int = 1
    eps: float = 1e-4
    singularity_handling: Literal["guarded", "robust", "strict"] = "guarded"


@dataclass
class GMMResult:
    """Canonical fitted GMM container."""

    mu: np.ndarray
    var: np.ndarray
    pi: np.ndarray
    responsibilities: np.ndarray | None = None
    bic: float | None = None
    model: Any = None


__all__ = [
    "GMMFitConfig",
    "GMMResult",
]
