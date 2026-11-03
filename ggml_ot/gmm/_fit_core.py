from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class GMMFitConfig:
    """Fit configuration for the torch-native backend."""

    n_components: int
    covariance_type: Literal["diag", "full", "lowrank"] = "full"
    max_iter: int = 100
    tol: float = 1e-3
    n_init: int = 1
    eps: float = 1e-4
    singularity_handling: Literal["guarded", "robust", "strict"] = "guarded"
    # Low-rank (MFA) covariance parameters; only used when covariance_type == "lowrank".
    factor_rank: int | None = None
    psi_floor: float = 1e-4


def free_parameter_count(k: int, d: int, covariance_type: str, factor_rank: int | None = None) -> int:
    """Number of free parameters for a GMM, used by AIC/BIC.

    Single source of truth shared by ``GaussianMixture.bic`` and the
    information-criterion scoring in ``_select_k_comps``.

    Counts ``k * d`` means and ``k - 1`` mixing weights, plus the covariance
    parameters:

    - ``"diag"``: ``k * d``
    - ``"full"``: ``k * d * (d + 1) / 2``
    - ``"lowrank"`` (MFA, component-specific ``Psi_k``):
      ``k * (d * q - q * (q - 1) / 2)`` loadings (rotation-identifiability
      corrected) ``+ k * d`` uniquenesses.
    """
    k = int(k)
    d = int(d)
    if covariance_type == "diag":
        cov_params = k * d
    elif covariance_type == "lowrank":
        if factor_rank is None:
            raise ValueError("free_parameter_count requires factor_rank for covariance_type='lowrank'.")
        q = int(factor_rank)
        cov_params = k * (d * q - q * (q - 1) // 2) + k * d
    else:  # "full"
        cov_params = k * (d * (d + 1) // 2)
    return (k * d) + cov_params + (k - 1)


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
    "free_parameter_count",
]
