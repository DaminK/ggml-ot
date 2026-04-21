"""Covariance matrix helpers.

Centralises covariance validation, canonicalisation from diagonal to full form,
regularisation (symmetrise + jitter), and presence checks.
"""

from __future__ import annotations

import numpy as np
import torch
import warnings


# ---------------------------------------------------------------------------
# Presence check
# ---------------------------------------------------------------------------


def has_covariances(covariances) -> bool:
    """Return ``True`` if *covariances* is present and non-empty."""
    if covariances is None:
        return False

    if isinstance(covariances, torch.Tensor):
        return covariances.numel() > 0
    if isinstance(covariances, np.ndarray):
        return covariances.size > 0

    if len(covariances) == 0:
        return False

    first = covariances[0]
    if isinstance(first, torch.Tensor):
        return first.numel() > 0
    if isinstance(first, np.ndarray):
        return first.size > 0

    return True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_square_covariances(*, supports, covariances, identical_supports: bool):
    """Validate covariance layout for Gaussian OT computations.

    Covariances must be provided as full square matrices, not diagonal
    vectors.  Raises :class:`ValueError` on shape mismatches.
    """
    from ggml_ot._utils._array import to_numpy as _to_numpy

    def _as_list(x):
        arr = _to_numpy(x)
        if arr.ndim == 2:
            return [arr]
        if arr.ndim >= 3:
            return [arr[i] for i in range(arr.shape[0])]
        raise ValueError(f"Expected at least 2D array-like data, got shape={arr.shape}.")

    if identical_supports:
        supports_np = _to_numpy(supports)
        cov_np = _to_numpy(covariances)
        if supports_np.ndim != 2:
            raise ValueError(f"Expected identical-support means with shape (K, D), got {supports_np.shape}.")
        if cov_np.ndim != 3:
            raise ValueError(
                "Covariances must be full square matrices with shape (K, D, D) for "
                f"identical supports. Got {cov_np.shape}. "
                "Diagonal-vector storage is not supported at compute_OT level."
            )
        k, d = int(supports_np.shape[0]), int(supports_np.shape[1])
        if cov_np.shape[0] != k or cov_np.shape[1] != d or cov_np.shape[2] != d:
            raise ValueError(
                "Covariance/support shape mismatch for identical supports: "
                f"supports={supports_np.shape}, covariances={cov_np.shape}."
            )
        return

    supports_list = _as_list(supports)
    covariances_list = _as_list(covariances)
    if len(supports_list) != len(covariances_list):
        raise ValueError(
            "supports and covariances must contain the same number of distributions. "
            f"Got len(supports)={len(supports_list)}, len(covariances)={len(covariances_list)}."
        )

    for idx, (supports_i, cov_i) in enumerate(zip(supports_list, covariances_list)):
        s_np = _to_numpy(supports_i)
        c_np = _to_numpy(cov_i)
        if s_np.ndim != 2:
            raise ValueError(f"Each support must have shape (K_i, D); distribution idx={idx} has shape {s_np.shape}.")
        if c_np.ndim != 3:
            raise ValueError(
                "Each covariance block must be full square matrices with shape "
                f"(K_i, D, D); distribution idx={idx} has shape {c_np.shape}. "
                "Diagonal-vector storage is not supported at compute_OT level."
            )
        k_i, d_i = int(s_np.shape[0]), int(s_np.shape[1])
        if c_np.shape[0] != k_i or c_np.shape[1] != d_i or c_np.shape[2] != d_i:
            raise ValueError(
                "Covariance/support shape mismatch for distribution idx="
                f"{idx}: supports={s_np.shape}, covariances={c_np.shape}."
            )


def validate_mean_covariance_shapes(
    means: torch.Tensor,
    covariances: torch.Tensor,
    *,
    means_name: str,
    covariances_name: str,
) -> None:
    """Validate a single Gaussian-parameter block `(means, covariances)`."""
    if means.ndim != 2:
        raise ValueError(f"`{means_name}` must have shape (N, D). Got {tuple(means.shape)}.")
    if covariances.ndim != 3 or covariances.shape[-2] != covariances.shape[-1]:
        raise ValueError(
            f"Expected square covariances for `{covariances_name}` with shape (N, D, D). "
            f"Got {tuple(covariances.shape)}."
        )
    if covariances.shape[0] != means.shape[0] or covariances.shape[-1] != means.shape[-1]:
        raise ValueError(
            f"Mean/covariance shape mismatch for `{means_name}`/{covariances_name}: "
            f"{means_name}={tuple(means.shape)}, {covariances_name}={tuple(covariances.shape)}."
        )


def validate_cross_mean_covariance_shapes(
    means_i: torch.Tensor,
    covariances_i: torch.Tensor,
    means_j: torch.Tensor,
    covariances_j: torch.Tensor,
    *,
    means_i_name: str = "mu_i",
    covariances_i_name: str = "sigma_i",
    means_j_name: str = "mu_j",
    covariances_j_name: str = "sigma_j",
) -> None:
    """Validate cross-Gaussian parameter blocks for distance computations."""
    validate_mean_covariance_shapes(
        means_i,
        covariances_i,
        means_name=means_i_name,
        covariances_name=covariances_i_name,
    )
    validate_mean_covariance_shapes(
        means_j,
        covariances_j,
        means_name=means_j_name,
        covariances_name=covariances_j_name,
    )


# ---------------------------------------------------------------------------
# Canonicalisation (diag → full)
# ---------------------------------------------------------------------------


def canonicalize_covariances(
    covariances: np.ndarray,
    supports: np.ndarray,
    *,
    covariance_type: str,
) -> np.ndarray:
    """Canonicalize covariance representation to full square form ``(..., D, D)``.

    For ``covariance_type='diag'``, diagonal-vector storage ``(..., D)`` is
    accepted and embedded into matrices.  Full-matrix storage is returned
    as-is for both covariance types.
    """
    cov_np = np.asarray(covariances, dtype=np.float64)
    mu_np = np.asarray(supports, dtype=np.float64)

    if covariance_type not in {"diag", "full"}:
        raise ValueError(f"Unsupported covariance_type: {covariance_type!r}.")
    if mu_np.ndim < 2:
        raise ValueError(f"supports must have shape (..., D). Got {mu_np.shape}.")

    is_full = cov_np.ndim == mu_np.ndim + 1 and cov_np.shape[:-1] == mu_np.shape and cov_np.shape[-1] == mu_np.shape[-1]
    if is_full:
        return cov_np

    if covariance_type == "diag" and cov_np.shape == mu_np.shape:
        d = int(mu_np.shape[-1])
        return cov_np[..., :, np.newaxis] * np.eye(d, dtype=np.float64)

    raise ValueError(
        "Unexpected covariance shape for GMM payload: "
        f"supports shape={mu_np.shape}, covariances shape={cov_np.shape}, "
        f"covariance_type={covariance_type!r}."
    )


# ---------------------------------------------------------------------------
# Regularisation (torch)
# ---------------------------------------------------------------------------


def _sanitize_full_covariances(var: torch.Tensor, eps: float) -> torch.Tensor:
    """Symmetrize and clamp covariance tensors to finite, PSD-like values."""
    eps_val = float(eps)
    d = int(var.shape[-1])
    eye = torch.eye(d, device=var.device, dtype=var.dtype).view(1, 1, d, d)
    var = 0.5 * (var + var.transpose(-1, -2))
    var = torch.nan_to_num(var, nan=eps_val, posinf=1.0 / eps_val, neginf=eps_val)
    return var + eps_val * eye


def symmetrise_and_jitter_covariances(covs: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Symmetrise projected covariances and add a tiny diagonal jitter.

    This guards against numerical instability in the matrix square root
    used by the Bures distance computation.
    """
    if float(eps) <= 0:
        return covs

    squeeze_back = covs.ndim == 2
    if squeeze_back:
        covs = covs.unsqueeze(0)

    covs = 0.5 * (covs + covs.transpose(-1, -2))
    covs = torch.nan_to_num(covs, nan=0.0, posinf=0.0, neginf=0.0)
    eye = torch.eye(covs.shape[-1], dtype=covs.dtype, device=covs.device).unsqueeze(0)
    covs = covs + float(eps) * eye
    return covs[0] if squeeze_back else covs


def apply_singularity_handling(
    var: np.ndarray,
    *,
    covariance_type: str,
    eps: float,
    singularity_handling: str,
    n_components: int,
) -> np.ndarray:
    """Apply singularity handling to fitted full covariance matrices.

    Parameters
    ----------
    var
        Covariance array with shape ``(1, K, D, D)`` or ``(K, D, D)``.
    covariance_type
        ``"full"`` or ``"diag"``. Only ``"full"`` is processed; diagonal
        covariances are returned unchanged.
    eps
        Eigenvalue floor.  Components whose minimum eigenvalue is below
        this threshold are considered near-singular.
    singularity_handling
        ``"guarded"`` (default): apply eigenvalue-floor clamping
        (``V·max(Λ,eps)·Vᵀ``); raise :class:`ValueError` if any component
        is still singular after clamping (fix attempted, outcome validated).
        ``"robust"``: apply eigenvalue-floor clamping; emit a
        :class:`UserWarning` if any component is still singular after clamping
        (permissive — never breaks execution).
        ``"strict"``: raise :class:`ValueError` immediately if any eigenvalue
        is below *eps*; no clamping is attempted.
    n_components
        Total number of mixture components (used in warning/error messages).

    Returns
    -------
    np.ndarray
        Covariance array in the same shape as *var*, with eigenvalue-floor
        applied for ``"guarded"`` and ``"robust"`` modes.

    Raises
    ------
    ValueError
        For ``"strict"``: any eigenvalue below *eps*.
        For ``"guarded"``: any eigenvalue still below *eps* after clamping.
        For all modes: non-finite covariance entries.
    """
    if covariance_type != "full":
        return var

    var_np = np.asarray(var, dtype=np.float64)
    original_shape = var_np.shape

    if var_np.ndim == 4 and var_np.shape[0] == 1:
        mats = var_np[0]  # (K, D, D)
    elif var_np.ndim == 3:
        mats = var_np
    else:
        return var  # unexpected shape — leave unchanged

    n_comps = mats.shape[0]
    result = mats.copy()
    floor = max(float(eps), 1e-12)

    _SINGULARITY_MESSAGE = (
        "This typically occurs when n_components exceeds the effective rank of the data — "
        "i.e., the signal subspace rank is smaller than n_features, or too few observations "
        "are assigned to a component to estimate a full-rank covariance. "
        "Consider reducing n_components if numerical instabilities occur downstream."
    )

    for k in range(n_comps):
        cov = result[k]
        if not np.all(np.isfinite(cov)):
            raise ValueError(
                f"GMM component {k} has non-finite covariance entries after fitting. "
                "This is likely a numerical failure during EM; try increasing `eps`."
            )
        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eig = float(np.min(eigvals))

        if min_eig >= floor:
            continue  # component is fine

        if singularity_handling == "strict":
            raise ValueError(
                f"GMM component {k}/{n_components} has a near-singular covariance matrix "
                f"(minimum eigenvalue={min_eig:.3e}, eps={floor:.3e}). "
                + _SINGULARITY_MESSAGE
                + " Or use singularity_handling='guarded' or 'robust' to attempt eigenvalue-floor clamping."
            )

        # guarded / robust: eigenvalue-floor — reconstruct V · max(Λ, floor) · Vᵀ
        clamped = np.maximum(eigvals, floor)
        result[k] = (eigvecs * clamped) @ eigvecs.T

        # Check residual singularity after reconstruction (floating-point safety).
        min_eig_after = float(np.min(np.linalg.eigvalsh(result[k])))
        if min_eig_after < floor:
            if singularity_handling == "robust":
                warnings.warn(
                    f"GMM component {k}/{n_components} is still near-singular after eigenvalue-floor "
                    f"clamping (minimum eigenvalue={min_eig_after:.3e}, eps={floor:.3e}). " + _SINGULARITY_MESSAGE,
                    UserWarning,
                    stacklevel=4,
                )
            else:  # guarded
                raise ValueError(
                    f"GMM component {k}/{n_components} is still near-singular after eigenvalue-floor "
                    f"clamping (minimum eigenvalue={min_eig_after:.3e}, eps={floor:.3e}). "
                    + _SINGULARITY_MESSAGE
                    + " Try increasing `eps` or reducing n_components."
                )

    if original_shape != mats.shape:
        result = result.reshape(original_shape)

    return result


# ---------------------------------------------------------------------------
# Dataset-level covariance presence checks
# ---------------------------------------------------------------------------


def has_dataset_covariances(dataset) -> bool:
    """Return ``True`` when a dataset carries non-empty covariances."""
    covariances = getattr(dataset, "covariances", None)
    try:
        has = covariances is not None and len(covariances) > 0 and covariances.shape[-1] != 0
    except TypeError:
        has = covariances is not None

    return has


def require_dataset_covariances(dataset, *, caller: str = "train") -> None:
    """Require dataset covariances and raise with a clear message when absent."""
    if not has_dataset_covariances(dataset):
        raise ValueError(
            f"{caller} requires a dataset with covariances. Please fit a GMM to your dataset using .fit_gmm() before training."
        )
