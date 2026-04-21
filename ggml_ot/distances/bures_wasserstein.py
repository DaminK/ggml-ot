import torch

from ggml_ot._utils._linalg import matrix_sqrt
from ggml_ot import settings
from ggml_ot._utils._array import cast_many_to_torch, to_output_backend
from ggml_ot._utils._covariance import (
    validate_cross_mean_covariance_shapes,
    validate_mean_covariance_shapes,
)
from ggml_ot._utils._covariance import symmetrise_and_jitter_covariances as _regularize_projected_covariances
from ggml_ot._utils._transformation import project_gaussians
from ggml_ot._utils._validate import validate_gaussian_ground_metric

from torch.distributions import MultivariateNormal, kl_divergence


def bures_covariance_distance(
    cov_i: torch.Tensor,
    cov_j: torch.Tensor,
    *,
    diag_approx: bool = False,
    eps: float = 1e-4,
    symmetric: bool = False,
) -> torch.Tensor:
    """Compute Bures covariance term between covariance sets.

    Batched implementation: replaces the former Python triple-loop over
    (batch, n_i, n_j) with broadcasted matmul and a single batched
    ``eigvalsh`` call for significantly improved throughput.

    Notes
    -----
    In the full Bures path (`diag_approx=False`), `eps` is applied as an
    unconditional symmetrise+jitter step on `cov_i` before the Cholesky square
    root, and as a clamp floor on the cross-product eigenvalues. The jitter
    both prevents failure on near-singular projected covariances and bounds the
    Cholesky condition number to avoid gradient explosion.
    """
    if cov_i.ndim < 3 or cov_j.ndim < 3:
        raise ValueError(
            f"Expected covariance tensors with ndim>=3 (..., N, D, D). Got cov_i={tuple(cov_i.shape)}, cov_j={tuple(cov_j.shape)}."
        )
    if cov_i.shape[-1] != cov_i.shape[-2] or cov_j.shape[-1] != cov_j.shape[-2]:
        raise ValueError(
            f"Expected square covariance matrices on trailing dims. Got cov_i={tuple(cov_i.shape)}, cov_j={tuple(cov_j.shape)}."
        )

    if diag_approx:
        diag_i = torch.diagonal(cov_i, dim1=-2, dim2=-1)
        diag_j = torch.diagonal(cov_j, dim1=-2, dim2=-1)
        root_i = torch.sqrt(torch.clamp(diag_i, min=0.0))
        root_j = torch.sqrt(torch.clamp(diag_j, min=0.0))
        return torch.cdist(root_i, root_j, p=2) ** 2

    if cov_i.shape[:-3] != cov_j.shape[:-3]:
        raise ValueError(
            f"Leading batch dimensions must match. Got cov_i={tuple(cov_i.shape)}, cov_j={tuple(cov_j.shape)}."
        )

    n_i, n_j = int(cov_i.shape[-3]), int(cov_j.shape[-3])
    d_i, d_j = int(cov_i.shape[-1]), int(cov_j.shape[-1])
    if d_i != d_j:
        raise ValueError(f"Covariance feature dimensions must match. Got D_i={d_i}, D_j={d_j}.")

    batch_shape = cov_i.shape[:-3]
    cov_i_b = cov_i.reshape(-1, n_i, d_i, d_i)
    cov_j_b = cov_j.reshape(-1, n_j, d_j, d_j)
    B = cov_i_b.shape[0]

    _eps = float(eps) if eps is not None and float(eps) > 0 else 0.0

    # Traces: (B, n_i) and (B, n_j)
    tr_i = torch.diagonal(cov_i_b, dim1=-2, dim2=-1).sum(-1)
    tr_j = torch.diagonal(cov_j_b, dim1=-2, dim2=-1).sum(-1)

    # Batched square root of cov_i: (B, n_i, D, D)
    # Use Cholesky for stable backward when gradients flow through
    # cross-distribution pairs; eigendecomposition for symmetric (pairwise).
    # Apply eps jitter before Cholesky to handle near-singular GMM covariances
    # (e.g. padded phantom components with eps*I).
    use_cholesky = (not symmetric) and (cov_i.requires_grad or cov_j.requires_grad)
    if use_cholesky:
        # Jitter unconditionally before Cholesky. This serves two purposes:
        # (1) prevents LinAlgError on near-singular projected covariances (e.g. when W
        #     develops near-zero singular values, making W·Σ·Wᵀ rank-deficient); and
        # (2) bounds the condition number of the Cholesky factor, preventing gradient
        #     explosion through L⁻¹ for nearly-singular matrices that would otherwise
        #     "pass" Cholesky but produce enormous gradients.
        # The eps shift is small relative to typical covariance magnitudes, so cost-matrix
        # bias is negligible for well-conditioned components.
        cov_i_b = _regularize_projected_covariances(cov_i_b, eps=_eps)
        root_i = torch.linalg.cholesky(cov_i_b)  # (B, n_i, D, D) lower triangular
        root_i_t = root_i.transpose(-1, -2)
    else:
        root_i = matrix_sqrt(cov_i_b, eps=_eps)  # (B, n_i, D, D) symmetric
        root_i_t = root_i  # symmetric square root: transpose is identity

    # Batched cross product: root_i @ cov_j @ root_i^T  →  (B, n_i, n_j, D, D)
    root_i_exp = root_i.unsqueeze(2)  # (B, n_i, 1, D, D)
    root_i_t_exp = root_i_t.unsqueeze(2)  # (B, n_i, 1, D, D)
    cov_j_exp = cov_j_b.unsqueeze(1)  # (B, 1, n_j, D, D)
    cross = torch.matmul(torch.matmul(root_i_exp, cov_j_exp), root_i_t_exp)  # (B, n_i, n_j, D, D)

    # Symmetrise + jitter for numerical stability (reuse existing utility)
    cross = _regularize_projected_covariances(cross, eps=_eps)

    # Batched eigenvalue computation: tr(sqrt(cross))
    cross_flat = cross.reshape(-1, d_i, d_i)
    eig_cross = torch.linalg.eigvalsh(cross_flat)  # (-1, D)
    tr_cross = torch.sqrt(torch.clamp(eig_cross, min=_eps)).sum(-1)
    tr_cross = tr_cross.reshape(B, n_i, n_j)

    # Assemble pairwise Bures distances: tr(Σ_i) + tr(Σ_j) - 2·tr(√(√Σ_i·Σ_j·√Σ_i))
    out = tr_i.unsqueeze(-1) + tr_j.unsqueeze(-2) - 2.0 * tr_cross

    if symmetric and n_i == n_j:
        out = 0.5 * (out + out.transpose(-2, -1))

    return out.reshape(*batch_shape, n_i, n_j)


def _compute_bures_term(
    sigma_i: torch.Tensor,
    sigma_j: torch.Tensor,
    *,
    diag_bures_approx: bool,
    eps: float,
    symmetric: bool,
) -> torch.Tensor:
    """Shared Bures-term assembly for pairwise/cross Gaussian distance."""
    if diag_bures_approx:
        return bures_covariance_distance(
            sigma_i,
            sigma_j,
            diag_approx=True,
            eps=eps,
            symmetric=symmetric,
        )
    sigma_i = _regularize_projected_covariances(sigma_i, eps=eps)
    sigma_j = _regularize_projected_covariances(sigma_j, eps=eps)
    return bures_covariance_distance(
        sigma_i,
        sigma_j,
        diag_approx=False,
        eps=eps,
        symmetric=symmetric,
    )


def _resolve_gaussian_ground_metric_inputs(
    mu_i: torch.Tensor,
    sigma_i: torch.Tensor,
    w,
    *,
    as_numpy: bool,
    mu_j: torch.Tensor | None = None,
    sigma_j: torch.Tensor | None = None,
):
    """Resolve gaussian metric mode: KL/euclidean string or linear-map projection.

    Returns
    -------
    tuple
        `(early_result, resolved_mu_i, resolved_sigma_i, resolved_mu_j, resolved_sigma_j)`.
        `early_result` is set only for the KL-divergence fast path.
    """
    has_cross = (mu_j is not None) or (sigma_j is not None)
    if has_cross and (mu_j is None or sigma_j is None):
        raise ValueError("Pass both `mu_j` and `sigma_j` together for cross-distance mode.")

    if isinstance(w, str):
        validate_gaussian_ground_metric(w)
        if w == "kl_divergence":
            if has_cross:
                kl_out = compute_cross_kl_gaussians(mu_i, sigma_i, mu_j, sigma_j)
            else:
                kl_out = compute_pairwise_kl_gaussians(mu_i, sigma_i)
            return to_output_backend(kl_out, as_numpy=as_numpy), None, None, None, None

        # Euclidean string path: keep moments unchanged except dtype casting.
        if has_cross:
            return None, mu_i.float(), sigma_i.float(), mu_j.float(), sigma_j.float()
        return None, mu_i.float(), sigma_i.float(), None, None

    # Linear-map path.
    (w_tensor,) = cast_many_to_torch(w, device=settings.device)
    resolved_mu_i, resolved_sigma_i = project_gaussians(mu_i, w_tensor, covariances=sigma_i)
    if has_cross:
        resolved_mu_j, resolved_sigma_j = project_gaussians(mu_j, w_tensor, covariances=sigma_j)
        return None, resolved_mu_i, resolved_sigma_i, resolved_mu_j, resolved_sigma_j
    return None, resolved_mu_i, resolved_sigma_i, None, None


def pairwise_gaussian_distance(
    mu,
    sigma,
    w,
    diag_bures_approx=False,
    as_numpy=False,
    eps: float = 1e-4,
    squared_ground_cost: bool = True,
):
    """Compute the Gaussian EMD distance for all pairs from a list of distributions using w (with torch)."""
    mu, sigma = cast_many_to_torch(mu, sigma, device=settings.device)
    validate_mean_covariance_shapes(mu, sigma, means_name="mu", covariances_name="sigma")

    early_result, w_mu, w_sigma, _, _ = _resolve_gaussian_ground_metric_inputs(
        mu,
        sigma,
        w,
        as_numpy=as_numpy,
    )
    if early_result is not None:
        return early_result

    wasserstein_bures = _compute_bures_term(
        w_sigma,
        w_sigma,
        diag_bures_approx=diag_bures_approx,
        eps=eps,
        symmetric=True,
    )

    w_mu_flat = w_mu.reshape(w_mu.size(0), -1)
    if squared_ground_cost:
        # Compute ||mu_i - mu_j||^2 directly without intermediate sqrt.
        diff = w_mu_flat.unsqueeze(1) - w_mu_flat.unsqueeze(0)
        mean_dist = (diff**2).sum(-1)
    else:
        mean_dist = torch.cdist(w_mu_flat, w_mu_flat, p=2)

    return to_output_backend(mean_dist + wasserstein_bures, as_numpy=as_numpy, clamp_min=0.0)


# OPTIMIZE: add outer batch dimension B to cross_gaussian_distance (currently only handles n_i × n_j without a batch axis)
def cross_gaussian_distance(
    mu_i,
    sigma_i,
    mu_j,
    sigma_j,
    w,
    diag_bures_approx=False,
    as_numpy=False,
    eps: float = 1e-4,
    squared_ground_cost: bool = False,
):
    """Compute Gaussian ground distances between two (potentially different) Gaussian sets."""
    mu_i, sigma_i, mu_j, sigma_j = cast_many_to_torch(mu_i, sigma_i, mu_j, sigma_j, device=settings.device)
    validate_cross_mean_covariance_shapes(mu_i, sigma_i, mu_j, sigma_j)

    early_result, w_mu_i, w_sigma_i, w_mu_j, w_sigma_j = _resolve_gaussian_ground_metric_inputs(
        mu_i,
        sigma_i,
        w,
        as_numpy=as_numpy,
        mu_j=mu_j,
        sigma_j=sigma_j,
    )
    if early_result is not None:
        return early_result

    wasserstein_bures = _compute_bures_term(
        w_sigma_i,
        w_sigma_j,
        diag_bures_approx=diag_bures_approx,
        eps=eps,
        symmetric=False,
    )

    w_mu_i_flat = w_mu_i.reshape(w_mu_i.size(0), -1)
    w_mu_j_flat = w_mu_j.reshape(w_mu_j.size(0), -1)
    if squared_ground_cost:
        diff = w_mu_i_flat.unsqueeze(1) - w_mu_j_flat.unsqueeze(0)
        mean_dist = (diff**2).sum(-1)
    else:
        mean_dist = torch.cdist(w_mu_i_flat, w_mu_j_flat, p=2)

    return to_output_backend(mean_dist + wasserstein_bures, as_numpy=as_numpy, clamp_min=0.0)


def compute_pairwise_kl_gaussians(means, covs):
    """
    Computes the pairwise Symmetric KL Divergence matrix for N Gaussians.

    Args:
        means: (N, D) numpy array or tensor
        covs:  (N, D, D) numpy array or tensor

    Returns:
        dist_matrix: (N, N) symmetric matrix where M[i,j] = KL(P_i||P_j) + KL(P_j||P_i)
    """
    means, covs = cast_many_to_torch(means, covs, device=settings.device)

    # 2. Prepare Distributions for Broadcasting
    # We want to compare every 'i' with every 'j'.
    # Distribution A: Shape (N, 1, D) -> Represents rows
    # Distribution B: Shape (1, N, D) -> Represents columns

    mean_A = means.unsqueeze(1)  # (N, 1, D)
    cov_A = covs.unsqueeze(1)  # (N, 1, D, D)

    mean_B = means.unsqueeze(0)  # (1, N, D)
    cov_B = covs.unsqueeze(0)  # (1, N, D, D)

    # Create the distribution objects
    # PyTorch automatically broadcasts the batch dimensions (N,1) against (1,N)
    dist_A = MultivariateNormal(mean_A, cov_A)
    dist_B = MultivariateNormal(mean_B, cov_B)

    # 3. Compute KL Divergence (Forward)
    # Output shape will be (N, N) automatically
    kl_forward = kl_divergence(dist_A, dist_B)  # KL(A || B)

    # 4. Symmetrize (KL is not symmetric, but a metric must be)
    # Distance = KL(A||B) + KL(B||A)
    # Since kl_forward[i,j] = KL(i||j) and kl_forward[j,i] = KL(j||i),
    # we can just add the matrix to its transpose.
    symmetric_kl = kl_forward + kl_forward.t()

    return symmetric_kl


def compute_cross_kl_gaussians(means_i, covs_i, means_j, covs_j):
    """Compute symmetric KL divergence between every Gaussian in set i and set j."""
    means_i, covs_i, means_j, covs_j = cast_many_to_torch(
        means_i,
        covs_i,
        means_j,
        covs_j,
        device=settings.device,
    )

    dist_i = MultivariateNormal(means_i.unsqueeze(1), covs_i.unsqueeze(1))
    dist_j = MultivariateNormal(means_j.unsqueeze(0), covs_j.unsqueeze(0))

    kl_ij = kl_divergence(dist_i, dist_j)
    kl_ji = kl_divergence(dist_j, dist_i)
    return kl_ij + kl_ji
