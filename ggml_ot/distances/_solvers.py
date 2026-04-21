import numpy as np
import torch
import ot

from ggml_ot._utils._weights import is_empty_weights, normalize_weight_vector
from ggml_ot._utils._validate import assert_finite_cost_matrix, assert_finite_tensor


def _emd2(
    a,
    b,
    M,
    return_pi: bool = False,
    n_threads: int = 32,
    reg: float | None = None,
    numItermax: int | None = None,
    stopThr: float | None = None,
):
    # Allow omitting weights (uniform weights will be used).
    if is_empty_weights(a):
        a = ot.unif(M.shape[0], type_as=M)
    if is_empty_weights(b):
        b = ot.unif(M.shape[1], type_as=M)

    # Guard against tiny mass drift (e.g. float32 roundoff) before POT's strict checks.
    # compute_OT already normalizes weights once per call; this remains a defensive
    # fallback for direct/internal _emd2 callers.
    a = normalize_weight_vector(a)
    b = normalize_weight_vector(b)
    assert_finite_cost_matrix(M)

    if reg is not None and float(reg) > 0:
        if isinstance(M, torch.Tensor):
            a = a.to(device=M.device, dtype=M.dtype)
            b = b.to(device=M.device, dtype=M.dtype)
        else:
            a = np.asarray(a, dtype=M.dtype)
            b = np.asarray(b, dtype=M.dtype)

        sinkhorn_kwargs = {
            "numItermax": int(numItermax) if numItermax is not None else 100,
            # Match the training Sinkhorn path where ``None`` means "run the
            # configured iteration budget without early stopping".
            "stopThr": 0.0 if stopThr is None else float(stopThr),
            "warn": False,
        }
        M_scale = max(float(M.mean()), 1e-12)
        M_scaled = M / M_scale

        if return_pi:
            Pi = ot.sinkhorn(a, b, M_scaled, float(reg), **sinkhorn_kwargs)
            return (Pi * M).sum(), Pi
        return ot.sinkhorn2(a, b, M_scaled, float(reg), **sinkhorn_kwargs) * M_scale

    if return_pi:
        # We disable POT marginal checks because marginals are explicitly normalized above.
        W, log = ot.emd2(a, b, M, log=True, return_matrix=True, numThreads=n_threads, check_marginals=False)
        Pi = log["G"]
        return W, Pi
    else:
        return ot.emd2(a, b, M, numThreads=n_threads, check_marginals=False)


def _sinkhorn2_batched_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    *,
    reg: float,
    numItermax: int = 100,
    stopThr: float | None = 1e-9,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Batched Sinkhorn cost for CUDA acceleration.

    This is a torch-only implementation that supports gradients w.r.t. M.
    Shapes:
      a: (B, n)
      b: (B, m)
      M: (B, n, m)
    Returns:
      cost: (B,)
    """
    # Allow omitting weights (uniform weights will be used).
    if a.shape[-1] == 0:
        a = torch.full((a.shape[0], M.shape[1]), fill_value=1 / float(M.shape[1]), device=a.device, dtype=M.dtype)
    if b.shape[-1] == 0:
        b = torch.full((b.shape[0], M.shape[2]), fill_value=1 / float(M.shape[2]), device=b.device, dtype=M.dtype)

    # FUTURE: handle condensed distance matrix in batched sinkhorn instead to reduce VRAM usage!
    if a.ndim != 2 or b.ndim != 2 or M.ndim != 3:
        raise ValueError("Expected a,b as (B,n)/(B,m) and M as (B,n,m)")
    if M.shape[0] != a.shape[0] or M.shape[0] != b.shape[0]:
        raise ValueError("Batch dimension mismatch between a, b, and M")
    if M.shape[1] != a.shape[1] or M.shape[2] != b.shape[1]:
        raise ValueError("Support-size mismatch between a/b and M")
    assert_finite_tensor(M.detach(), name="sinkhorn M (input)")

    # Normalize weights (robust to slight numerical drift).
    a = a.clamp_min(0)
    b = b.clamp_min(0)
    a = a / (a.sum(dim=1, keepdim=True) + eps)
    b = b / (b.sum(dim=1, keepdim=True) + eps)
    assert_finite_tensor(a.detach(), name="sinkhorn a (normalized)")
    assert_finite_tensor(b.detach(), name="sinkhorn b (normalized)")

    # Normalise M by its mean so that `reg` is a dimensionless fraction of the
    # mean cost (e.g. reg=0.5 ≈ half the mean cost).  The scale factor is
    # detached so it is treated as a constant during backprop: without detach,
    # differentiating through mean(M) adds a uniform shift (1/(n·m))·total_cost
    # to every gradient entry, corrupting the OT-plan gradient.  With detach,
    # ∂cost/∂M_ij = plan_ij exactly.  The factor is re-applied to the output so
    # distances remain in the original cost units.
    M_scale = M.detach().mean().clamp(min=eps)
    M = M / M_scale

    # Clamp min to prevent underflow (K becoming 0 which kills gradients)
    K = torch.exp((-M / float(reg)).clamp(min=-100.0, max=50.0))
    assert_finite_tensor(K.detach(), name="sinkhorn K (kernel)")

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(int(numItermax)):
        u_prev = u
        Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
        u = a / (Kv + eps)
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
        v = b / (KTu + eps)
        if stopThr is not None:
            # Stop on max absolute scaling update, matching standard Sinkhorn fixed-point checks.
            if (u - u_prev).abs().amax() < float(stopThr):
                break

    # FUTURE: handle unsqueezing blowing up memory
    # try cost = torch.einsum('bi,bij,bj,bij->b', u, K, v, M)
    cost = (u.unsqueeze(2) * K * v.unsqueeze(1) * M).sum(dim=(1, 2))
    cost = cost * M_scale  # restore original cost units
    assert_finite_tensor(cost.detach(), name="sinkhorn cost (output)")
    return cost
