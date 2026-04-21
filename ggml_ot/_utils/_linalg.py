"""Linear algebra primitives (torch).

Batched matrix operations used across distance and optimization modules.
"""

from __future__ import annotations

import torch


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """Multiply batched matrices with reduced memory overhead.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    mat_a : torch.Tensor
        Shape ``(n, k, 1, d)``.
    mat_b : torch.Tensor
        Shape ``(1, k, d, d)``.
    """
    res = torch.zeros(mat_a.shape, device=mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """Multiply batched vectors with reduced memory overhead."""
    if mat_a.shape[-2] != 1 or mat_b.shape[-1] != 1:
        raise ValueError(f"Expected mat_a.shape[-2]==1 and mat_b.shape[-1]==1, got {mat_a.shape} and {mat_b.shape}")
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


def _pinv_with_jitter(var: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pseudo-inverse with exactly one configured jitter level."""
    eps_val = float(eps)
    d = int(var.shape[-1])
    eye = torch.eye(d, device=var.device, dtype=var.dtype).view(1, 1, d, d)
    var_reg = torch.nan_to_num(var + eps_val * eye, nan=eps_val, posinf=1.0 / eps_val, neginf=eps_val)
    precision = torch.linalg.pinv(var_reg)
    if not torch.all(torch.isfinite(precision)):
        raise RuntimeError("Could not compute finite covariance precision with current eps.")
    return precision, var_reg


def matrix_sqrt(matrices: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable batched square root of SPD matrices."""
    eigvals, eigvecs = torch.linalg.eigh(matrices, "L")
    eigvals_sqrt = torch.sqrt(torch.clamp(eigvals, min=float(eps)))
    eigvals_diag = torch.diag_embed(eigvals_sqrt, dim1=-2, dim2=-1)
    return eigvecs @ eigvals_diag @ eigvecs.transpose(-1, -2)
