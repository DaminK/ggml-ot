"""Torch-native GMM backend primitives."""

from __future__ import annotations

import warnings
from math import pi

import numpy as np
import torch

from ggml_ot import settings
from ggml_ot._utils._batch import move_batch_to_device
from ggml_ot._utils._linalg import calculate_matmul, calculate_matmul_n_times, _pinv_with_jitter

from ._fit_core import GMMFitConfig, GMMResult
from ggml_ot._utils._weights import normalize_weight_vector
from ggml_ot._utils._covariance import apply_singularity_handling, _sanitize_full_covariances


class GaussianMixture(torch.nn.Module):
    """Gaussian mixture model with EM fitting and torch tensors."""

    def __init__(
        self,
        n_components,
        n_features,
        covariance_type="full",
        eps=1.0e-4,
        init_params="kmeans",
        mu_init=None,
        var_init=None,
    ):
        """Initialize a torch-native Gaussian mixture model.

        Parameters
        ----------
        n_components
            Number of mixture components.
        n_features
            Feature dimensionality.
        covariance_type
            Covariance parameterization (`"diag"` or `"full"`).
        eps
            Numerical floor added to covariance estimates.
        init_params
            Mean initialization strategy (`"kmeans"` or `"random"`).
        mu_init
            Optional initial means tensor.
        var_init
            Optional initial covariance tensor.
        """
        super().__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        if self.covariance_type not in ["full", "diag"]:
            raise ValueError(f"covariance_type must be 'full' or 'diag', got '{self.covariance_type}'")
        if self.init_params not in ["kmeans", "random"]:
            raise ValueError(f"init_params must be 'kmeans' or 'random', got '{self.init_params}'")

        self._init_params()

    def _init_params(self):
        """Initialize model parameters (`mu`, `var`, `pi`)."""
        if self.mu_init is not None:
            if self.mu_init.size() != (1, self.n_components, self.n_features):
                raise ValueError(
                    "Input mu_init does not have required tensor dimensions (1, %i, %i)"
                    % (self.n_components, self.n_features)
                )
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                if self.var_init.size() != (1, self.n_components, self.n_features):
                    raise ValueError(
                        "Input var_init does not have required tensor dimensions (1, %i, %i)"
                        % (self.n_components, self.n_features)
                    )
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                if self.var_init.size() != (1, self.n_components, self.n_features, self.n_features):
                    raise ValueError(
                        "Input var_init does not have required tensor dimensions (1, %i, %i, %i)"
                        % (self.n_components, self.n_features, self.n_features)
                    )
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features)
                    .reshape(1, 1, self.n_features, self.n_features)
                    .repeat(1, self.n_components, 1, 1),
                    requires_grad=False,
                )

        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(
            1.0 / self.n_components
        )
        self.params_fitted = False

    @staticmethod
    def _normalize_pi(pi: np.ndarray, *, n_components: int) -> np.ndarray:
        """Normalize a pi payload to shape ``(K,)``."""
        pi_np = np.asarray(pi, dtype=np.float64)
        if pi_np.size == 0:
            raise ValueError("pi payload must not be empty.")

        pi_np = np.squeeze(pi_np)
        if pi_np.ndim == 0:
            pi_np = pi_np.reshape(1)

        if pi_np.ndim == 1 and pi_np.shape[0] == n_components:
            return normalize_weight_vector(pi_np)
        if pi_np.ndim == 2 and pi_np.shape == (1, n_components):
            return normalize_weight_vector(pi_np[0])
        if pi_np.ndim == 2 and pi_np.shape == (n_components, 1):
            return normalize_weight_vector(pi_np[:, 0])
        if pi_np.ndim == 3 and pi_np.shape == (1, n_components, 1):
            return normalize_weight_vector(pi_np[0, :, 0])

        raise ValueError(f"Unsupported pi shape {np.asarray(pi).shape}; expected (K,), (1,K), (K,1), or (1,K,1).")

    @classmethod
    def from_dict(
        cls,
        payload: dict,
        *,
        device: str | torch.device | None = None,
        eps: float | None = None,
    ) -> "GaussianMixture":
        """Construct a fitted model from a persisted ``{'mu','var','pi'}`` payload."""
        if not isinstance(payload, dict):
            raise TypeError(f"payload must be a dict, got {type(payload).__name__}.")
        required = {"mu", "var", "pi"}
        missing = sorted(required.difference(payload.keys()))
        if missing:
            raise KeyError(f"Missing keys in GMM payload: {missing}")

        covariance_type = str(payload.get("covariance_type", "full"))
        eps_value = float(payload["eps"] if "eps" in payload else (1.0e-4 if eps is None else eps))

        mu_np = np.asarray(payload["mu"], dtype=np.float32)
        if mu_np.ndim == 3 and mu_np.shape[0] == 1:
            mu_np = mu_np[0]
        if mu_np.ndim != 2:
            raise ValueError(f"mu must have shape (K, d) or (1, K, d). Got {np.asarray(payload['mu']).shape}.")

        n_components, n_features = int(mu_np.shape[0]), int(mu_np.shape[1])
        if n_components < 1 or n_features < 1:
            raise ValueError("mu must define at least one component and one feature.")

        cov_np = np.asarray(payload["var"], dtype=np.float32)
        if covariance_type == "diag":
            if cov_np.ndim == 4 and cov_np.shape[0] == 1:
                cov_np = cov_np[0]
            if cov_np.ndim == 3 and cov_np.shape == (1, n_components, n_features):
                cov_np = cov_np[0]
            elif cov_np.ndim == 3 and cov_np.shape == (n_components, n_features, n_features):
                cov_np = np.diagonal(cov_np, axis1=-2, axis2=-1)
            if cov_np.ndim != 2 or cov_np.shape != (n_components, n_features):
                raise ValueError(
                    "var must match diag layout (K, d) or full-matrix layout (K, d, d) "
                    f"for covariance_type='diag'. Got {np.asarray(payload['var']).shape}."
                )
            var_t = torch.as_tensor(cov_np.reshape(1, n_components, n_features), dtype=torch.float32)
        elif covariance_type == "full":
            if cov_np.ndim == 4 and cov_np.shape[0] == 1:
                cov_np = cov_np[0]
            if cov_np.ndim != 3 or cov_np.shape != (n_components, n_features, n_features):
                raise ValueError(
                    "var must have shape (K, d, d) or (1, K, d, d) "
                    f"for covariance_type='full'. Got {np.asarray(payload['var']).shape}."
                )
            var_t = torch.as_tensor(cov_np.reshape(1, n_components, n_features, n_features), dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported covariance_type: {covariance_type!r}.")

        pi_vec = cls._normalize_pi(np.asarray(payload["pi"]), n_components=n_components).astype(np.float32, copy=False)
        mu_t = torch.as_tensor(mu_np.reshape(1, n_components, n_features), dtype=torch.float32)
        pi_t = torch.as_tensor(pi_vec.reshape(1, n_components, 1), dtype=torch.float32)

        model = cls(
            n_components=n_components,
            n_features=n_features,
            covariance_type=str(covariance_type),
            eps=eps_value,
            mu_init=mu_t,
            var_init=var_t,
        )
        if device is not None:
            model = model.to(torch.device(device))

        model.mu.data = mu_t.to(model.mu.device)
        model.var.data = var_t.to(model.var.device)
        model.pi.data = pi_t.to(model.pi.device)
        model.params_fitted = True
        return model

    def to_dict(self, *, squeeze_batch: bool = True) -> dict[str, np.ndarray | str]:
        """Serialize fitted parameters into a storage-friendly dictionary."""
        mu = np.asarray(self.mu.detach().cpu().numpy(), dtype=np.float64)
        var = np.asarray(self.var.detach().cpu().numpy(), dtype=np.float64)
        pi = np.asarray(self.pi.detach().cpu().numpy(), dtype=np.float64)

        if squeeze_batch:
            if mu.ndim == 3 and mu.shape[0] == 1:
                mu = mu[0]
            if var.ndim in (3, 4) and var.shape[0] == 1:
                var = var[0]
            pi = np.squeeze(pi)

        return {
            "mu": mu,
            "var": var,
            "pi": pi,
            "covariance_type": str(self.covariance_type),
        }

    def check_size(self, x):
        """Ensure inputs have explicit component axis."""
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        return x

    def bic(self, x):
        """Compute Bayesian information criterion.

        Parameters
        ----------
        x
            Input samples.

        Returns
        -------
        torch.Tensor
            BIC value.
        """
        x = self.check_size(x)
        n = x.shape[0]

        if self.covariance_type == "diag":
            cov_params = self.n_components * self.n_features
        else:
            cov_params = self.n_components * (self.n_features * (self.n_features + 1) // 2)
        free_params = (self.n_components * self.n_features) + cov_params + (self.n_components - 1)

        bic = -2.0 * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

        return bic

    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """Fit model parameters via EM.

        Parameters
        ----------
        x
            Training samples.
        delta
            Convergence threshold for log-likelihood improvement.
        n_iter
            Maximum number of EM iterations.
        warm_start
            If True, continue from current parameters.
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):
            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            try:
                self.__em(x)
                self.log_likelihood = self.__score(x)
            except RuntimeError:
                device = self.mu.device
                self.__init__(
                    self.n_components,
                    self.n_features,
                    covariance_type=self.covariance_type,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps,
                )
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data = self.get_kmeans_mu(x, n_centers=self.n_components)
                i += 1
                continue

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                self.__init__(
                    self.n_components,
                    self.n_features,
                    covariance_type=self.covariance_type,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps,
                )
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True

    def predict(self, x, probs=False):
        """Predict hard component assignments or probabilities.

        Parameters
        ----------
        x
            Input samples.
        probs
            If True, return posterior probabilities.

        Returns
        -------
        torch.Tensor
            Component ids or posterior matrix.
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))

    def predict_proba(self, x):
        """Predict posterior component probabilities.

        Parameters
        ----------
        x
            Input samples.

        Returns
        -------
        torch.Tensor
            Posterior probabilities.
        """
        return self.predict(x, probs=True)

    def predict_responsibilities_numpy(self, x: np.ndarray, *, device: str | torch.device | None = None) -> np.ndarray:
        """Return posterior responsibilities for numpy inputs."""
        target_device = self.mu.device if device is None else torch.device(device)
        x_t = move_batch_to_device(np.asarray(x), device=target_device, dtype=torch.float32)
        responsibilities = self.predict_proba(x_t).detach().cpu().numpy()
        if responsibilities.ndim == 1:
            responsibilities = responsibilities.reshape(-1, 1)
        return np.asarray(responsibilities, dtype=np.float64)

    def predict_hard_components_numpy(self, x: np.ndarray, *, device: str | torch.device | None = None) -> np.ndarray:
        """Return MAP component assignments for numpy inputs."""
        if int(x.shape[0]) == 0:
            return np.zeros(0, dtype=int)
        responsibilities = self.predict_responsibilities_numpy(x, device=device)
        return np.argmax(responsibilities, axis=1).astype(int)

    def sample(self, n):
        """Sample observations from the fitted mixture.

        Parameters
        ----------
        n
            Number of samples to draw.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Sample matrix and sampled component labels.
        """
        probs = self.pi.reshape(-1)
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=probs).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        active_components = torch.nonzero(counts > 0, as_tuple=False).reshape(-1).tolist()
        for k in active_components:
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(
                    self.var[0, k]
                )
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x, y

    def score_samples(self, x):
        """Return per-sample log-likelihood scores.

        Parameters
        ----------
        x
            Input samples.

        Returns
        -------
        torch.Tensor
            Log-likelihood per sample.
        """
        x = self.check_size(x)
        score = self.__score(x, as_average=False)
        return score

    def _estimate_log_prob(self, x):
        """Estimate per-component log-density values.

        Parameters
        ----------
        x
            Input samples.

        Returns
        -------
        torch.Tensor
            Log-density tensor per component.
        """
        x = self.check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var

            d = x.shape[-1]
            var_sanitized = _sanitize_full_covariances(var, self.eps)
            # Use pseudo-inverse for robustness in near-singular covariance regimes.
            precision, var_reg = _pinv_with_jitter(var_sanitized, self.eps)

            log_2pi = d * np.log(2.0 * pi)
            sign, log_det_cov = torch.linalg.slogdet(var_reg)
            log_det_cov = torch.nan_to_num(log_det_cov, nan=np.log(self.eps), neginf=np.log(self.eps))
            invalid = (sign <= 0) | ~torch.isfinite(sign)
            if torch.any(invalid):
                log_det_cov = torch.where(
                    invalid,
                    torch.log(torch.clamp(torch.abs(torch.linalg.det(var_reg)), min=self.eps)),
                    log_det_cov,
                )

            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)

            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -0.5 * (log_2pi + log_det_cov.unsqueeze(-1) + x_mu_T_precision_x_mu)

        if self.covariance_type == "diag":
            mu = self.mu
            var = torch.clamp(self.var, min=self.eps)
            inv_var = torch.reciprocal(var)

            quad = torch.sum((x - mu) * (x - mu) * inv_var, dim=2, keepdim=True)
            log_det = torch.sum(torch.log(var), dim=2, keepdim=True)

            return -0.5 * (self.n_features * np.log(2.0 * pi) + log_det + quad)

        raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")

    def _e_step(self, x):
        """Run the expectation step.

        Parameters
        ----------
        x
            Input samples.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean log normalizer and component log-responsibilities.
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        """Run the maximization step.

        Parameters
        ----------
        x
            Input samples.
        log_resp
            Log-responsibilities from the E-step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated mixture weights, means, and covariances.
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)
        resp = torch.nan_to_num(resp, nan=0.0, posinf=1.0, neginf=0.0)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = (
                torch.sum(
                    (x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1),
                    dim=0,
                    keepdim=True,
                )
                / pi.unsqueeze(-1)
                + eps
            )
            var = _sanitize_full_covariances(var, self.eps)

        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps
            var = torch.nan_to_num(var, nan=self.eps, posinf=1.0 / self.eps, neginf=self.eps)
            var = torch.clamp(var, min=self.eps)

        pi = pi / x.shape[0]
        pi = torch.nan_to_num(pi, nan=1.0 / self.n_components, posinf=1.0, neginf=self.eps)
        pi = torch.clamp(pi, min=self.eps)
        pi = pi / torch.sum(pi, dim=1, keepdim=True)

        return pi, mu, var

    def __em(self, x):
        """Perform one EM update."""
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):
        """Compute log-likelihood scores.

        Parameters
        ----------
        x
            Input samples.
        as_average
            If True, return mean score; else per-sample scores.

        Returns
        -------
        torch.Tensor
            Log-likelihood score(s).
        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        return torch.squeeze(per_sample_score)

    def __update_mu(self, mu):
        """Update mean parameter tensor."""
        if mu.size() not in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)]:
            raise ValueError(
                "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
                % (self.n_components, self.n_features, self.n_components, self.n_features)
            )

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):
        """Update covariance parameter tensor."""
        if self.covariance_type == "full":
            if var.size() not in [
                (self.n_components, self.n_features, self.n_features),
                (1, self.n_components, self.n_features, self.n_features),
            ]:
                raise ValueError(
                    "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)"
                    % (
                        self.n_components,
                        self.n_features,
                        self.n_features,
                        self.n_components,
                        self.n_features,
                        self.n_features,
                    )
                )

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            if var.size() not in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)]:
                raise ValueError(
                    "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
                    % (self.n_components, self.n_features, self.n_components, self.n_features)
                )

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):
        """Update mixture weight parameter tensor."""
        if pi.size() not in [(1, self.n_components, 1)]:
            raise ValueError(
                "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)
            )

        self.pi.data = pi

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """Initialize means via repeated k-means restarts.

        Parameters
        ----------
        x
            Input samples.
        n_centers
            Number of clusters/components.
        init_times
            Number of random restarts for k-means initialization.
        min_delta
            Convergence threshold for center updates.

        Returns
        -------
        torch.Tensor
            Initialized means with shape `(1, n_centers, n_features)`.
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)

        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)
        min_cost = np.inf

        center = None
        for _ in range(init_times):
            tmp_center = x[settings.numpy_generator.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = torch.zeros((), device=x.device)
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        if center is None:
            warnings.warn("Fallback in case no center was found; this can happen when many samples are identical.")
            center = x[settings.numpy_generator.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return center.unsqueeze(0) * (x_max - x_min) + x_min

    @classmethod
    def fit_from_numpy(cls, *, x: np.ndarray, config: GMMFitConfig) -> GMMResult:
        """Fit one global GMM from a numpy matrix using EM with multiple restarts."""
        if config.n_components < 1:
            raise ValueError("n_components must be >= 1.")

        device = settings.device
        x_t = move_batch_to_device(np.asarray(x), device=device, dtype=torch.float32)
        if not bool(torch.all(torch.isfinite(x_t))):
            bad = int((~torch.isfinite(x_t)).sum().item())
            raise ValueError(
                f"Input matrix contains {bad} non-finite values (NaN/Inf). "
                "Clean or impute non-finite entries before GMM fit."
            )
        n_features = int(x_t.shape[1])
        eps_value = float(config.eps)

        n_init = max(1, int(config.n_init))
        best_result: GMMResult | None = None
        fit_errors: list[str] = []

        for i in range(n_init):
            try:
                model = cls(
                    n_components=config.n_components,
                    n_features=n_features,
                    covariance_type=config.covariance_type,
                    eps=eps_value,
                ).to(device)
                model.fit(x_t, delta=config.tol, n_iter=config.max_iter)

                responsibilities = model.predict_responsibilities_numpy(x)
                bic = float(model.bic(x_t).detach().cpu().item())
                if not np.isfinite(bic):
                    fit_errors.append(f"n_init_idx={i}, eps={eps_value}: non-finite BIC={bic}")
                    continue

                candidate = GMMResult(
                    mu=np.asarray(model.mu.detach().cpu().numpy(), dtype=np.float64),
                    var=np.asarray(model.var.detach().cpu().numpy(), dtype=np.float64),
                    pi=np.asarray(model.pi.detach().cpu().numpy(), dtype=np.float64).reshape(1, -1),
                    responsibilities=responsibilities,
                    bic=bic,
                    model=model,
                )
                # Keep best restart by BIC to make EM randomness less brittle across runs.
                if best_result is None or float(candidate.bic) < float(best_result.bic):
                    best_result = candidate
            except Exception as exc:  # noqa: BLE001
                fit_errors.append(f"n_init_idx={i}, eps={eps_value}: {exc}")
                continue

        if best_result is None:
            raise ValueError(
                "native fit failed for all attempts. "
                f"Last error: {fit_errors[-1] if fit_errors else 'unknown'}. "
                f"Try increasing eps manually (current eps={eps_value})."
            )

        # Apply singularity handling to the best result's covariances.
        # "guarded" (default): clamp + raise if still singular after clamping.
        # "robust": clamp + warn if still singular after clamping (permissive).
        # "strict": raise immediately if any eigenvalue is below eps (no clamping).
        sanitized_var = apply_singularity_handling(
            best_result.var,
            covariance_type=config.covariance_type,
            eps=eps_value,
            singularity_handling=config.singularity_handling,
            n_components=config.n_components,
        )
        if sanitized_var is not best_result.var:
            best_result = GMMResult(
                mu=best_result.mu,
                var=sanitized_var,
                pi=best_result.pi,
                responsibilities=best_result.responsibilities,
                bic=best_result.bic,
                model=best_result.model,
            )
        return best_result

    @classmethod
    def fit_many_from_numpy(
        cls,
        *,
        x_by_distribution: dict[str, np.ndarray],
        config: GMMFitConfig,
        verbose: bool = False,
    ) -> dict[str, GMMResult]:
        """Fit one GMM per distribution from numpy matrices."""
        from tqdm.auto import tqdm

        results: dict[str, GMMResult] = {}
        dist_iter = (
            tqdm(x_by_distribution.items(), desc="[fit_gmm] fitting distributions", leave=False)
            if verbose
            else x_by_distribution.items()
        )
        for distribution_id, x in dist_iter:
            results[distribution_id] = cls.fit_from_numpy(x=x, config=config)
        return results


__all__ = [
    "GaussianMixture",
]
