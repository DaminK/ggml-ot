"""Synthetic GMM dataset generation for GGML evaluation.

The generator exposes one unified synthetic family:

- ``signal_mean_shift = 0`` gives a covariance-only benchmark.
- ``signal_mean_shift > 0`` adds mean separation in the signal subspace.

Signal structure lives in a latent 2-D plane. The remaining dimensions carry
structured nuisance variation. A random orthogonal mixing matrix hides that
decomposition from the learner.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ortho_group

from ggml_ot.data.generic import TripletDataset


def synth_gmm(
    *,
    n_dim: int = 10,
    n_patients: int = 6,
    n_samples: int = 250,
    signal_mass_ratio: float = 0.2,
    n_modes: int = 10,
    signal_means_offset: float = 12.0,
    signal_means_jitter: float = 0.75,
    noise_means_offset: float = 3.0,
    noise_means_jitter: float = 0.75,
    noise_subspace_rank: int = 2,
    signal_weight_concentration: float | None = None,
    noise_weight_concentration: float | None = None,
    signal_mean_shift: float = 1.0,
    signal_cov_scale: float = 1.2,
    signal_anisotropy: float = 12.0,
    cov_rotation_jitter: float = 10.0,
    cov_scale_jitter: float = 0.15,
    global_rotation: float = 30.0,
    random_seed: int = 42,
) -> dict:
    """Generate a synthetic GMM dataset.

    Parameters
    ----------
    n_dim
        Total feature dimensionality. The signal lives in a 2-D latent plane.
    n_patients
        Total number of patient distributions. The dataset is binary, so this
        must be even and is split equally between the two classes.
    n_samples
        Number of sampled cells per patient.
    signal_mass_ratio
        Fraction of each patient's cells assigned to the two signal modes.
        When ``n_modes == 2`` there are no noise modes and all mass is placed
        on the signal modes regardless of this value.
    n_modes
        Total number of mixture components per patient. Two are always signal
        modes, so ``n_modes == 2`` yields a noise-free debugging dataset.
    signal_means_offset
        Fixed radius of the signal-mode archetypes in the 2-D signal plane.
    signal_means_jitter
        Per-patient standard deviation of signal-mode mean jitter. It is
        applied axis-aligned in the 2-D signal plane and independently across
        the nuisance subspace around zero.
    noise_means_offset
        Fixed radius of the noise-mode archetypes in the nuisance subspace.
        Noise modes have zero fixed offset in the 2-D signal plane.
    noise_means_jitter
        Per-patient standard deviation of noise-mode mean jitter. It is
        applied isotropically in the 2-D signal plane around zero and
        independently across the nuisance subspace around the noise archetype.
    noise_subspace_rank
        Rank of the low-rank spiked covariance archetype used for noise modes
        in the nuisance subspace. It is clipped to the available nuisance
        dimensionality.
    signal_weight_concentration
        Optional symmetric Dirichlet concentration for distributing the total
        signal mass across the two active signal modes. ``None`` keeps the
        current deterministic near-equal split.
    noise_weight_concentration
        Optional symmetric Dirichlet concentration for distributing the total
        noise mass across the noise modes. Smaller positive values create more
        uneven noise weights. ``None`` keeps the current deterministic split.
    signal_mean_shift
        Class-specific signal-mode offset in the signal plane. Setting this to
        ``0`` gives a covariance-only benchmark.
    signal_cov_scale
        Overall variance scale of the signal-mode covariances.
    signal_anisotropy
        Ratio between the major and minor signal variances. ``1`` is isotropic.
    cov_rotation_jitter
        Per-patient covariance rotation jitter. In the 2-D signal plane this is
        a standard planar rotation in degrees. In the nuisance subspace it is
        applied as a product of random Givens rotations with angles sampled in
        the same degree range.
    cov_scale_jitter
        Per-patient multiplicative covariance-scale jitter around 1.
    global_rotation
        Shared rotation of the latent 2-D plane before mixing into the
        high-dimensional observation space.
    random_seed
        Seed for reproducibility.

    Returns
    -------
    dict
        Synthetic dataset payload with sampled cells, aggregate latent GMM
        parameters, mixing metadata, and analytical per-patient GMMs in the
        mixed observation space.
    """
    _validate_synth_inputs(
        n_dim=n_dim,
        n_patients=n_patients,
        n_samples=n_samples,
        signal_mass_ratio=signal_mass_ratio,
        n_modes=n_modes,
        signal_means_offset=signal_means_offset,
        signal_means_jitter=signal_means_jitter,
        noise_means_offset=noise_means_offset,
        noise_means_jitter=noise_means_jitter,
        noise_subspace_rank=noise_subspace_rank,
        signal_weight_concentration=signal_weight_concentration,
        noise_weight_concentration=noise_weight_concentration,
        signal_cov_scale=signal_cov_scale,
        signal_anisotropy=signal_anisotropy,
        cov_rotation_jitter=cov_rotation_jitter,
        cov_scale_jitter=cov_scale_jitter,
    )

    rng = np.random.RandomState(random_seed)
    patients_per_class = n_patients // 2
    n_noise = n_modes - 2
    n_noise_subspace_dims = n_dim - 2

    signal_plane_offset = float(signal_means_offset)
    noise_subspace_offset = float(noise_means_offset)
    signal_dim_jitter = float(signal_means_jitter)
    signal_subspace_jitter = float(signal_means_jitter)
    noise_plane_jitter = float(noise_means_jitter)
    noise_subspace_jitter = float(noise_means_jitter)

    signal_means = _build_signal_means(signal_plane_offset, signal_mean_shift)
    signal_covs = _build_signal_covariances(
        signal_cov_scale=signal_cov_scale,
        signal_anisotropy=signal_anisotropy,
        rng=rng,
    )

    noise_means = np.zeros((n_noise, 2))
    if n_noise == 0:
        noise_covs = np.zeros((0, 2, 2))
    else:
        noise_covs = np.array(
            [
                _random_2d_cov(
                    rng,
                    cov_scale=signal_cov_scale * 0.55,
                    anisotropy_low=1.0,
                    anisotropy_high=max(1.5, signal_anisotropy / 2.0),
                )
                for _ in range(n_noise)
            ]
        )

    signal_process_subspace_covs = _build_subspace_covariances(
        n_modes=2,
        n_dims=n_noise_subspace_dims,
        rng=rng,
        cov_scale=max(0.05, signal_cov_scale * 0.08),
    )
    signal_noise_subspace_covs = signal_process_subspace_covs[[0, 0, 1, 1]]

    noise_subspace_means = _build_subspace_archetypes(
        n_modes=n_noise,
        n_dims=n_noise_subspace_dims,
        rng=rng,
        radius_low=noise_subspace_offset * 0.9,
        radius_high=noise_subspace_offset * 1.1,
    )
    noise_subspace_covs = _build_spiked_subspace_covariances(
        n_modes=n_noise,
        n_dims=n_noise_subspace_dims,
        rng=rng,
        cov_scale=max(0.1, signal_cov_scale * 0.45),
        rank=noise_subspace_rank,
    )

    Q_high = ortho_group.rvs(dim=n_dim, random_state=random_seed + 1)
    R_global = _rotation_matrix(global_rotation)

    agg_means_2d = np.vstack([signal_means, noise_means])
    agg_covs_2d = np.concatenate([signal_covs, noise_covs])
    agg_means_rot = agg_means_2d @ R_global.T
    agg_covs_rot = np.array([_project_psd(R_global @ cov @ R_global.T) for cov in agg_covs_2d])

    all_samples = {
        "high": [],
        "rot": [],
        "clean": [],
        "labels": [],
        "mode_names": [],
        "patient_ids": [],
        "signal_noise": [],
    }
    patient_gmms = []

    patient_id_counter = 0
    for label in [0, 1]:
        for _ in range(patients_per_class):
            patient_id = patient_id_counter
            patient_id_counter += 1

            signal_indices = [0, 2] if label == 0 else [1, 3]
            signal_counts, noise_counts = _split_mode_counts(
                n_samples=n_samples,
                signal_mass_ratio=signal_mass_ratio,
                n_noise=n_noise,
                rng=rng,
                signal_weight_concentration=signal_weight_concentration,
                noise_weight_concentration=noise_weight_concentration,
            )

            gm_means = []
            gm_covs = []
            gm_ns = []

            for local_idx, signal_idx in enumerate(signal_indices):
                signal_mean_clean = signal_means[signal_idx] + _signal_axis_jitter(
                    signal_idx=signal_idx,
                    rng=rng,
                    scale=signal_dim_jitter,
                )
                signal_cov_clean = _jitter_covariance(
                    signal_covs[signal_idx],
                    rng=rng,
                    rotation_jitter=cov_rotation_jitter,
                    scale_jitter=cov_scale_jitter,
                )
                signal_mean_rot = signal_mean_clean @ R_global.T
                signal_cov_rot = _project_psd(R_global @ signal_cov_clean @ R_global.T)
                signal_subspace_mean = _jitter_subspace_mean(
                    np.zeros(n_noise_subspace_dims),
                    rng=rng,
                    scale=signal_subspace_jitter,
                )
                signal_subspace_cov = _jitter_covariance(
                    signal_noise_subspace_covs[signal_idx],
                    rng=rng,
                    rotation_jitter=cov_rotation_jitter,
                    scale_jitter=cov_scale_jitter,
                )

                n_sig = signal_counts[local_idx]
                X_clean = rng.multivariate_normal(signal_mean_clean, signal_cov_clean, n_sig)
                X_rot = X_clean @ R_global.T
                X_high = _embed_to_high_dim(
                    X_rot,
                    subspace_mean=signal_subspace_mean,
                    subspace_cov=signal_subspace_cov,
                    Q_high=Q_high,
                    rng=rng,
                )

                mode_name = "A" if signal_idx in [0, 1] else "B"
                all_samples["clean"].append(X_clean)
                all_samples["rot"].append(X_rot)
                all_samples["high"].append(X_high)
                all_samples["labels"].extend([label] * n_sig)
                all_samples["mode_names"].extend([mode_name] * n_sig)
                all_samples["patient_ids"].extend([patient_id] * n_sig)
                all_samples["signal_noise"].extend(["signal"] * n_sig)

                gm_means.append(_high_dim_mean(signal_mean_rot, signal_subspace_mean, Q_high))
                gm_covs.append(_high_dim_cov(signal_cov_rot, signal_subspace_cov, Q_high))
                gm_ns.append(n_sig)

            for noise_idx in range(n_noise):
                noise_mean_clean = noise_means[noise_idx] + rng.normal(scale=noise_plane_jitter, size=2)
                noise_cov_clean = _jitter_covariance(
                    noise_covs[noise_idx],
                    rng=rng,
                    rotation_jitter=cov_rotation_jitter,
                    scale_jitter=cov_scale_jitter,
                )
                noise_mean_rot = noise_mean_clean @ R_global.T
                noise_cov_rot = _project_psd(R_global @ noise_cov_clean @ R_global.T)

                noise_subspace_mean = _jitter_subspace_mean(
                    noise_subspace_means[noise_idx],
                    rng=rng,
                    scale=noise_subspace_jitter,
                )
                noise_subspace_cov = _jitter_covariance(
                    noise_subspace_covs[noise_idx],
                    rng=rng,
                    rotation_jitter=cov_rotation_jitter,
                    scale_jitter=cov_scale_jitter,
                )

                n_noi = noise_counts[noise_idx]
                X_clean = rng.multivariate_normal(noise_mean_clean, noise_cov_clean, n_noi)
                X_rot = X_clean @ R_global.T
                X_high = _embed_to_high_dim(
                    X_rot,
                    subspace_mean=noise_subspace_mean,
                    subspace_cov=noise_subspace_cov,
                    Q_high=Q_high,
                    rng=rng,
                )

                all_samples["clean"].append(X_clean)
                all_samples["rot"].append(X_rot)
                all_samples["high"].append(X_high)
                all_samples["labels"].extend([label] * n_noi)
                all_samples["mode_names"].extend(["Noise"] * n_noi)
                all_samples["patient_ids"].extend([patient_id] * n_noi)
                all_samples["signal_noise"].extend(["noise"] * n_noi)

                gm_means.append(_high_dim_mean(noise_mean_rot, noise_subspace_mean, Q_high))
                gm_covs.append(_high_dim_cov(noise_cov_rot, noise_subspace_cov, Q_high))
                gm_ns.append(n_noi)

            gm_weights = np.asarray(gm_ns, dtype=np.float64)
            gm_weights /= gm_weights.sum()
            patient_gmms.append(
                {
                    "means": np.asarray(gm_means),
                    "covs": np.asarray(gm_covs),
                    "weights": gm_weights,
                    "label": label,
                    "patient_id": patient_id,
                }
            )

    return {
        "agg_means_2d": agg_means_2d,
        "agg_covs_2d": agg_covs_2d,
        "agg_means_rot": agg_means_rot,
        "agg_covs_rot": agg_covs_rot,
        "samples": all_samples,
        "X_high_all": np.vstack(all_samples["high"]),
        "Q_mixing": Q_high,
        "R_rotation": R_global,
        "n_noise": n_noise,
        "n_modes": n_modes,
        "patient_gmms": patient_gmms,
        "generator_params": {
            "n_dim": n_dim,
            "n_patients": n_patients,
            "n_samples": n_samples,
            "signal_mass_ratio": signal_mass_ratio,
            "n_modes": n_modes,
            "signal_means_offset": signal_means_offset,
            "signal_means_jitter": signal_means_jitter,
            "noise_means_offset": noise_means_offset,
            "noise_means_jitter": noise_means_jitter,
            "noise_subspace_rank": noise_subspace_rank,
            "signal_weight_concentration": signal_weight_concentration,
            "noise_weight_concentration": noise_weight_concentration,
            "signal_mean_shift": signal_mean_shift,
            "signal_cov_scale": signal_cov_scale,
            "signal_anisotropy": signal_anisotropy,
            "cov_rotation_jitter": cov_rotation_jitter,
            "cov_scale_jitter": cov_scale_jitter,
            "global_rotation": global_rotation,
            "random_seed": random_seed,
        },
    }


def _build_anndata_from_synth(data: dict, n_dim: int):
    """Build an AnnData object from a :func:`synth_gmm` payload."""
    import anndata
    import scanpy as sc

    samples = data["samples"]
    X = data["X_high_all"]

    obs = pd.DataFrame(
        {
            "sample": [f"patient_{pid}" for pid in samples["patient_ids"]],
            "patient_group": [f"class_{label}" for label in samples["labels"]],
            "label": samples["labels"],
            "mode_name": samples["mode_names"],
            "signal_noise": samples["signal_noise"],
        }
    )
    obs.index = obs.index.astype(str)

    adata = anndata.AnnData(X=X.astype(np.float32), obs=obs)
    adata.var_names = [f"gene_{idx}" for idx in range(X.shape[1])]
    adata.obsm["X_clean"] = np.vstack(samples["clean"]).astype(np.float32)
    adata.obsm["X_rotated"] = np.vstack(samples["rot"]).astype(np.float32)

    max_comps = min(X.shape[0], X.shape[1]) - 1
    n_comps = min(max_comps, 50)
    if n_comps > 0:
        sc.pp.pca(adata, n_comps=n_comps)
    else:
        adata.obsm["X_pca"] = X.astype(np.float32)

    adata.uns["Q_mixing"] = data["Q_mixing"]
    adata.uns["R_rotation"] = data["R_rotation"]
    adata.uns["agg_means_2d"] = data["agg_means_2d"]
    adata.uns["agg_covs_2d"] = data["agg_covs_2d"]
    adata.uns["agg_means_rot"] = data["agg_means_rot"]
    adata.uns["agg_covs_rot"] = data["agg_covs_rot"]
    adata.uns["n_noise"] = data["n_noise"]
    adata.uns["n_modes"] = data["n_modes"]
    adata.uns["generator_params"] = data["generator_params"]
    return adata


def _ordered_patient_gmms_for_adata(data: dict, adata) -> list[dict]:
    """Return analytical patient GMMs in the AnnData sample order."""
    patient_gmms = data["patient_gmms"]
    sorted_names = sorted(adata.obs["sample"].unique())
    gmm_by_name = {f"patient_{entry['patient_id']}": entry for entry in patient_gmms}
    return [gmm_by_name[name] for name in sorted_names]


def _persist_ground_truth_gmm_schema(adata, data: dict, *, gmm_key: str) -> None:
    """Persist the analytical synthetic GMM under ``adata.uns[gmm_key]``."""
    from ggml_ot.gmm._anndata_dataset_interface import _store_gmm_in_adata

    patient_gmms = _ordered_patient_gmms_for_adata(data, adata)
    supports_out = np.stack([entry["means"] for entry in patient_gmms])
    covs_out = np.stack([entry["covs"] for entry in patient_gmms])
    weights_out = np.stack([entry["weights"] for entry in patient_gmms])
    distribution_ids = np.asarray([f"patient_{entry['patient_id']}" for entry in patient_gmms])
    n_components = int(supports_out.shape[1])

    _store_gmm_in_adata(
        adata=adata,
        gmm_key=gmm_key,
        component_sharing="sample_specific",
        use_rep=None,
        covariance_type="full",
        n_components=n_components,
        distribution_n_components=np.full(len(patient_gmms), n_components, dtype=int),
        fit_params={
            "max_iter": 0,
            "tol": 0.0,
            "n_init": 0,
            "eps": 1.0e-4,
        },
        selection=None,
        backend="synthetic_ground_truth",
        backend_metadata={
            "fit_indices_count": int(adata.n_obs),
            "refit": "none",
            "gmm_provenance": "synthetic_ground_truth",
        },
        model_payload={
            "mu": supports_out,
            "var": covs_out,
            "pi": weights_out,
        },
        distribution_weights=weights_out,
        distribution_ids=distribution_ids,
        weights_fit_scope="generator",
        weights_source="stored",
    )


def synth_gmm_anndata(
    *,
    gmm_key: str | None = None,
    n_dim: int = 10,
    n_patients: int = 6,
    n_samples: int = 250,
    signal_mass_ratio: float = 0.2,
    n_modes: int = 10,
    signal_means_offset: float = 12.0,
    signal_means_jitter: float = 0.75,
    noise_means_offset: float = 3.0,
    noise_means_jitter: float = 0.75,
    noise_subspace_rank: int = 2,
    signal_weight_concentration: float | None = None,
    noise_weight_concentration: float | None = None,
    signal_mean_shift: float = 1.0,
    signal_cov_scale: float = 1.2,
    signal_anisotropy: float = 12.0,
    cov_rotation_jitter: float = 10.0,
    cov_scale_jitter: float = 0.15,
    global_rotation: float = 30.0,
    random_seed: int = 42,
):
    """Generate a synthetic GMM dataset wrapped in an AnnData object.

    Parameters
    ----------
    gmm_key
        Optional key used to persist the analytical raw-space ground-truth GMM
        under ``adata.uns[gmm_key]`` in the standard GGML schema.
    """
    data = synth_gmm(
        n_dim=n_dim,
        n_patients=n_patients,
        n_samples=n_samples,
        signal_mass_ratio=signal_mass_ratio,
        n_modes=n_modes,
        signal_means_offset=signal_means_offset,
        signal_means_jitter=signal_means_jitter,
        noise_means_offset=noise_means_offset,
        noise_means_jitter=noise_means_jitter,
        noise_subspace_rank=noise_subspace_rank,
        signal_weight_concentration=signal_weight_concentration,
        noise_weight_concentration=noise_weight_concentration,
        signal_mean_shift=signal_mean_shift,
        signal_cov_scale=signal_cov_scale,
        signal_anisotropy=signal_anisotropy,
        cov_rotation_jitter=cov_rotation_jitter,
        cov_scale_jitter=cov_scale_jitter,
        global_rotation=global_rotation,
        random_seed=random_seed,
    )
    adata = _build_anndata_from_synth(data, n_dim)
    if gmm_key is not None:
        _persist_ground_truth_gmm_schema(adata, data, gmm_key=gmm_key)
    return adata


def from_synth_gmm(
    *,
    representation: Literal["cells", "gmm"] = "cells",
    adata: bool = False,
    gmm_key: str | None = None,
    t: int = 4,
    n_dim: int = 10,
    n_patients: int = 6,
    n_samples: int = 250,
    signal_mass_ratio: float = 0.2,
    n_modes: int = 10,
    signal_means_offset: float = 12.0,
    signal_means_jitter: float = 0.75,
    noise_means_offset: float = 3.0,
    noise_means_jitter: float = 0.75,
    noise_subspace_rank: int = 2,
    signal_weight_concentration: float | None = None,
    noise_weight_concentration: float | None = None,
    signal_mean_shift: float = 1.0,
    signal_cov_scale: float = 1.2,
    signal_anisotropy: float = 12.0,
    cov_rotation_jitter: float = 10.0,
    cov_scale_jitter: float = 0.15,
    global_rotation: float = 30.0,
    random_seed: int = 42,
) -> TripletDataset:
    """Create a GGML dataset from the synthetic GMM generator.

    Wraps :func:`synth_gmm` and returns a dataset that can be used directly
    with training and evaluation functions.

    Parameters
    ----------
    representation
        How patient distributions are represented in the dataset.
        ``"cells"`` (default) samples ``n_samples`` cells per patient and
        stores them as empirical point clouds.
        ``"gmm"`` stores the analytical per-patient GMM component
        parameters directly (means, covariances, weights).
    adata
        If ``True``, wrap the dataset in an
        :class:`~ggml_ot.data.AnnData_TripletDataset` backed by an
        ``AnnData`` object. Required for ``gmm_key`` to have any effect.
    gmm_key
        When ``adata=True``, persist the analytical raw-space ground-truth
        GMM under ``dataset.adata.uns[gmm_key]``. Requires ``adata=True``.
    t
        Number of triplets sampled per anchor distribution.
    **kwargs
        All remaining keyword arguments (``n_dim``, ``n_patients``,
        ``n_samples``, etc.) are forwarded to :func:`synth_gmm`.
        See its documentation for details.

    Returns
    -------
    TripletDataset | AnnData_TripletDataset
        A dataset ready for use with :func:`ggml_ot.train` or
        :func:`ggml_ot.train_gmm`.

    Raises
    ------
    ValueError
        If ``representation`` is not ``"cells"`` or ``"gmm"``, or if
        ``gmm_key`` is set without ``adata=True``.
    """
    if representation not in ("cells", "gmm"):
        raise ValueError(f"representation must be 'cells' or 'gmm', got {representation!r}")
    if gmm_key is not None and not adata:
        raise ValueError("`gmm_key` requires `adata=True`.")

    synth_kwargs = dict(
        n_dim=n_dim,
        n_patients=n_patients,
        n_samples=n_samples,
        signal_mass_ratio=signal_mass_ratio,
        n_modes=n_modes,
        signal_means_offset=signal_means_offset,
        signal_means_jitter=signal_means_jitter,
        noise_means_offset=noise_means_offset,
        noise_means_jitter=noise_means_jitter,
        noise_subspace_rank=noise_subspace_rank,
        signal_weight_concentration=signal_weight_concentration,
        noise_weight_concentration=noise_weight_concentration,
        signal_mean_shift=signal_mean_shift,
        signal_cov_scale=signal_cov_scale,
        signal_anisotropy=signal_anisotropy,
        cov_rotation_jitter=cov_rotation_jitter,
        cov_scale_jitter=cov_scale_jitter,
        global_rotation=global_rotation,
        random_seed=random_seed,
    )
    data = synth_gmm(**synth_kwargs)
    samples = data["samples"]
    patient_ids = np.asarray(samples["patient_ids"])
    labels_arr = np.asarray(samples["labels"])
    X_high = data["X_high_all"]

    if adata:
        from ggml_ot.data.anndata import AnnData_TripletDataset

        adata_obj = _build_anndata_from_synth(data, n_dim)
        if gmm_key is not None:
            _persist_ground_truth_gmm_schema(adata_obj, data, gmm_key=gmm_key)
        dataset = AnnData_TripletDataset(
            adata_obj,
            patient_col="sample",
            label_col="patient_group",
            n_cells=n_samples,
            n_triplets=t,
        )
    else:
        unique_pids = np.unique(patient_ids)
        distributions = []
        distribution_labels = []
        for patient_id in unique_pids:
            mask = patient_ids == patient_id
            distributions.append(X_high[mask])
            distribution_labels.append(int(labels_arr[mask][0]))
        dataset = TripletDataset(distributions, distribution_labels, t)

    if representation == "gmm":
        from ggml_ot.gmm._generic_dataset_interface import _apply_gmm_fields_in_place

        patient_gmms = _ordered_patient_gmms_for_adata(data, adata_obj) if adata else data["patient_gmms"]

        supports_out = np.stack([entry["means"] for entry in patient_gmms])
        covs_out = np.stack([entry["covs"] for entry in patient_gmms])
        weights_out = np.stack([entry["weights"] for entry in patient_gmms])
        _apply_gmm_fields_in_place(
            dataset,
            supports=supports_out,
            covariances=covs_out,
            weights=weights_out,
            identical_supports=False,
            gmm_provenance="synthetic_ground_truth",
        )
        dataset.distribution_labels = [entry["label"] for entry in patient_gmms]

    dataset.synth_data = data
    dataset.synth_data["fitted_gmm_provenance"] = "synthetic_ground_truth" if representation == "gmm" else "none"
    return dataset


def _validate_synth_inputs(**kwargs) -> None:
    if kwargs["n_dim"] < 2:
        raise ValueError("`n_dim` must be at least 2.")
    if kwargs["n_patients"] <= 0 or kwargs["n_patients"] % 2 != 0:
        raise ValueError("`n_patients` must be a positive even integer.")
    if kwargs["n_samples"] < 2:
        raise ValueError("`n_samples` must be at least 2.")
    if kwargs["n_samples"] < kwargs["n_modes"]:
        raise ValueError("`n_samples` must be at least `n_modes` so every mode can carry mass.")
    if not 0 < kwargs["signal_mass_ratio"] <= 1:
        raise ValueError("`signal_mass_ratio` must be in the interval (0, 1].")
    if kwargs["n_modes"] < 2:
        raise ValueError("`n_modes` must be at least 2.")
    for name in (
        "signal_means_offset",
        "signal_means_jitter",
        "noise_means_offset",
        "noise_means_jitter",
        "signal_cov_scale",
        "signal_anisotropy",
        "cov_rotation_jitter",
        "cov_scale_jitter",
    ):
        if kwargs[name] < 0:
            raise ValueError(f"`{name}` must be non-negative.")
    if int(kwargs["noise_subspace_rank"]) < 1:
        raise ValueError("`noise_subspace_rank` must be a positive integer.")
    for name in ("signal_weight_concentration", "noise_weight_concentration"):
        value = kwargs[name]
        if value is not None and value <= 0:
            raise ValueError(f"`{name}` must be positive when provided.")


def _build_signal_means(signal_plane_offset: float, signal_mean_shift: float) -> np.ndarray:
    return np.array(
        [
            [signal_plane_offset - signal_mean_shift, 0.0],
            [signal_plane_offset + signal_mean_shift, 0.0],
            [0.0, signal_plane_offset + signal_mean_shift],
            [0.0, signal_plane_offset - signal_mean_shift],
        ]
    )


def _build_signal_covariances(
    *,
    signal_cov_scale: float,
    signal_anisotropy: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    major = signal_cov_scale * np.sqrt(signal_anisotropy)
    minor = signal_cov_scale / np.sqrt(signal_anisotropy)
    covariances = np.array(
        [
            [[major * 0.9, 0.0], [0.0, minor * 4.0]],
            [[major * 1.8, 0.0], [0.0, minor * 0.4]],
            [[minor * 1.5, 0.0], [0.0, major * 1.6]],
            [[minor * 2.5, 0.0], [0.0, major * 0.7]],
        ]
    )
    rotations = [20.0, -5.0, -10.0 + rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0)]

    covs = []
    for cov, angle in zip(covariances, rotations):
        rotated = _rotation_matrix(angle) @ cov @ _rotation_matrix(angle).T
        covs.append(_project_psd(rotated))
    return np.asarray(covs)


def _signal_axis_jitter(*, signal_idx: int, rng: np.random.RandomState, scale: float) -> np.ndarray:
    if scale == 0:
        return np.zeros(2)
    delta = rng.normal(scale=scale)
    if signal_idx in (0, 1):
        return np.array([delta, 0.0])
    return np.array([0.0, delta])


def _build_planar_archetypes(
    *,
    n_modes: int,
    rng: np.random.RandomState,
    radius_low: float,
    radius_high: float,
) -> np.ndarray:
    if n_modes == 0:
        return np.zeros((0, 2))
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_modes, endpoint=False)
    rng.shuffle(angles)
    radii = rng.uniform(radius_low, radius_high, size=n_modes)
    return np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])


def _build_subspace_archetypes(
    *,
    n_modes: int,
    n_dims: int,
    rng: np.random.RandomState,
    radius_low: float,
    radius_high: float,
) -> np.ndarray:
    if n_modes == 0:
        return np.zeros((0, n_dims))
    if n_dims == 0:
        return np.zeros((n_modes, 0))
    dirs = rng.normal(size=(n_modes, n_dims))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    radii = rng.uniform(radius_low, radius_high, size=(n_modes, 1))
    return dirs * radii


def _build_subspace_covariances(
    *,
    n_modes: int,
    n_dims: int,
    rng: np.random.RandomState,
    cov_scale: float,
) -> np.ndarray:
    if n_modes == 0:
        return np.zeros((0, n_dims, n_dims))
    if n_dims == 0:
        return np.zeros((n_modes, 0, 0))
    lambda_min = max(1.0e-4, 0.1 * cov_scale)
    lambda_max = max(lambda_min, 2.0 * cov_scale - lambda_min)
    covs = []
    for _ in range(n_modes):
        Q, _ = np.linalg.qr(rng.normal(size=(n_dims, n_dims)))
        eigvals = rng.uniform(lambda_min, lambda_max, size=n_dims)
        covs.append(_project_psd(Q @ np.diag(eigvals) @ Q.T))
    return np.asarray(covs)


def _build_spiked_subspace_covariances(
    *,
    n_modes: int,
    n_dims: int,
    rng: np.random.RandomState,
    cov_scale: float,
    rank: int,
) -> np.ndarray:
    if n_modes == 0:
        return np.zeros((0, n_dims, n_dims))
    if n_dims == 0:
        return np.zeros((n_modes, 0, 0))

    resolved_rank = min(max(1, int(rank)), n_dims)
    isotropic_floor = max(1.0e-4, 0.15 * cov_scale)
    total_variance = cov_scale * n_dims
    spike_budget = max(0.0, total_variance - isotropic_floor * n_dims)

    covs = []
    for _ in range(n_modes):
        U, _ = np.linalg.qr(rng.normal(size=(n_dims, resolved_rank)))
        spike_weights = rng.dirichlet(np.full(resolved_rank, 2.0, dtype=np.float64))
        spike_strengths = spike_budget * spike_weights
        cov = isotropic_floor * np.eye(n_dims) + U @ np.diag(spike_strengths) @ U.T
        covs.append(_project_psd(cov))
    return np.asarray(covs)


def _random_2d_cov(
    rng: np.random.RandomState,
    *,
    cov_scale: float,
    anisotropy_low: float,
    anisotropy_high: float,
) -> np.ndarray:
    angle = rng.uniform(-180.0, 180.0)
    anisotropy = rng.uniform(anisotropy_low, anisotropy_high)
    major = cov_scale * np.sqrt(anisotropy)
    minor = cov_scale / np.sqrt(anisotropy)
    cov = _rotation_matrix(angle) @ np.diag([major, minor]) @ _rotation_matrix(angle).T
    return _project_psd(cov)


def _sample_positive_mode_counts(
    total: int,
    n_modes: int,
    *,
    rng: np.random.RandomState,
    concentration: float | None,
) -> list[int]:
    if n_modes == 0:
        return []
    if total < n_modes:
        raise ValueError("`total` must be at least `n_modes` to assign positive mass to every mode.")
    if concentration is None:
        base = total // n_modes
        counts = [base] * n_modes
        for idx in range(total - base * n_modes):
            counts[idx] += 1
        return counts

    remainder = total - n_modes
    if remainder == 0:
        return [1] * n_modes
    probs = rng.dirichlet(np.full(n_modes, float(concentration), dtype=np.float64))
    extra = rng.multinomial(remainder, probs)
    return (1 + extra).tolist()


def _split_mode_counts(
    n_samples: int,
    signal_mass_ratio: float,
    n_noise: int,
    *,
    rng: np.random.RandomState,
    signal_weight_concentration: float | None,
    noise_weight_concentration: float | None,
) -> tuple[list[int], list[int]]:
    if n_noise == 0:
        return _sample_positive_mode_counts(
            n_samples,
            2,
            rng=rng,
            concentration=signal_weight_concentration,
        ), []

    signal_total = int(round(n_samples * signal_mass_ratio))
    signal_total = min(max(signal_total, 2), n_samples - n_noise)
    signal_counts = _sample_positive_mode_counts(
        signal_total,
        2,
        rng=rng,
        concentration=signal_weight_concentration,
    )

    noise_total = n_samples - sum(signal_counts)
    noise_counts = _sample_positive_mode_counts(
        noise_total,
        n_noise,
        rng=rng,
        concentration=noise_weight_concentration,
    )
    return signal_counts, noise_counts


def _jitter_covariance(
    cov: np.ndarray,
    *,
    rng: np.random.RandomState,
    rotation_jitter: float,
    scale_jitter: float,
) -> np.ndarray:
    out = cov.copy()
    if rotation_jitter > 0:
        R = _random_rotation_like_identity(
            n_dims=out.shape[0],
            rng=rng,
            rotation_jitter=rotation_jitter,
        )
        out = R @ out @ R.T
    out = _scale_covariance(out, _sample_cov_scale(rng, scale_jitter))
    return _project_psd(out)


def _sample_cov_scale(rng: np.random.RandomState, scale_jitter: float) -> float:
    if scale_jitter == 0:
        return 1.0
    lower = max(1e-6, 1.0 - scale_jitter)
    upper = 1.0 + scale_jitter
    return float(rng.uniform(lower, upper))


def _scale_covariance(cov: np.ndarray, scale: float) -> np.ndarray:
    if cov.size == 0:
        return cov.copy()
    return _project_psd(cov * scale)


def _jitter_subspace_mean(mean: np.ndarray, *, rng: np.random.RandomState, scale: float) -> np.ndarray:
    if mean.size == 0 or scale == 0:
        return mean.copy()
    return mean + rng.normal(scale=scale, size=mean.shape)


def _rotation_matrix(angle_degrees: float) -> np.ndarray:
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _random_rotation_like_identity(
    *,
    n_dims: int,
    rng: np.random.RandomState,
    rotation_jitter: float,
) -> np.ndarray:
    if n_dims < 2 or rotation_jitter == 0:
        return np.eye(n_dims)
    if n_dims == 2:
        return _rotation_matrix(rng.uniform(-rotation_jitter, rotation_jitter))

    R = np.eye(n_dims)
    n_steps = max(1, n_dims // 2)
    for _ in range(n_steps):
        i, j = rng.choice(n_dims, size=2, replace=False)
        angle = rng.uniform(-rotation_jitter, rotation_jitter)
        G = np.eye(n_dims)
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = -s
        G[j, i] = s
        R = G @ R
    return R


def _project_psd(cov: np.ndarray) -> np.ndarray:
    if cov.size == 0:
        return cov.copy()
    cov = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _embed_to_high_dim(
    X_rot: np.ndarray,
    *,
    subspace_mean: np.ndarray,
    subspace_cov: np.ndarray,
    Q_high: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    n_samples, n_dim = X_rot.shape[0], Q_high.shape[0]
    X_pad = np.zeros((n_samples, n_dim))
    X_pad[:, :2] = X_rot
    if subspace_mean.size > 0:
        X_pad[:, 2:] = rng.multivariate_normal(subspace_mean, subspace_cov, n_samples)
    return X_pad @ Q_high.T


def _high_dim_mean(mean_rot: np.ndarray, subspace_mean: np.ndarray, Q_high: np.ndarray) -> np.ndarray:
    mu_pad = np.zeros(Q_high.shape[0])
    mu_pad[:2] = mean_rot
    if subspace_mean.size > 0:
        mu_pad[2:] = subspace_mean
    return mu_pad @ Q_high.T


def _high_dim_cov(cov_rot: np.ndarray, subspace_cov: np.ndarray, Q_high: np.ndarray) -> np.ndarray:
    sigma_pad = np.zeros((Q_high.shape[0], Q_high.shape[0]))
    sigma_pad[:2, :2] = cov_rot
    if subspace_cov.size > 0:
        sigma_pad[2:, 2:] = subspace_cov
    return _project_psd(Q_high @ sigma_pad @ Q_high.T)
