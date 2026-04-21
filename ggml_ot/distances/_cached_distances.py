"""Cached distance matrix helpers used by OT computations."""

from __future__ import annotations

import numpy as np

from ggml_ot import settings
from ggml_ot.distances.mahalanobis import pairwise_mahalanobis_distance


class Computed_Distances:
    """Compute and cache pairwise Mahalanobis distances on demand."""

    def __init__(self, points, theta):
        self.points = points
        self.theta = theta
        self.n_threads = settings.n_threads

        self.data = np.full((len(points), len(points)), np.nan)
        self.ndim = self.data.ndim
        self.shape = self.data.shape

    def __getitem__(self, slice_):
        if np.isnan(self.data[slice_]).any():
            ranges = [np.squeeze(np.arange(len(self.data))[slice_[i]]) for i in range(len(slice_))]
            entry_nan_index = ([], [])
            for entry in ranges[0]:
                check = np.isnan(self.data[entry, :])
                if check.ndim == 2 and np.isnan(self.data[entry, :][:, slice_[1]]).any():
                    entry_nan_index[0].append(entry)
                elif check.ndim == 1 and np.isnan(self.data[entry, :][slice_[1]]).any():
                    entry_nan_index[0].append(entry)
            for entry in ranges[1]:
                if np.isnan(self.data[slice_[0], entry]).any():
                    entry_nan_index[1].append(entry)

            dist = pairwise_mahalanobis_distance(
                self.points[entry_nan_index[0], :],
                self.points[entry_nan_index[1], :],
                ground_metric=self.theta,
            )
            self.data[np.ix_(entry_nan_index[0], entry_nan_index[1])] = dist
            return self.data[slice_]

        return self.data[slice_]


__all__ = ["Computed_Distances"]
