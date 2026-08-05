"""Noise estimation strategies and noise statistics objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseStats:
    """Noise summary values for a profile and selected regions."""

    baseline: float
    off_rms: float
    on_rms: float
    off_mean: float
    on_mean: float
    total_mean: float
    total_rms: float


class NoiseEstimator(ABC):
    """Strategy for identifying profile on/off-pulse regions."""

    @abstractmethod
    def estimate_regions(
        self, samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return off-pulse and on-pulse bin indices."""


class AutoWindowNoiseEstimator(NoiseEstimator):
    """Estimate off-pulse region by minimizing the integrated window area."""

    def __init__(self, windowsize: int | None = None):
        self.windowsize = windowsize

    def estimate_regions(
        self, samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nbins = len(samples)
        bins = np.arange(nbins)
        windowsize = (
            self.windowsize if self.windowsize is not None else nbins // 8
        )

        integral = np.zeros_like(samples)
        for idx in range(nbins):
            win = (
                np.arange(idx - windowsize // 2, idx + windowsize // 2) % nbins
            )
            integral[idx] = np.trapz(samples[win])

        minidx = np.argmin(integral)
        off_bins = (
            np.arange(minidx - windowsize // 2, minidx + windowsize // 2)
            % nbins
        )
        on_bins = bins[np.logical_not(np.in1d(bins, off_bins))]
        return off_bins, on_bins


class UserDefinedOnPulseNoiseEstimator(NoiseEstimator):
    """Infer off-pulse bins from a user-defined on-pulse range."""

    def __init__(self, on_start: int, on_end: int):
        self.on_start = on_start
        self.on_end = on_end

    @classmethod
    def from_string(
        cls, onpulse_estimator: str
    ) -> UserDefinedOnPulseNoiseEstimator:
        start, end = onpulse_estimator.split(" ")
        return cls(int(start), int(end))

    def estimate_regions(
        self, samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nbins = len(samples)
        bins = np.arange(nbins)
        on_bins = bins[self.on_start : self.on_end]
        off_bins = bins[np.logical_not(np.in1d(bins, on_bins))]
        return off_bins, on_bins
