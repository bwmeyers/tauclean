"""Profile domain object with owned noise and region properties."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .noise import NoiseEstimator, NoiseStats


@dataclass
class ProfileData:
    """Profile container that owns region and noise state."""

    samples: np.ndarray
    period: float
    off_bins: np.ndarray | None = None
    on_bins: np.ndarray | None = None
    baseline_corrected: np.ndarray | None = None
    initial_noise: NoiseStats | None = None
    current_noise: NoiseStats | None = None
    _baseline: float = field(default=0.0, init=False, repr=False)

    @property
    def nbins(self) -> int:
        return self.samples.size

    @property
    def bin_width(self) -> float:
        return self.period / self.nbins

    @property
    def baseline(self) -> float:
        return self._baseline

    def initialize_noise(self, estimator: NoiseEstimator) -> None:
        """Estimate regions, remove baseline, and compute initial noise."""
        off_bins, on_bins = estimator.estimate_regions(self.samples)
        self.off_bins = off_bins
        self.on_bins = on_bins

        self._baseline = float(np.mean(self.samples[off_bins]))
        self.baseline_corrected = self.samples - self._baseline
        stats = self._compute_stats(self.baseline_corrected)
        self.initial_noise = stats
        self.current_noise = stats

    def update_noise(self, residual: np.ndarray) -> NoiseStats:
        """Update current noise values using the latest residual."""
        stats = self._compute_stats(residual)
        self.current_noise = stats
        return stats

    def off_pulse(self, arr: np.ndarray | None = None) -> np.ndarray:
        data = self._resolve_data(arr)
        return data[self._require_off_bins()]

    def on_pulse(self, arr: np.ndarray | None = None) -> np.ndarray:
        data = self._resolve_data(arr)
        return data[self._require_on_bins()]

    def _compute_stats(self, arr: np.ndarray) -> NoiseStats:
        off = arr[self._require_off_bins()]
        on = arr[self._require_on_bins()]
        return NoiseStats(
            baseline=self._baseline,
            off_rms=float(np.std(off)),
            on_rms=float(np.std(on)),
            off_mean=float(np.mean(off)),
            on_mean=float(np.mean(on)),
            total_mean=float(np.mean(arr)),
            total_rms=float(np.std(arr)),
        )

    def _resolve_data(self, arr: np.ndarray | None) -> np.ndarray:
        if arr is not None:
            return arr
        if self.baseline_corrected is None:
            raise ValueError("Baseline-corrected profile is not initialized")
        return self.baseline_corrected

    def _require_off_bins(self) -> np.ndarray:
        if self.off_bins is None:
            raise ValueError("Off-pulse bins are not initialized")
        return self.off_bins

    def _require_on_bins(self) -> np.ndarray:
        if self.on_bins is None:
            raise ValueError("On-pulse bins are not initialized")
        return self.on_bins
