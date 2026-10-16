"""Noise estimation strategies and noise statistics objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .circular import wrap_indices


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
    """Infer off-pulse bins from user-defined on-pulse range(s).

    The on-pulse region is the union of one or more ``(start, end)`` bin
    ranges (end-exclusive). Profiles are cyclic, so ``start > end`` selects a
    range that wraps through the end of the profile (e.g. ``900 100`` for 1024
    bins covers bins 900-1023 and 0-99). Everything outside the union is
    off-pulse.

    Either pass a single range as ``UserDefinedOnPulseNoiseEstimator(start,
    end)`` or several as ``UserDefinedOnPulseNoiseEstimator(ranges=[(s1, e1),
    (s2, e2)])``.
    """

    def __init__(
        self,
        on_start: int | None = None,
        on_end: int | None = None,
        ranges: Sequence[tuple[int, int]] | None = None,
    ):
        if ranges is not None:
            if on_start is not None or on_end is not None:
                raise ValueError(
                    "Provide either on_start/on_end or ranges, not both"
                )
            parsed = [(int(start), int(end)) for start, end in ranges]
        elif on_start is not None and on_end is not None:
            parsed = [(int(on_start), int(on_end))]
        else:
            raise ValueError("Both on_start and on_end are required")
        if not parsed:
            raise ValueError("At least one on-pulse range is required")
        self.ranges = parsed

    @property
    def on_start(self) -> int:
        return self.ranges[0][0]

    @property
    def on_end(self) -> int:
        return self.ranges[0][1]

    @classmethod
    def from_string(
        cls, onpulse_estimator: str
    ) -> UserDefinedOnPulseNoiseEstimator:
        """Parse ``'START END'`` or comma-separated ``'S1 E1,S2 E2'``.

        ``'START-END'`` is accepted as an alternative to ``'START END'``
        within each range.
        """
        ranges = []
        for text in onpulse_estimator.split(","):
            text = text.strip()
            if not text:
                continue
            parts = text.replace("-", " ").split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid on-pulse range {text!r} in "
                    f"{onpulse_estimator!r}: expected 'START END'."
                )
            try:
                ranges.append((int(parts[0]), int(parts[1])))
            except ValueError:
                raise ValueError(
                    f"Invalid on-pulse range {text!r} in "
                    f"{onpulse_estimator!r}: bounds must be integers."
                ) from None
        if not ranges:
            raise ValueError(
                f"No on-pulse ranges found in {onpulse_estimator!r}"
            )
        return cls(ranges=ranges)

    def estimate_regions(
        self, samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nbins = len(samples)
        bins = np.arange(nbins)
        on_bins = np.concatenate(
            [wrap_indices(start, end, nbins) for start, end in self.ranges]
        )
        if len(self.ranges) > 1:
            on_bins = np.unique(on_bins)
        off_bins = bins[np.logical_not(np.in1d(bins, on_bins))]
        return off_bins, on_bins
