"""Tests for pulse-region noise estimators."""

from __future__ import annotations

import numpy as np

from tauclean.domain.noise import (
    AutoWindowNoiseEstimator,
    UserDefinedOnPulseNoiseEstimator,
)


def test_user_defined_estimator_partitions_profile() -> None:
    samples = np.zeros(10)

    off_bins, on_bins = UserDefinedOnPulseNoiseEstimator(3, 7).estimate_regions(
        samples
    )

    np.testing.assert_array_equal(on_bins, [3, 4, 5, 6])
    np.testing.assert_array_equal(off_bins, [0, 1, 2, 7, 8, 9])
    assert not np.intersect1d(off_bins, on_bins).size


def test_user_defined_estimator_parses_cli_value() -> None:
    estimator = UserDefinedOnPulseNoiseEstimator.from_string("4 9")

    assert (estimator.on_start, estimator.on_end) == (4, 9)


def test_auto_window_estimator_selects_quiet_region() -> None:
    samples = np.zeros(32)
    samples[12:20] = 50.0

    off_bins, on_bins = AutoWindowNoiseEstimator(windowsize=8).estimate_regions(
        samples
    )

    assert len(off_bins) == 8
    assert not np.intersect1d(off_bins, on_bins).size
    assert np.all(samples[off_bins] == 0.0)