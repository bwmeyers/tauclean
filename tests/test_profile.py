"""Tests for the profile data container."""

from __future__ import annotations

import numpy as np
import pytest

from tauclean.noise import UserDefinedOnPulseNoiseEstimator
from tauclean.profile import ProfileData


def test_profile_initializes_regions_baseline_and_noise() -> None:
    samples = np.array([2.0, 2.0, 4.0, 8.0, 2.0, 2.0])
    profile = ProfileData(samples=samples, period=12.0)

    profile.initialize_noise(UserDefinedOnPulseNoiseEstimator(2, 4))

    assert profile.nbins == 6
    assert profile.bin_width == 2.0
    assert profile.baseline == 2.0
    np.testing.assert_array_equal(profile.on_bins, [2, 3])
    np.testing.assert_array_equal(profile.off_bins, [0, 1, 4, 5])
    np.testing.assert_array_equal(profile.baseline_corrected, [0, 0, 2, 6, 0, 0])
    assert profile.initial_noise == profile.current_noise


def test_profile_update_noise_uses_existing_regions() -> None:
    profile = ProfileData(samples=np.arange(8.0), period=8.0)
    profile.initialize_noise(UserDefinedOnPulseNoiseEstimator(2, 6))
    residual = np.zeros(8)
    residual[2:6] = 3.0

    stats = profile.update_noise(residual)

    assert profile.current_noise is stats
    assert stats.off_rms == 0.0
    assert stats.on_mean == 3.0
    np.testing.assert_array_equal(profile.on_pulse(residual), np.full(4, 3.0))


def test_profile_requires_initialized_regions() -> None:
    profile = ProfileData(samples=np.ones(4), period=4.0)

    with pytest.raises(ValueError, match="Baseline-corrected"):
        profile.off_pulse()
    with pytest.raises(ValueError, match="On-pulse bins"):
        profile.on_pulse(np.ones(4))