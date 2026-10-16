"""Tests for pulse-region noise estimators."""

from __future__ import annotations

import numpy as np
import pytest

from tauclean.noise import (
    AutoWindowNoiseEstimator,
    UserDefinedOnPulseNoiseEstimator,
)


def test_user_defined_estimator_partitions_profile() -> None:
    samples = np.zeros(10)

    off_bins, on_bins = UserDefinedOnPulseNoiseEstimator(
        3, 7
    ).estimate_regions(samples)

    np.testing.assert_array_equal(on_bins, [3, 4, 5, 6])
    np.testing.assert_array_equal(off_bins, [0, 1, 2, 7, 8, 9])
    assert not np.intersect1d(off_bins, on_bins).size


def test_user_defined_estimator_parses_cli_value() -> None:
    estimator = UserDefinedOnPulseNoiseEstimator.from_string("4 9")

    assert (estimator.on_start, estimator.on_end) == (4, 9)


def test_auto_window_estimator_selects_quiet_region() -> None:
    samples = np.zeros(32)
    samples[12:20] = 50.0

    off_bins, on_bins = AutoWindowNoiseEstimator(
        windowsize=8
    ).estimate_regions(samples)

    assert len(off_bins) == 8
    assert not np.intersect1d(off_bins, on_bins).size
    assert np.all(samples[off_bins] == 0.0)


def test_user_defined_estimator_wraps_through_profile_end() -> None:
    samples = np.zeros(10)

    off_bins, on_bins = UserDefinedOnPulseNoiseEstimator(
        8, 2
    ).estimate_regions(samples)

    np.testing.assert_array_equal(sorted(on_bins), [0, 1, 8, 9])
    np.testing.assert_array_equal(off_bins, [2, 3, 4, 5, 6, 7])
    assert not np.intersect1d(off_bins, on_bins).size


def test_user_defined_estimator_full_profile_end_bound() -> None:
    _, on_bins = UserDefinedOnPulseNoiseEstimator(0, 10).estimate_regions(
        np.zeros(10)
    )
    assert on_bins.size == 10


def test_user_defined_estimator_multiple_ranges() -> None:
    estimator = UserDefinedOnPulseNoiseEstimator(ranges=[(1, 3), (6, 8)])

    off_bins, on_bins = estimator.estimate_regions(np.zeros(10))

    np.testing.assert_array_equal(on_bins, [1, 2, 6, 7])
    np.testing.assert_array_equal(off_bins, [0, 3, 4, 5, 8, 9])


def test_user_defined_estimator_multiple_ranges_wrap_and_overlap() -> None:
    estimator = UserDefinedOnPulseNoiseEstimator(
        ranges=[(8, 2), (1, 4), (6, 7)]
    )

    off_bins, on_bins = estimator.estimate_regions(np.zeros(10))

    np.testing.assert_array_equal(on_bins, [0, 1, 2, 3, 6, 8, 9])
    np.testing.assert_array_equal(off_bins, [4, 5, 7])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4 9", [(4, 9)]),
        ("4-9", [(4, 9)]),
        ("100 200,400 450", [(100, 200), (400, 450)]),
        (" 900 100 , 10-20 ", [(900, 100), (10, 20)]),
    ],
)
def test_user_defined_estimator_from_string_ranges(text, expected) -> None:
    assert UserDefinedOnPulseNoiseEstimator.from_string(text).ranges == expected


@pytest.mark.parametrize("text", ["", " , ", "4", "1 2 3", "a b", "1 2,3"])
def test_user_defined_estimator_from_string_rejects_bad_input(text) -> None:
    with pytest.raises(ValueError):
        UserDefinedOnPulseNoiseEstimator.from_string(text)


def test_user_defined_estimator_constructor_validation() -> None:
    with pytest.raises(ValueError):
        UserDefinedOnPulseNoiseEstimator()
    with pytest.raises(ValueError):
        UserDefinedOnPulseNoiseEstimator(1, 2, ranges=[(3, 4)])
    with pytest.raises(ValueError):
        UserDefinedOnPulseNoiseEstimator(ranges=[])
