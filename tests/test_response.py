"""Tests for instrumental response helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tauclean.response import (
    get_instrumental_response,
    get_restoring_function,
    reconstruct,
)

PERIOD_MS = 500.0
NBINS = 1024
DT_MS = PERIOD_MS / NBINS


def _profile_with_peak(peak_idx: int = NBINS // 2) -> np.ndarray:
    profile = np.zeros(NBINS)
    profile[peak_idx] = 1.0
    return profile


def _area(response: np.ndarray) -> float:
    return float(np.trapz(y=response, dx=DT_MS))


def _effective_width(response: np.ndarray) -> float:
    return _area(response) / float(np.max(response))


def test_instrumental_response_rejects_non_finite_widths() -> None:
    profile = _profile_with_peak()

    with pytest.raises(ValueError, match="finite"):
        get_instrumental_response(
            profile,
            PERIOD_MS,
            r_dm_width=np.nan,
            r_pb_width=0.25,
            r_av_width=0.2,
            r_pd_width=0.1,
            fast=False,
        )


def test_instrumental_response_rejects_non_positive_widths() -> None:
    profile = _profile_with_peak()

    with pytest.raises(ValueError, match=r"> 0"):
        get_instrumental_response(
            profile,
            PERIOD_MS,
            r_dm_width=0.0,
            r_pb_width=-1.0,
            r_av_width=0.0,
            r_pd_width=-2.0,
            fast=False,
        )


def test_single_boxcar_response_is_normalized_and_centered() -> None:
    profile = _profile_with_peak()

    response, resp_width = get_instrumental_response(
        profile,
        PERIOD_MS,
        r_dm_width=3.0,
        r_pb_width=0.0,
        r_av_width=0.0,
        r_pd_width=0.0,
        fast=False,
    )

    assert np.all(np.isfinite(response))
    assert np.min(response) >= -1e-12
    assert np.isclose(_area(response), 1.0, rtol=2e-3)
    assert abs(int(np.argmax(response)) - NBINS // 2) <= 4
    assert resp_width > 0


def test_two_equal_boxcars_broaden_response() -> None:
    profile = _profile_with_peak()

    one_boxcar, _ = get_instrumental_response(
        profile,
        PERIOD_MS,
        r_dm_width=3.0,
        r_pb_width=0.0,
        r_av_width=0.0,
        r_pd_width=0.0,
        fast=False,
    )
    two_boxcars, _ = get_instrumental_response(
        profile,
        PERIOD_MS,
        r_dm_width=3.0,
        r_pb_width=3.0,
        r_av_width=0.0,
        r_pd_width=0.0,
        fast=False,
    )

    assert np.isclose(_area(one_boxcar), 1.0, rtol=2e-3)
    assert np.isclose(_area(two_boxcars), 1.0, rtol=2e-3)
    assert _effective_width(two_boxcars) > _effective_width(one_boxcar)


def test_mixed_boxcars_are_finite_and_order_robust() -> None:
    profile = _profile_with_peak()

    resp_a, width_a = get_instrumental_response(
        profile,
        PERIOD_MS,
        r_dm_width=2.0,
        r_pb_width=3.5,
        r_av_width=6.0,
        r_pd_width=0.0,
        fast=False,
    )
    resp_b, width_b = get_instrumental_response(
        profile,
        PERIOD_MS,
        r_dm_width=3.5,
        r_pb_width=6.0,
        r_av_width=2.0,
        r_pd_width=0.0,
        fast=False,
    )

    assert np.all(np.isfinite(resp_a))
    assert np.all(np.isfinite(resp_b))
    assert np.min(resp_a) >= -1e-12
    assert np.min(resp_b) >= -1e-12
    assert np.isclose(_area(resp_a), 1.0, rtol=2e-3)
    assert np.isclose(_area(resp_b), 1.0, rtol=2e-3)
    assert np.isclose(width_a, width_b, rtol=2e-2)
    assert np.allclose(resp_a, resp_b, rtol=3e-2, atol=2e-4)


def test_fast_mode_returns_delta_at_profile_peak() -> None:
    peak_idx = 123
    profile = _profile_with_peak(peak_idx=peak_idx)

    response, resp_width = get_instrumental_response(
        profile,
        PERIOD_MS,
        r_dm_width=np.nan,
        r_pb_width=np.nan,
        r_av_width=np.nan,
        r_pd_width=np.nan,
        fast=True,
    )

    assert np.count_nonzero(response) == 1
    assert int(np.argmax(response)) == peak_idx
    assert np.isclose(resp_width, DT_MS)


def test_get_restoring_function_uses_equivalent_width_policy() -> None:
    profile = _profile_with_peak()
    inst_width = 4.0

    rest_func = get_restoring_function(profile, PERIOD_MS, inst_width)
    effective_width = _effective_width(rest_func)

    assert np.isclose(_area(rest_func), 1.0, rtol=2e-3)
    assert np.isclose(effective_width, inst_width, rtol=5e-2)


def test_get_restoring_function_rejects_invalid_width() -> None:
    profile = _profile_with_peak()

    with pytest.raises(ValueError, match=r"finite and > 0"):
        get_restoring_function(profile, PERIOD_MS, np.nan)


def test_reconstruct_handles_zero_components_without_nan() -> None:
    clean_components = np.zeros(NBINS)

    reconstruction = reconstruct(clean_components)

    assert np.all(np.isfinite(reconstruction))
    assert np.allclose(reconstruction, 0.0)
