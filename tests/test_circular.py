"""Tests for cyclic profile helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tauclean.circular import (
    bin_grid,
    center_to_origin,
    circ_convolve,
    circ_shift_to,
    wrap_indices,
)


def _brute_force(a, b):
    n = a.size
    out = np.zeros(n)
    for i in range(n):
        for j in range(n):
            out[(i + j) % n] += a[i] * b[j]
    return out


def test_circ_convolve_matches_brute_force(rng):
    a = rng.normal(size=33)
    b = rng.normal(size=33)
    np.testing.assert_allclose(circ_convolve(a, b), _brute_force(a, b))


def test_circ_convolve_wraps_tail_to_start():
    a = np.zeros(8)
    a[7] = 1.0
    b = np.zeros(8)
    b[2] = 1.0
    out = circ_convolve(a, b)
    assert np.argmax(out) == 1
    np.testing.assert_allclose(out.sum(), 1.0)


def test_circ_convolve_agrees_with_padded_linear_convolution(rng):
    a = rng.normal(size=16)
    b = rng.normal(size=16)
    full = np.convolve(a, b)
    folded = full[:16].copy()
    folded[: full.size - 16] += full[16:]
    np.testing.assert_allclose(circ_convolve(a, b), folded)


def test_delta_at_origin_is_identity(rng):
    a = rng.normal(size=20)
    delta = np.zeros(20)
    delta[0] = 1
    np.testing.assert_allclose(circ_convolve(a, delta), a, atol=1e-12)


def test_circ_convolve_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        circ_convolve(np.zeros(4), np.zeros(5))


def test_circ_shift_to_and_center_to_origin():
    arr = np.zeros(10)
    arr[5] = 1
    assert np.argmax(center_to_origin(arr)) == 0
    assert np.argmax(circ_shift_to(arr, 5, 9)) == 9
    assert np.argmax(circ_shift_to(arr, 5, 1)) == 1


def test_wrap_indices():
    np.testing.assert_array_equal(wrap_indices(2, 5, 10), [2, 3, 4])
    np.testing.assert_array_equal(wrap_indices(8, 2, 10), [8, 9, 0, 1])
    assert wrap_indices(3, 3, 10).size == 0
    np.testing.assert_array_equal(wrap_indices(-2, 1, 10), [8, 9, 0])


def test_bin_grid_excludes_endpoint_and_has_uniform_spacing():
    x = bin_grid(1024, 500.0)
    assert x[0] == 0.0
    np.testing.assert_allclose(np.diff(x), 500.0 / 1024)
    assert x[-1] < 500.0


def test_bin_grid_default_period_is_phase():
    np.testing.assert_allclose(bin_grid(4), [0, 0.25, 0.5, 0.75])


def test_kernel_area_on_bin_grid():
    from tauclean.kernels import get_kernel

    x = bin_grid(2048, 500.0)
    h = get_kernel("thin")(x, 5.0)
    assert np.sum(h) * (500.0 / 2048) == pytest.approx(1.0, rel=0.05)


def test_wrap_indices_accepts_full_range_and_trailing_end():
    np.testing.assert_array_equal(wrap_indices(0, 10, 10), np.arange(10))
    np.testing.assert_array_equal(wrap_indices(7, 10, 10), [7, 8, 9])
