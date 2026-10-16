"""Tests for cyclic profile helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tauclean.circular import (
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
