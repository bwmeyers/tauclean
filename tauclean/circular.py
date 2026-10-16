"""Helpers for treating pulsar profiles as cyclic (periodic) arrays.

Convention: a circular response/kernel is defined on the same bins as the
profile, with its reference point (the "zero lag") at index 0. Convolving a
profile with such a function therefore does not shift the profile.
"""

from __future__ import annotations

import numpy as np


def bin_grid(nbins: int, period: float = 1.0) -> np.ndarray:
    """Bin start times ``period * i / nbins`` for a cyclic profile.

    The endpoint is excluded: bin ``nbins`` is identical to bin 0, so the
    spacing is exactly ``period / nbins``.
    """
    return period * np.arange(nbins) / nbins


def circ_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circularly convolve two equal-length arrays via FFT."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(
            "circ_convolve requires 1-D arrays of identical length, got "
            f"{a.shape} and {b.shape}"
        )
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=a.size)


def circ_shift_to(arr: np.ndarray, from_idx: int, to_idx: int) -> np.ndarray:
    """Cyclically shift ``arr`` so the sample at ``from_idx`` lands on ``to_idx``."""
    return np.roll(arr, int(to_idx) - int(from_idx))


def center_to_origin(arr: np.ndarray, center_idx: int | None = None) -> np.ndarray:
    """Move a function centred at ``center_idx`` (default ``size // 2``) to index 0."""
    arr = np.asarray(arr)
    if center_idx is None:
        center_idx = arr.size // 2
    return circ_shift_to(arr, center_idx, 0)


def wrap_indices(start: int, end: int, nbins: int) -> np.ndarray:
    """Return bin indices from ``start`` up to (excluding) ``end``, wrapping.

    If ``start < end`` this matches ``np.arange(start, end)``. If
    ``start > end`` the range wraps through the end of the profile. Indices
    are taken modulo ``nbins``; ``start == end`` gives an empty range.
    """
    if nbins <= 0:
        raise ValueError("nbins must be positive")
    start = int(start) % nbins
    end = int(end) % nbins
    if start <= end:
        return np.arange(start, end)
    return np.concatenate([np.arange(start, nbins), np.arange(0, end)])
