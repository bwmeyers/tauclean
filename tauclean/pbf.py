#!/usr/bin/env python3

import numpy as np
from scipy.integrate import simpson as simps


def _smooth(
    fn1: np.ndarray,
    fn2: np.ndarray,
    scale: float,
    x_array: np.ndarray,
    x_offset: float = 0,
    peak_idx: int = 0,
) -> np.ndarray:
    """Join functions without discontinuities using a smoothing transition
    function.

    Uses tanh to accomplish smoothing, with a smoothing factor k. As k
    decreases, the smoothing is more pronounced.

    See https://math.stackexchange.com/a/45335

    :param fn1: first function to combine
    :type fn1: np.ndarray
    :param fn2: second function to combine
    :type fn2: np.ndarray
    :param scale: smoothing scale parameter
    :type scale: float
    :param x_array: independent variable array
    :type x_array: np.ndarray
    :param x_offset: offset position for smoothing transition
    :type x_offset: float
    :param peak_idx: index of the peak
    :type peak_idx: int
    :return: smoothed and combined kernel
    :rtype: np.ndarray
    """

    # Empirically, k = 0.09 smooths appropriately for tau = 30 ms, and
    # should decrease as tau increases
    k = (0.09 / scale) * 30

    # When x < x_offset, this factor tends to 0, whereas when x > x_offset
    # it tends towards 1
    b = 0.5 * (1 + np.tanh(k * (x_array - x_offset)))

    # The final combination of these functions produces the smoothed and
    # combined kernel
    smoothed = fn1 + b * (fn2 - fn1)

    # Enforce that the rise-time mimics h1
    smoothed[:peak_idx] = (
        fn1[:peak_idx] / np.max(fn1[:peak_idx]) * smoothed[peak_idx]
    )

    return smoothed


def gaussian(x, mu: float = 0, sigma: float = 1) -> np.ndarray:
    """Calculate a Gaussian shape over x.

    :param x: independent variable
    :type x: np.ndarray
    :param mu: the mean (position) of the Gaussian
    :type mu: float
    :param sigma: the standard deviation (width) of the Gaussian
    :type sigma: float
    :return: a numerically evaluated Gaussian
    :rtype: np.ndarray
    """

    amp = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    g = amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    return g


def thin(x: np.ndarray, tau: float, x0: float = 0) -> np.ndarray:
    """Classical, square-law structure media thin screen approximation for
    pulse broadening.

    See e.g. Cordes & Rickett (1998) and Lambert & Rickett (1999).

    :param x: time over which to evaluate the PBF
    :type x: np.ndarray
    :param tau: pulse broadening time scale
    :type tau: float
    :param x0: where the PBF turns on (in range of x)
    :type x0: float
    :return: evaluated thin screen PBF
    :rtype: np.ndarray
    """

    t = x - x0
    h = (1 / tau) * np.exp(-t / tau)  # normalised

    # Turn on a unit step function at the given x0 offset, and turn nans into 0
    h[np.where((x < x0) | np.isnan(h))] = 0
    h = h / simps(x=t, y=h)  # enforce normalisation of PBF

    return h


def thick(x: np.ndarray, tau: float, x0: float = 0) -> np.ndarray:
    """Thick screen pulse broadening function as presented in
    Williamson (1972).

    :param x: time over which to evaluate the PBF
    :type x: np.ndarray
    :param tau: pulse broadening time scale
    :type tau: float
    :param x0: where the PBF turns on (in range of x)
    :type x0: float
    :return: evaluated thick screen PBF
    :rtype: np.ndarray
    """

    t = x - x0

    # ignore divide by zero and consequent invalid operation warnings due to
    # very negative numbers (caused by providing large offsets through x0,
    # size you can end up dividing by 0 or evaluating very large negative
    # exponentials)
    old_settings = np.seterr(divide="ignore", invalid="ignore")

    h = np.sqrt((np.pi * tau) / (4 * t**3)) * np.exp(
        -tau * np.pi**2 / (16 * t)
    )  # normalised

    np.seterr(**old_settings)  # restore old behaviour

    # nominally, h1 is not defined at t <= 0
    h[np.where(np.isnan(h))] = 0
    h = h / simps(x=t, y=h)  # enforce normalisation of PBF

    return h


def thick_exp(x: np.ndarray, tau: float, x0: float = 0) -> np.ndarray:
    """Thick screen PBF from Williamson (1972) with exponential delay shape.

    Modified to exhibit the classical exponential delay shape, ensuring that at
    t -> infinity, the PBF vanishes.

    See p68 of Williamson 1972, just after Figure 9, for discussion on this
    kind of modification.

    :param x: time over which to evaluate the PBF
    :type x: np.ndarray
    :param tau: pulse broadening time scale
    :type tau: float
    :param x0: where the PBF turns on (in range of x)
    :type x0: float
    :return: evaluated thick screen with exponential decay PBF
    :rtype: np.ndarray
    """

    t = x - x0
    expdelay = np.log(4 / np.pi)

    # ignore divide by zero and consequent invalid operation warnings due to
    # very negative numbers (caused by providing large offsets through x0,
    # size you can end up dividing by 0 or evaluating very large negative
    # exponentials)
    old_settings = np.seterr(divide="ignore", invalid="ignore")

    h1 = np.sqrt((np.pi * tau) / (4 * t**3)) * np.exp(
        -tau * np.pi**2 / (16 * t)
    )  # normalised
    h1[np.where(np.isnan(h1))] = 0  # nominally, h1 is not defined at t <= 0

    np.seterr(**old_settings)  # restore old behaviour

    # now figure out the peak of the PBF and begin the normal exponential
    # decay after the appropriate delay
    pbfmax = x0 + np.pi**2 * tau / 24  # in ms
    pbfmax_idx = np.where(t >= pbfmax)[0][0]
    decay_start_idx = np.where(t >= pbfmax + expdelay * tau)[0][0]
    decay_amp = h1[decay_start_idx]
    decay_time = t[decay_start_idx]

    h2 = np.exp(-t / tau)
    h2 = (h2 / h2[decay_start_idx]) * decay_amp
    h2[np.where(np.isnan(h2))] = 0

    h = _smooth(h1, h2, tau, t, x_offset=decay_time, peak_idx=pbfmax_idx)

    h[np.where((x < x0) | np.isnan(h))] = 0
    h = h / simps(x=t, y=h)  # enforce normalisation of PBF

    return h


def uniform(x: np.ndarray, tau: float, x0: float = 0) -> np.ndarray:
    """Uniform media pulse broadening function as presented in
    Williamson (1972).

    :param x: time over which to evaluate the PBF
    :type x: np.ndarray
    :param tau: pulse broadening time scale
    :type tau: float
    :param x0: where the PBF turns on (in range of x)
    :type x0: float
    :return: evaluated PBF for a uniform scattering medium
    :rtype: np.ndarray
    """

    t = x - x0

    # ignore divide by zero and consequent invalid operation warnings due to
    # very negative numbers (caused by providing large offsets through x0,
    # size you can end up dividing by 0 or evaluating very large negative
    # exponentials)
    old_settings = np.seterr(divide="ignore", invalid="ignore")

    h = np.sqrt((np.pi**5 * tau**3) / (8 * t**5)) * np.exp(
        -tau * np.pi**2 / (4 * t)
    )  # normalised

    np.seterr(**old_settings)  # restore old behaviour

    # nominally, h1 is not defined at t <= 0
    h[np.where(np.isnan(h))] = 0
    h = h / simps(x=t, y=h)  # enforce normalisation of PBF

    return h


def uniform_exp(x: np.ndarray, tau: float, x0: float = 0) -> np.ndarray:
    """Uniform media PBF from Williamson (1972) with exponential delay shape.

    Modified to exhibit the classical exponential delay shape, ensuring that at
    t -> infinity, the PBF vanishes.

    See p68 of Williamson 1972, just after Figure 9, for discussion on this
    kind of modification.

    :param x: time over which to evaluate the PBF
    :type x: np.ndarray
    :param tau: pulse broadening time scale
    :type tau: float
    :param x0: where the PBF turns on (in range of x)
    :type x0: float
    :return: evaluated PBF for a uniform scattering medium with exponential
        decay
    :rtype: np.ndarray
    """

    t = x - x0
    expdelay = np.log(2)

    # ignore divide by zero and consequent invalid operation warnings due to
    # very negative numbers (caused by providing large offsets through x0,
    # size you can end up dividing by 0 or evaluating very large negative
    # exponentials)
    old_settings = np.seterr(divide="ignore", invalid="ignore")

    h1 = np.sqrt((np.pi**5 * tau**3) / (8 * t**5)) * np.exp(
        -tau * np.pi**2 / (4 * t)
    )  # normalised
    h1[np.where(np.isnan(h1))] = 0  # nominally, h1 is not defined at t <= 0

    np.seterr(**old_settings)  # restore old behaviour

    # now figure out the peak of the PBF and begin the normal exponential decay after the appropriate delay
    pbfmax = x0 + np.pi**2 * tau / 10  # in ms
    pbfmax_idx = np.where(t >= pbfmax)[0][0]
    decay_start_idx = np.where(t >= pbfmax + expdelay * tau)[0][0]
    decay_amp = h1[decay_start_idx]
    decay_time = t[decay_start_idx]

    h2 = np.exp(-t / tau)
    h2 = (h2 / h2[decay_start_idx]) * decay_amp
    h2[np.where(np.isnan(h2))] = 0

    h = _smooth(h1, h2, tau, t, x_offset=decay_time, peak_idx=pbfmax_idx)

    h[np.where(np.isnan(h))] = 0
    h = h / simps(x=t, y=h)  # enforce normalisation of PBF

    return h
