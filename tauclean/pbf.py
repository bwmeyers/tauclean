#!/usr/bin/env python3
"""
########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
"""

import numpy as np
from scipy.integrate import simpson as simps


def _smooth(fn1, fn2, scale, x_array, x_offset=0, peak_idx=0):
    """To join the functions without discontinuities, we need to use a smoothing transition function.
    Here, we use tanh to accomplish this, with a smoothing factor k, where in this case, as k decreases,
    the smoothing is more pronounced.

    See https://math.stackexchange.com/a/45335
    """

    # Empirically, k = 0.09 smooths appropriately for tau = 30 ms, and should decrease as tau increases
    k = (0.09 / scale) * 30

    # When x < x_offset, this factor tends to 0, whereas when x > x_offset it tends to 1
    b = 0.5 * (1 + np.tanh(k * (x_array - x_offset)))

    # The final combination of these functions produces the smoothed, combined kernel
    smoothed = fn1 + b * (fn2 - fn1)

    # Enforce that the rise-time mimics h1
    smoothed[:peak_idx] = fn1[:peak_idx] / np.max(fn1[:peak_idx]) * smoothed[peak_idx]

    return smoothed


def thin(x, tau, x0=0):
    """The classical, square-law structure media thin screen approximation for a pulse broadening function.
    See e.g. Cordes & Rickett (1998) and Lambert & Rickett (1999).

    :param x: time over which to evaluate the PBF [array-like]
    :param tau: pulse broadening time scale [float]
    :param x0: where the PBF turns on [float, in range of x]
    :return: evaluated thin screen PBF [array-like]
    """

    t = x - x0
    h = (1 / tau) * np.exp(-t / tau)  # normalised

    # Turn on a unit step function at the given x0 offset, and turn nans into 0
    h[np.where((x < x0) | np.isnan(h))] = 0
    h = h / simps(x=t, y=h)  # enforce normalisation of PBF

    return h


def thick(x, tau, x0=0):
    """The thick screen pulse broadening function as presented in Williamson (1972).

    :param x: time over which to evaluate the PBF [array-like]
    :param tau: pulse broadening time scale [float]
    :param x0: where the PBF turns on [float, in range of x]
    :return: evaluated thick screen PBF [array-like]
    """

    t = x - x0

    # ignore divide by zero and consequent invalid operation warnings due to very negative numbers (caused by providing
    # large offsets through x0, size you can end up dividing by 0 or evaluating very large negative exponentials)
    old_settings = np.seterr(divide="ignore", invalid="ignore")

    h = np.sqrt((np.pi * tau) / (4 * t**3)) * np.exp(
        -tau * np.pi**2 / (16 * t)
    )  # normalised

    np.seterr(**old_settings)  # restore old behaviour

    # nominally, h1 is not defined at t <= 0
    h[np.where(np.isnan(h))] = 0
    h = h / simps(x=t, y=h)  # enforce normalisation of PBF

    return h


def thick_exp(x, tau, x0=0):
    """The thick screen pulse broadening function as presented in Williamson (1972), modified to exhibit the classical
    exponential delay shape (see p68 of Williamson 1972, just after Figure 9). This ensures that at t -> infinity,
    the PBF vanishes.

    :param x: time over which to evaluate the PBF [array-like]
    :param tau: pulse broadening time scale [float]
    :param x0: where the PBF turns on [float, in range of x]
    :return: evaluated thick screen with exp. decay PBF [array-like]
    """

    t = x - x0
    expdelay = np.log(4 / np.pi)

    # ignore divide by zero and consequent invalid operation warnings due to very negative numbers (caused by providing
    # large offsets through x0, size you can end up dividing by 0 or evaluating very large negative exponentials)
    old_settings = np.seterr(divide="ignore", invalid="ignore")

    h1 = np.sqrt((np.pi * tau) / (4 * t**3)) * np.exp(
        -tau * np.pi**2 / (16 * t)
    )  # normalised
    h1[np.where(np.isnan(h1))] = 0  # nominally, h1 is not defined at t <= 0

    np.seterr(**old_settings)  # restore old behaviour

    # now figure out the peak of the PBF and begin the normal exponential decay after the appropriate delay
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


def uniform(x, tau, x0=0):
    """The uniform media pulse broadening function as presented in Williamson (1972).

    :param x: time over which to evaluate the PBF [array-like]
    :param tau: pulse broadening time scale [float]
    :param x0: where the PBF turns on [float, in range of x]
    :return: evaluated PBF for a uniform scattering medium [array-like]
    """

    t = x - x0

    # ignore divide by zero and consequent invalid operation warnings due to very negative numbers (caused by providing
    # large offsets through x0, size you can end up dividing by 0 or evaluating very large negative exponentials)
    old_settings = np.seterr(divide="ignore", invalid="ignore")

    h = np.sqrt((np.pi**5 * tau**3) / (8 * t**5)) * np.exp(
        -tau * np.pi**2 / (4 * t)
    )  # normalised

    np.seterr(**old_settings)  # restore old behaviour

    # nominally, h1 is not defined at t <= 0
    h[np.where(np.isnan(h))] = 0
    h = h / simps(x=t, y=h)  # enforce normalisation of PBF

    return h


def uniform_exp(x, tau, x0=0):
    """The uniform media pulse broadening function as presented in Williamson (1972), modified to exhibit the classical
    exponential delay shape (see p68 of Williamson 1972, just after Figure 9). This ensures that at t -> infinity,
    the PBF vanishes.

    :param x: time over which to evaluate the PBF [array-like]
    :param tau: pulse broadening time scale [float]
    :param x0: where the PBF turns on [float, in range of x]
    :return: evaluated PBF for a uniform scattering medium with exp. decay [array-like]
    """

    t = x - x0
    expdelay = np.log(2)

    # ignore divide by zero and consequent invalid operation warnings due to very negative numbers (caused by providing
    # large offsets through x0, size you can end up dividing by 0 or evaluating very large negative exponentials)
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
