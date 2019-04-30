#! /usr/bin/env python
"""
Copyright 2019 Bradley Meyers
Licensed under the Academic Free License version 3.0

Test pbf.py
"""

import numpy as np
from scipy.integrate import simps

from tauclean.pbf import thin, thick, uniform

np.random.seed(12345)


def test_thin_normalised():
    nbins = 1024
    period = 500.0
    tau = 10
    x = period * np.linspace(0, 1, nbins)
    x0 = 100
    h = thin(x, tau, x0=x0)

    # seeing as integration can sometimes cause rounding errors, use approx-equal method to 7 decimal places
    np.testing.assert_almost_equal(simps(x=x, y=h), 1)


def test_thick_normalised():
    nbins = 1024
    period = 500.0
    tau = 3
    x = period * np.linspace(0, 1, nbins)
    x0 = 50
    h = thick(x, tau, x0=x0)

    # seeing as integration can sometimes cause rounding errors, use approx-equal method to 7 decimal places
    np.testing.assert_almost_equal(simps(x=x, y=h), 1)


def test_uniform_normalised():
    nbins = 1024
    period = 500.0
    tau = 3
    x = period * np.linspace(0, 1, nbins)
    x0 = 50
    h = uniform(x, tau, x0=x0)

    # seeing as integration can sometimes cause rounding errors, use approx-equal method to 7 decimal places
    np.testing.assert_almost_equal(simps(x=x, y=h), 1)
