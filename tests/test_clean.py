#! /usr/bin/env python
"""
Test clean.py
"""
from tauclean.clean import keep_cleaning, dm_delay, gaussian, reconstruct, clean
import numpy as np
from scipy.integrate import simps

np.random.seed(12345)


def test_keep_cleaning_true():
    nbins = 256
    ts = np.random.normal(size=nbins)

    ts[50] += 100
    on = ts[40:60]
    off = np.concatenate((ts[:40], ts[60:]))

    k = keep_cleaning(on, off)
    assert k == True


def test_keep_cleaning_false():
    nbins = 256
    ts = np.random.normal(size=nbins)

    on = ts[40:60]
    off = np.concatenate((ts[:40], ts[60:]))

    k = keep_cleaning(on, off)
    assert k == False


def test_dm_delay_zero():
    dm = 0
    flo = 1.4
    fhi = 1.6
    dt = dm_delay(dm, flo, fhi)
    assert dt == 0


def test_dm_delay_100():
    dm = 100
    flo = 1.4
    fhi = 7 * np.sqrt(518601. / 273601.) / 5  # solved so that dt = 100 ms
    dt = dm_delay(dm, flo, fhi)

    np.testing.assert_approx_equal(dt, 100.0)


def test_gaussian_normalised():
    x = np.linspace(0, 10, 1000)
    g = gaussian(x, 5, 0.5)

    assert simps(y=g, x=x) == 1.0



if __name__ == "__main__":
    test_keep_cleaning_true()
    test_keep_cleaning_false()
    test_dm_delay_100()
    test_gaussian_normalised()

