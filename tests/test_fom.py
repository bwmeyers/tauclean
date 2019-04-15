#! /usr/bin/env python
"""
Test fom.py
"""
from tauclean.fom import consistence, positivity, moment, skewness
import numpy as np
np.random.seed(12345)


def test_consistence_zeros():
    nbins = 256

    residuals = np.zeros(nbins)
    nf = consistence(residuals, np.std(residuals), np.mean(residuals), onlims=(100, 150))
    assert not np.isnan(nf)
    assert nf == 50


def test_consistence_random():
    nbins = 256
    residuals = np.random.normal(size=nbins)
    nf = consistence(residuals, np.std(residuals), np.mean(residuals), onlims=(100, 150))
    assert not np.isnan(nf)
    assert nf == 50


def test_consistence_random_plus():
    nbins = 256
    residuals = np.random.normal(size=nbins)
    offrms = np.std(residuals)
    offmean = np.mean(residuals)
    residuals[100:150] += 1000  # make sure on-pulse region is well above 3-sigma threshold
    nf = consistence(residuals, offrms, offmean, onlims=(100, 150))
    assert not np.isnan(nf)
    assert nf == 0


def test_positivity_zeros():
    nbins = 256
    residuals = np.zeros(nbins)
    offrms = np.std(residuals)

    f_r = positivity(residuals, offrms)  # should be NaN as np.all(residuals) == 0 is true
    assert np.isnan(f_r)


def test_positivity_random():
    nbins = 256
    residuals = np.random.normal(size=nbins)
    offrms = np.std(residuals)

    f_r = positivity(residuals, offrms, x=10)
    assert not np.isnan(f_r)
    assert f_r == 0  # there should be no points more negative than 10-sigma, thus f_r should be zero

    f_r = positivity(residuals, offrms, x=5)
    assert not np.isnan(f_r)
    assert f_r == 0  # similarly, for this sample size

    f_r = positivity(residuals, offrms, x=1.5)
    assert not np.isnan(f_r)
    assert f_r > 0



def test_first_moment():
    a1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    a2 = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
    a3 = np.array([10.0, 2.0, 1.0, 2.0, 10.0])
    t1 = np.array([1, 2, 3, 4, 5])
    t2 = np.array([1, 2, 4, 5, 9])

    # first moments (weighted means)
    m11 = np.average(t1, weights=a1)
    m21 = np.average(t1, weights=a2)
    m31 = np.average(t1, weights=a3)
    m12 = np.average(t2, weights=a1)
    m22 = np.average(t2, weights=a2)
    m32 = np.average(t2, weights=a3)

    assert moment(a1, t1, n=1) == m11  # == 3.0
    assert moment(a2, t1, n=1) == m21  # == 3.0
    assert moment(a3, t1, n=1) == m31  # == 3.0
    assert moment(a1, t2, n=1) == m12  # == 4.2
    assert moment(a2, t2, n=1) == m22  # == 4.071428571428571
    assert moment(a3, t2, n=1) == m32  # == 4.72


def test_second_moments():
    a1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    a2 = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
    a3 = np.array([10.0, 2.0, 1.0, 2.0, 10.0])
    t1 = np.array([1, 2, 3, 4, 5])
    t2 = np.array([1, 2, 4, 5, 9])

    # first moments (weighted means)
    m11 = np.average(t1, weights=a1)
    m21 = np.average(t1, weights=a2)
    m31 = np.average(t1, weights=a3)
    m12 = np.average(t2, weights=a1)
    m22 = np.average(t2, weights=a2)
    m32 = np.average(t2, weights=a3)

    # second moments (weighted variances)
    assert moment(a1, t1, n=2) == np.average((t1 - m11) ** 2, weights=a1)  # == 2.0
    assert moment(a2, t1, n=2) == np.average((t1 - m21) ** 2, weights=a2)  # == 1.5714285714285714
    assert moment(a3, t1, n=2) == np.average((t1 - m31) ** 2, weights=a3)  # == 3.36
    assert moment(a1, t2, n=2) == np.average((t2 - m12) ** 2, weights=a1)  # == 7.76
    assert moment(a2, t2, n=2) == np.average((t2 - m22) ** 2, weights=a2)  # == 5.923469387755101
    assert moment(a3, t2, n=2) == np.average((t2 - m32) ** 2, weights=a3)  # == 13.481599999999998


def test_third_moments():
    a1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    a2 = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
    a3 = np.array([10.0, 2.0, 1.0, 2.0, 10.0])
    t1 = np.array([1, 2, 3, 4, 5])
    t2 = np.array([1, 2, 4, 5, 9])

    # first moments (weighted means)
    m11 = np.average(t1, weights=a1)
    m21 = np.average(t1, weights=a2)
    m31 = np.average(t1, weights=a3)
    m12 = np.average(t2, weights=a1)
    m22 = np.average(t2, weights=a2)
    m32 = np.average(t2, weights=a3)

    # third moments (weighted kurtosis)
    assert moment(a1, t1, n=3) == np.average((t1 - m11) ** 3, weights=a1)  # == 0
    assert moment(a2, t1, n=3) == np.average((t1 - m21) ** 3, weights=a2)  # == 0
    assert moment(a3, t1, n=3) == np.average((t1 - m31) ** 3, weights=a3)  # == 0
    assert moment(a1, t2, n=3) == np.average((t2 - m12) ** 3, weights=a1)  # == 13.535999999999994
    assert moment(a2, t2, n=3) == np.average((t2 - m22) ** 3, weights=a2)  # == 11.230320699708459
    assert moment(a3, t2, n=3) == np.average((t2 - m32) ** 3, weights=a3)  # == 9.146496000000006



if __name__ == "__main__":
    test_positivity_random()