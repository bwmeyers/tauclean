#! /usr/bin/env python
"""
Copyright 2019 Bradley Meyers
Licensed under the Academic Free License version 3.0

Test fom.py
"""

from tauclean.fom import consistence, positivity, skewness
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

    # With 256 elements, we would expect << 1 sample to be greater than 10-sigma
    nf = consistence(residuals, np.std(residuals), np.mean(residuals), onlims=(100, 150), thresh=10)
    assert not np.isnan(nf)
    assert nf == 50

    # With 256 elements, we would expect ~1 sample to be greater than 3-sigma
    nf = consistence(residuals, np.std(residuals), np.mean(residuals), onlims=(100, 150), thresh=3)
    assert not np.isnan(nf)
    assert 49 <= nf <= 50


def test_consistence_random_offset():
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


def test_skewness_ones():
    nbins = 256
    period = 10.0  # ms
    cc_amps = np.ones(nbins)

    # gamma should be a very small, negative number in this case
    gamma = skewness(cc_amps, period=period)
    assert -1.0e-10 < gamma < 0


def test_skewness_single():
    nbins = 256
    period = 10.0
    cc_amps = np.zeros(nbins)
    cc_amps[nbins//2] = 50

    gamma = skewness(cc_amps, period=period)
    assert gamma == 0  # a completely symmetric deconvolution means gamma should be identically 0


def test_skewness_cluster_symmetric():
    nbins = 256
    period = 10.0
    cc_amps = np.zeros(nbins)
    cc_amps[124:133] += 10
    cc_amps[128] = 50

    # gamma should be a number very close to zero
    gamma = skewness(cc_amps, period=period)
    assert abs(gamma) < 1.0e-8


def test_skewness_cluster_right_skew():
    nbins = 256
    period = 10.0
    cc_amps = np.zeros(nbins)

    # dominate with one component, but have a significant number of non-zero components to the right
    cc_amps[124:133] += 10
    cc_amps[125] = 200

    # gamma should be a positive number greater than 1
    gamma = skewness(cc_amps, period=period)
    assert gamma > 1


def test_skewness_cluster_left_skew():
    nbins = 256
    period = 10.0
    cc_amps = np.zeros(nbins)

    # dominate with one component, but have a significant number of non-zero components to the left
    cc_amps[124:133] += 10
    cc_amps[132] = 200

    # gamma should be a negative number less than -1
    gamma = skewness(cc_amps, period=period)
    assert gamma < -1



if __name__ == "__main__":
    test_consistence_random_offset()