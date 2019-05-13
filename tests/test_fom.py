#! /usr/bin/env python
"""
Copyright 2019 Bradley Meyers
Licensed under the Academic Free License version 3.0

Test fom.py
"""

import numpy as np
import os
import pickle

from tauclean.fom import consistence, positivity, skewness, get_error

np.random.seed(12345)

TEST_DIR = '/'.join(os.path.realpath(__file__).split('/')[0:-1])

results = pickle.load(open("{TEST_DIR}/test_sample.p".format(TEST_DIR=TEST_DIR), "rb"))


def test_consistence_zeros():
    nbins = 256

    residuals = np.zeros(nbins)
    nf = consistence(residuals, np.std(residuals), np.mean(residuals), onlims=(100, 150))
    if np.isnan(nf):
        raise AssertionError()
    if not nf == 50:
        raise AssertionError()


def test_consistence_random():
    nbins = 256
    residuals = np.random.normal(size=nbins)

    # With 256 elements, we would expect << 1 sample to be greater than 10-sigma
    nf = consistence(residuals, np.std(residuals), np.mean(residuals), onlims=(100, 150), thresh=10)
    if np.isnan(nf):
        raise AssertionError()
    if not nf == 50:
        raise AssertionError()

    # With 256 elements, we would expect ~1 sample to be greater than 3-sigma
    nf = consistence(residuals, np.std(residuals), np.mean(residuals), onlims=(100, 150), thresh=3)
    if np.isnan(nf):
        raise AssertionError()
    if not (49 <= nf <= 50):
        raise AssertionError()


def test_consistence_random_offset():
    nbins = 256
    residuals = np.random.normal(size=nbins)
    offrms = np.std(residuals)
    offmean = np.mean(residuals)
    residuals[100:150] += 1000  # make sure on-pulse region is well above 3-sigma threshold
    nf = consistence(residuals, offrms, offmean, onlims=(100, 150))

    if np.isnan(nf):
        raise AssertionError()
    if not nf == 0:
        raise AssertionError()


def test_positivity_zeros():
    nbins = 256
    residuals = np.zeros(nbins)
    offrms = np.std(residuals)

    f_r = positivity(residuals, offrms)  # should be NaN as np.all(residuals) == 0 is true

    if not np.isnan(f_r):
        raise AssertionError()


def test_positivity_random():
    nbins = 256
    residuals = np.random.normal(size=nbins)
    offrms = np.std(residuals)

    f_r = positivity(residuals, offrms, x=10)
    if np.isnan(f_r):
        raise AssertionError()
    if not f_r == 0:
        # there should be no points more negative than 10-sigma, thus f_r should be zero
        raise AssertionError()

    f_r = positivity(residuals, offrms, x=5)
    if np.isnan(f_r):
        raise AssertionError()
    if not f_r == 0:
        # similarly, for this sample size
        raise AssertionError()

    f_r = positivity(residuals, offrms, x=1.5)
    if np.isnan(f_r):
        raise AssertionError()
    if not f_r > 0:
        raise AssertionError()


def test_skewness_ones():
    nbins = 256
    period = 10.0  # ms
    cc_amps = np.ones(nbins)

    # gamma should be a very small, negative number in this case
    gamma = skewness(cc_amps, period=period)

    if not -1.0e-10 < gamma < 0:
        raise AssertionError()


def test_skewness_single():
    nbins = 256
    period = 10.0
    cc_amps = np.zeros(nbins)
    cc_amps[nbins//2] = 50

    gamma = skewness(cc_amps, period=period)

    # a completely symmetric deconvolution means gamma should be identically 0
    if not gamma == 0:
        raise AssertionError()


def test_skewness_cluster_symmetric():
    nbins = 256
    period = 10.0
    cc_amps = np.zeros(nbins)
    cc_amps[124:133] += 10
    cc_amps[128] = 50

    # gamma should be a number very close to zero
    gamma = skewness(cc_amps, period=period)

    if not abs(gamma) < 1.0e-8:
        raise AssertionError()


def test_skewness_cluster_right_skew():
    nbins = 256
    period = 10.0
    cc_amps = np.zeros(nbins)

    # dominate with one component, but have a significant number of non-zero components to the right
    cc_amps[124:133] += 10
    cc_amps[125] = 200

    # gamma should be a positive number greater than 1
    gamma = skewness(cc_amps, period=period)

    if not gamma > 1:
        raise AssertionError()


def test_skewness_cluster_left_skew():
    nbins = 256
    period = 10.0
    cc_amps = np.zeros(nbins)

    # dominate with one component, but have a significant number of non-zero components to the left
    cc_amps[124:133] += 10
    cc_amps[132] = 200

    # gamma should be a negative number less than -1
    gamma = skewness(cc_amps, period=period)

    if not gamma < -1:
        raise AssertionError()
