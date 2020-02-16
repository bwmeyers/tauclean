#! /usr/bin/env python
"""
Copyright 2019 Bradley Meyers
Licensed under the Academic Free License version 3.0

Test clean.py
"""

import multiprocessing as mp
import os

import numpy as np
from scipy.integrate import simps

from tauclean.clean import keep_cleaning, dm_delay, gaussian, reconstruct, clean

np.random.seed(12345)
TEST_DIR = '/'.join(os.path.realpath(__file__).split('/')[0:-1])


def run_clean(taus, data, clean_kwargs):
    """
    Utility function that just runs the clean algorithm using the multiprocessing module, as done in the `tauclean`
    script
    """
    result_list = []

    def log_results(worker_results):
        result_list.append(worker_results)

    pool = mp.Pool(processes=1)
    for t in taus:
        results = pool.apply_async(clean, (data, t), clean_kwargs, callback=log_results)

    pool.close()
    pool.join()

    sorted_results = sorted(result_list, key=lambda r: r['tau'])

    return sorted_results


def check_clean_finite(sorted_results):
    """Utility function to test whether the results (specifically the off-pulse rms before and after) from clean are
    finite, and that the correct object was returned by the function
    """
    if not isinstance(sorted_results, list):
        raise AssertionError()
    if not np.isfinite(sorted_results[0]["init_rms"]):
        raise AssertionError()
    if not np.isfinite(sorted_results[0]["off_rms"]):
        raise AssertionError()
    if not np.isfinite(sorted_results[0]["off_mean"]):
        raise AssertionError()


def test_keep_cleaning_true():
    nbins = 256
    ts = np.random.normal(size=nbins)

    ts[50] += 100
    on = ts[40:60]
    off = np.concatenate((ts[:40], ts[60:]))

    k = keep_cleaning(on, off)

    np.testing.assert_equal(k, True)


def test_keep_cleaning_false():
    nbins = 256
    ts = np.random.normal(size=nbins)

    on = ts[40:60]
    off = np.concatenate((ts[:40], ts[60:]))

    k = keep_cleaning(on, off)

    np.testing.assert_equal(k, False)


def test_dm_delay_zero():
    dm = 0
    flo = 1.4
    fhi = 1.6
    dt = dm_delay(dm, flo, fhi)

    np.testing.assert_equal(dt, 0)


def test_dm_delay_100():
    dm = 100
    flo = 1.4
    fhi = 7 * np.sqrt(518601. / 273601.) / 5  # solved so that dt = 100 ms
    dt = dm_delay(dm, flo, fhi)

    np.testing.assert_approx_equal(dt, 100.0)


def test_gaussian_normalised():
    x = np.linspace(0, 10, 1000)
    g = gaussian(x, 5, 0.5)

    np.testing.assert_array_almost_equal(simps(y=g, x=x), 1)


def test_reconstruction_simple():
    nbins = 1024
    period = 500.0
    rest_width = 0.1

    x = np.linspace(0, 1, nbins) * period
    profile = np.exp(-(x - x[500]) ** 2 / (2 * rest_width ** 2))
    profile = profile / profile.max()

    # test when clean component is aligned with peak of the profile
    ccs = np.zeros_like(x)
    ccs[500] = 1.0

    recon = reconstruct(profile, ccs, period=period, rest_width=rest_width)
    diff = recon - profile

    # check that the difference is 0 to within 4 decimal places (0.1%)
    np.testing.assert_array_almost_equal(diff, np.zeros_like(profile), decimal=4)

    # test when the clean component is offset from the peak of the profile
    ccs = np.zeros_like(x)
    ccs[400] = 1.0

    recon = reconstruct(profile, ccs, period=period, rest_width=rest_width)
    diff = recon - profile

    # check that the difference is 0 to within 4 decimal places (0.1%)
    np.testing.assert_array_almost_equal(diff, np.zeros_like(profile), decimal=4)


def test_reconstruction_multiple():
    nbins = 1024
    period = 500.0
    rest_width = 0.1

    x = np.linspace(0, 1, nbins) * period
    profile = np.exp(-(x - x[500]) ** 2 / (2 * rest_width ** 2))
    profile += np.exp(-(x - x[650]) ** 2 / (2 * rest_width ** 2))
    profile = profile / profile.max()

    # test when clean component is aligned with peak of the profile
    ccs = np.zeros_like(x)
    ccs[500] = 1.0
    ccs[650] = 1.0

    recon = reconstruct(profile, ccs, period=period, rest_width=rest_width)
    diff = recon - profile

    # check that the difference is 0 to within 4 decimal places (0.1%)
    np.testing.assert_array_almost_equal(diff, np.zeros_like(profile), decimal=4)

    # test when the clean component is offset from the peak of the profile
    ccs = np.zeros_like(x)
    ccs[450] = 1.0
    ccs[600] = 1.0

    recon = reconstruct(profile, ccs, period=period, rest_width=rest_width)
    diff = recon - profile

    # check that the difference is 0 to within 4 decimal places (0.1%)
    np.testing.assert_array_almost_equal(diff, np.zeros_like(profile), decimal=4)


# TODO: should nominally also test when profile width is NOT equal to the restoring function width, but for now this
#       at least tests that the convolution and offset correction is working


def test_clean_invalid_pbf():
    # simulate -n 1024 -m 500 -a 12 -w 10 -k thin -t 20.0 -p 500
    data = np.genfromtxt("{TEST_DIR}/simulated_profile_tau20ms_thin.txt".format(TEST_DIR=TEST_DIR))

    ret_val = clean(data, 20, [], pbftype="unknown")

    np.testing.assert_equal([ret_val], [None])


def test_clean_iteration_limit():
    # simulate -n 1024 -m 500 -a 12 -w 10 -k thin -t 20.0 -p 500
    data = np.genfromtxt("{TEST_DIR}/simulated_profile_tau20ms_thin.txt".format(TEST_DIR=TEST_DIR))
    taus = [20.0]
    ilim = 1000

    clean_kwargs = dict(period=500, gain=0.05, pbftype="thin", on_start=440, on_end=900, rest_width=500 / len(data),
                        iter_limit=ilim)

    sorted_results = run_clean(taus, data, clean_kwargs)
    print(sorted_results)

    if not isinstance(sorted_results, list):
        raise AssertionError()
    if not sorted_results[0]["niter"] == ilim:
        raise AssertionError()


def test_clean_thin():
    # simulate -n 1024 -m 500 -a 12 -w 10 -k thin -t 20.0 -p 500
    data = np.genfromtxt("{TEST_DIR}/simulated_profile_tau20ms_thin.txt".format(TEST_DIR=TEST_DIR))
    intrinsic = np.genfromtxt("{TEST_DIR}/simulated_intrinsic_tau20ms_thin.txt".format(TEST_DIR=TEST_DIR))
    taus = [20.0]

    clean_kwargs = dict(period=500, gain=0.05, pbftype="thin", on_start=440, on_end=900, rest_width=500 / len(data))

    sorted_results = run_clean(taus, data, clean_kwargs)

    check_clean_finite(sorted_results)
    # allow 5% error in amplitude of reconstruction
    if not (abs(intrinsic.max() - sorted_results[0]["recon"].max()) < 0.05 * intrinsic.max()):
        raise AssertionError()


def test_clean_thick():
    # simulate -n 1024 -m 200 -a 12 -w 10 -k thick -t 1.0 -p 500
    data = np.genfromtxt("{TEST_DIR}/simulated_profile_tau1ms_thick.txt".format(TEST_DIR=TEST_DIR))
    intrinsic = np.genfromtxt("{TEST_DIR}/simulated_intrinsic_tau1ms_thick.txt".format(TEST_DIR=TEST_DIR))
    taus = [1.0]

    clean_kwargs = dict(period=500, gain=0.05, pbftype="thick", on_start=128, on_end=700, rest_width=500 / len(data))

    sorted_results = run_clean(taus, data, clean_kwargs)

    check_clean_finite(sorted_results)
    # allow 5% error in amplitude of reconstruction
    if not (abs(intrinsic.max() - sorted_results[0]["recon"].max()) < 0.05 * intrinsic.max()):
        raise AssertionError()


def test_clean_uniform():
    # simulate -n 1024 -m 200 -a 12 -w 10 -k uniform -t 3.0 -p 500
    data = np.genfromtxt("{TEST_DIR}/simulated_profile_tau3ms_uniform.txt".format(TEST_DIR=TEST_DIR))
    # intrinsic = np.genfromtxt("{TEST_DIR}/simulated_intrinsic_tau3ms_uniform.txt".format(TEST_DIR=TEST_DIR))
    taus = [3.0]

    clean_kwargs = dict(period=500, gain=0.05, pbftype="uniform", on_start=128, on_end=700, rest_width=500 / len(data))

    sorted_results = run_clean(taus, data, clean_kwargs)

    check_clean_finite(sorted_results)
    # allow 5% error in amplitude of reconstruction
    # TODO currently the "uniform" PBF produces weird amplitudes in the reconstruction that I don't full understand...
    # print(intrinsic.max(), sorted_results[0]["recon"].max(), abs(intrinsic.max() - sorted_results[0]["recon"].max()))
    # if not (abs(intrinsic.max() - sorted_results[0]["recon"].max()) < 0.05 * intrinsic.max()):
    #    raise AssertionError()
