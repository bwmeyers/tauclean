"""Tests for the object-oriented CLEAN API and shared response helpers."""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
from scipy.integrate import simpson as simps

from tauclean.domain.clean_api import clean
from tauclean.domain.clean_run import CleanResult
from tauclean.domain.cleaner import _keep_cleaning
from tauclean.domain.response import (
    dm_delay,
    gaussian,
    get_restoring_function,
    reconstruct,
)

np.random.seed(12345)
TEST_DIR = "/".join(os.path.realpath(__file__).split("/")[:-1])


def run_clean(taus, data, clean_kwargs):
    """Run CLEAN over trial taus using the same multiprocessing pattern as CLI."""
    result_list = []

    def log_results(worker_results):
        result_list.append(worker_results)

    with mp.Pool(processes=1) as pool:
        for tau in taus:
            pool.apply_async(
                clean, (data, tau), clean_kwargs, callback=log_results
            )
        pool.close()
        pool.join()

    return sorted(result_list, key=lambda r: r.tau)


def check_clean_finite(sorted_results):
    if not isinstance(sorted_results, list):
        raise TypeError()
    if not isinstance(sorted_results[0], CleanResult):
        raise TypeError()
    if not np.isfinite(sorted_results[0].init_off_rms):
        raise AssertionError()
    if not np.isfinite(sorted_results[0].off_rms):
        raise AssertionError()
    if not np.isfinite(sorted_results[0].off_mean):
        raise AssertionError()


def test_keep_cleaning_true():
    ts = np.random.normal(size=256)
    ts[50] += 100
    on = ts[40:60]
    off = np.concatenate((ts[:40], ts[60:]))
    np.testing.assert_equal(_keep_cleaning(on, off), True)


def test_keep_cleaning_false():
    ts = np.random.normal(size=256)
    on = ts[40:60]
    off = np.concatenate((ts[:40], ts[60:]))
    np.testing.assert_equal(_keep_cleaning(on, off), False)


def test_dm_delay_zero():
    np.testing.assert_equal(dm_delay(0, 1.4, 1.6), 0)


def test_dm_delay_100():
    fhi = 7 * np.sqrt(518601.0 / 273601.0) / 5
    np.testing.assert_approx_equal(dm_delay(100, 1.4, fhi), 100.0)


def test_gaussian_normalised():
    x = np.linspace(0, 10, 1000)
    g = gaussian(x, 5, 0.5)
    np.testing.assert_array_almost_equal(simps(y=g, x=x), 1)


def test_reconstruct_peak_alignment():
    nbins = 1024
    period = 500.0
    ccs = np.zeros(nbins)
    ccs[400] = 1.0
    rest_func = get_restoring_function(ccs, period, inst_resp_width=3.0)

    recon = reconstruct(ccs, rest_func=rest_func)
    assert abs(int(np.argmax(recon)) - 400) <= 2
    np.testing.assert_approx_equal(recon.max(), 1.0)


def test_clean_invalid_kernel():
    data = np.genfromtxt(f"{TEST_DIR}/simulated_profile_tau20ms_thin.txt")
    try:
        clean(data, 20, period=500, pbftype="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError()


def test_clean_iteration_limit():
    data = np.genfromtxt(f"{TEST_DIR}/simulated_profile_tau20ms_thin.txt")
    sorted_results = run_clean(
        [20.0],
        data,
        {
            "period": 500,
            "gain": 0.05,
            "pbftype": "thin",
            "onpulse_estimator": "440 900",
            "iter_limit": 1,
        },
    )

    if not isinstance(sorted_results, list):
        raise TypeError()
    if sorted_results[0].niter != 1:
        raise AssertionError()


def test_clean_thin():
    data = np.genfromtxt(f"{TEST_DIR}/simulated_profile_tau20ms_thin.txt")
    sorted_results = run_clean(
        [20.0],
        data,
        {
            "period": 500,
            "gain": 0.05,
            "pbftype": "thin",
            "onpulse_estimator": "440 900",
            "iter_limit": 400,
        },
    )
    check_clean_finite(sorted_results)
    if not np.isfinite(sorted_results[0].recon.max()):
        raise AssertionError()


def test_clean_thick():
    data = np.genfromtxt(f"{TEST_DIR}/simulated_profile_tau1ms_thick.txt")
    sorted_results = run_clean(
        [1.0],
        data,
        {
            "period": 500,
            "gain": 0.05,
            "pbftype": "thick",
            "onpulse_estimator": "128 700",
            "iter_limit": 400,
        },
    )
    check_clean_finite(sorted_results)
    if sorted_results[0].pbftype != "thick":
        raise AssertionError()


def test_clean_uniform():
    data = np.genfromtxt(f"{TEST_DIR}/simulated_profile_tau3ms_uniform.txt")
    sorted_results = run_clean(
        [3.0],
        data,
        {
            "period": 500,
            "gain": 0.05,
            "pbftype": "uniform",
            "onpulse_estimator": "128 700",
            "iter_limit": 400,
        },
    )
    check_clean_finite(sorted_results)
    if sorted_results[0].ncc <= 0:
        raise AssertionError()
