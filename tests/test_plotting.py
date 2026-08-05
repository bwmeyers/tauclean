"""Tests for plotting utilities using CleanResult objects."""

from __future__ import annotations

import glob
import os

import numpy as np

from tauclean.domain.clean_api import clean
from tauclean.plotting import (
    plot_clean_components,
    plot_clean_residuals,
    plot_figures_of_merit,
    plot_reconstruction,
    write_output,
)

np.random.seed(12345)
TEST_DIR = "/".join(os.path.realpath(__file__).split("/")[:-1])

init_data = np.genfromtxt(f"{TEST_DIR}/simulated_profile_tau20ms_thin.txt")


def _build_results():
    taus = [15.0, 20.0, 25.0]
    return [
        clean(
            init_data,
            tau,
            period=500.0,
            pbftype="thin",
            onpulse_estimator="440 900",
            iter_limit=400,
        )
        for tau in taus
    ]


results = _build_results()


def remove_files(pattern):
    flist = glob.glob(f"{os.getcwd()}/{pattern}")
    for f in flist:
        try:
            os.remove(f)
        except OSError:
            raise AssertionError("error when deleting files")


def test_plot_fom():
    if not plot_figures_of_merit(results):
        raise AssertionError()

    remove_files("*.png")


def test_plot_clean_residuals():
    if not plot_clean_residuals(init_data, results, period=500.0):
        raise AssertionError()

    remove_files("*.png")


def test_plot_clean_comps():
    if not plot_clean_components(results, period=500.0):
        raise AssertionError()

    remove_files("*.png")


def test_plot_reconstruction():
    if not plot_reconstruction(results, init_data, period=500.0):
        raise AssertionError()

    remove_files("*.png")


def test_write_output():
    write_output(results)

    # ensure the file was actually written
    try:
        lines = np.loadtxt("clean_components_thin-tau20.txt")
    except FileNotFoundError:
        raise AssertionError()

    # tau = 20 ms is item 2 in the local test result list
    np.testing.assert_array_equal(lines, results[1].cc)

    try:
        lines = np.genfromtxt("reconstruction_thin-tau20.txt")
    except FileNotFoundError:
        raise AssertionError()

    np.testing.assert_array_equal(lines[:, 0], results[1].recon)
    np.testing.assert_array_equal(lines[:, 1], results[1].profile)

    remove_files("*.txt")
