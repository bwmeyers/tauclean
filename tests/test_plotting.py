#! /usr/bin/env python
"""
Copyright 2019 Bradley Meyers
Licensed under the Academic Free License version 3.0

Test plotting.py
"""

import os
import pickle
import glob

import numpy as np

from tauclean.plotting import (plot_figures_of_merit, plot_clean_components, plot_clean_residuals,
                               plot_reconstruction, write_output)

np.random.seed(12345)
TEST_DIR = '/'.join(os.path.realpath(__file__).split('/')[0:-1])

init_data = np.genfromtxt("{TEST_DIR}/simulated_profile_tau20ms_thin.txt".format(TEST_DIR=TEST_DIR))
results = pickle.load(open("{TEST_DIR}/test_sample.p".format(TEST_DIR=TEST_DIR), "rb"))
results[-1]["fr"] = 100  # fake one fo the FoM to ensure that testing covers all code


def remove_files(pattern):
    flist = glob.glob("{0}/{1}".format(os.getcwd(), pattern))
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

    # tau = 20ms is item 2 in the results list
    np.testing.assert_array_equal(lines, results[2]['cc'])

    try:
        lines = np.genfromtxt("reconstruction_thin-tau20.txt")
    except FileNotFoundError:
        raise AssertionError()

    np.testing.assert_array_equal(lines[:, 0], results[2]['recon'])
    np.testing.assert_array_equal(lines[:, 1], results[2]['profile'])

    remove_files("*.txt")
