#! /usr/bin/env python
"""
Copyright 2019 Bradley Meyers
Licensed under the Academic Free License version 3.0

Test plotting.py
"""

import numpy as np
import os
import pickle
from tauclean.plotting import plot_figures_of_merit, plot_clean_components, plot_clean_residuals, plot_reconstruction

np.random.seed(12345)
TEST_DIR = '/'.join(os.path.realpath(__file__).split('/')[0:-1])

init_data = np.genfromtxt("{TEST_DIR}/simulated_profile_tau20ms_thin.txt".format(TEST_DIR=TEST_DIR))
results = pickle.load(open("{TEST_DIR}/test_sample.p".format(TEST_DIR=TEST_DIR), "rb"))
results[-1]["fr"] = 100  # fake one fo the FoM to ensure that testing covers all code


def test_plot_fom():
    assert plot_figures_of_merit(results)


def test_plot_clean_residuals():
    assert plot_clean_residuals(init_data, results, period=500.0)


def test_plot_clean_comps():
    assert plot_clean_components(results, period=500.0)


def test_plot_reconstruction():
    assert plot_reconstruction(results, init_data, period=500.0)

