"""Tests for figures-of-merit evaluation and tau selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tauclean.domain.figures_of_merit import (
    FigureOfMeritEvaluator,
    FigureOfMeritSet,
    TauSearchAnalyzer,
)


@pytest.fixture
def evaluator() -> FigureOfMeritEvaluator:
    return FigureOfMeritEvaluator()


def test_consistence_counts_values_within_threshold(
    evaluator: FigureOfMeritEvaluator
) -> None:
    profile = np.array([0.0, 0.5, -0.5, 4.0])

    assert evaluator.consistence(profile, off_rms=1.0, threshold=3.0) == 3


def test_positivity_handles_zero_and_negative_residuals(
    evaluator: FigureOfMeritEvaluator
) -> None:
    assert np.isnan(evaluator.positivity(np.zeros(4), off_rms=1.0))
    assert evaluator.positivity(np.array([0.0, -3.0]), off_rms=1.0) > 0


def test_skewness_of_single_component_is_zero(
    evaluator: FigureOfMeritEvaluator
) -> None:
    components = np.zeros(8)
    components[3] = 1.0

    assert evaluator.skewness(components) == 0.0


def test_figure_of_merit_set_exposes_derived_values() -> None:
    fom = FigureOfMeritSet(
        consistence=8,
        positivity=2.0,
        skewness=4.0,
        total_rms=3.0,
        initial_off_rms=1.5,
        nbins=10,
    )

    assert fom.combined == 3.0
    assert fom.residual_ratio == 2.0
    assert fom.consistence_fraction == 0.8


def test_tau_search_builds_expected_series() -> None:
    analyzer = TauSearchAnalyzer()
    results = [
        SimpleNamespace(
            tau=float(tau),
            figures_of_merit=FigureOfMeritSet(
                consistence=10 - tau,
                positivity=float(tau),
                skewness=float(-tau),
                total_rms=2.0,
                initial_off_rms=1.0,
                nbins=10,
            ),
        )
        for tau in (1, 2, 3)
    ]

    taus, series = analyzer.build_series(results)

    np.testing.assert_array_equal(taus, [1.0, 2.0, 3.0])
    assert [item.name for item in series] == [
        "f_r",
        "gamma",
        "f_c",
        "r_sigma",
        "r_phi",
    ]


def test_tau_search_short_grid_falls_back_to_heuristics() -> None:
    analyzer = TauSearchAnalyzer()
    fom = FigureOfMeritSet(8, 1.0, -1.0, 2.0, 1.0, 10)
    results = [
        SimpleNamespace(tau=float(tau), figures_of_merit=fom)
        for tau in (1, 2, 3)
    ]

    estimate = analyzer.estimate_best_tau(results)

    assert np.isfinite(estimate.best_tau)
    assert estimate.uncertainty >= 1.0