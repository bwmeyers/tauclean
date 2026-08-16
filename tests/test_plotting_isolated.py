"""Tests for plotting and output-file utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tauclean.domain.clean_run import CleanResult
from tauclean.plotting import (
    plot_clean_components,
    plot_clean_residuals,
    plot_figures_of_merit,
    plot_reconstruction,
    write_output,
)


def test_plotting_functions_write_expected_pngs(
    clean_results: list[CleanResult], thin_profile: np.ndarray, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert plot_figures_of_merit(clean_results)
    assert plot_clean_residuals(thin_profile, clean_results, period=500.0)
    assert plot_clean_components(clean_results, period=500.0)
    assert plot_reconstruction(clean_results, thin_profile, period=500.0)

    expected = {
        "tauclean_fom.png",
        "tauclean_fom.txt",
        "clean_residuals_thin-tau15.png",
        "clean_components_thin-tau20.png",
        "reconstruction_thin-tau25.png",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}


def test_write_output_preserves_clean_result_arrays(
    clean_result_factory: Callable[[float], CleanResult], tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = clean_result_factory(20.0)

    write_output([result])

    components = np.loadtxt(tmp_path / "clean_components_thin-tau20.txt")
    reconstruction = np.loadtxt(tmp_path / "reconstruction_thin-tau20.txt")
    np.testing.assert_allclose(components, result.cc)
    np.testing.assert_allclose(reconstruction[:, 0], result.recon)
    np.testing.assert_allclose(reconstruction[:, 1], result.profile)