"""Tests for plotting and output-file utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from tauclean.cleaner import CleanResult
from tauclean.plotting import (
    plot_clean_components,
    plot_clean_residuals,
    plot_figures_of_merit,
    plot_instrumental_response,
    plot_reconstruction,
    write_output,
)
from tauclean.response import get_instrumental_response


def test_plot_instrumental_response_with_and_without_components(
    thin_profile: np.ndarray, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    response, width, components = get_instrumental_response(
        thin_profile,
        500.0,
        r_dm_width=2.0,
        r_pb_width=0.5,
        r_av_width=0.0,
        r_pd_width=0.0,
        fast=False,
        return_components=True,
    )

    assert plot_instrumental_response(
        response, width, 500.0, components=components
    )
    assert (tmp_path / "instrumental_response.png").exists()

    assert plot_instrumental_response(
        response, width, 500.0, filename="no_components.png"
    )
    assert (tmp_path / "no_components.png").exists()


def test_plotting_functions_write_expected_pngs(
    clean_results: list[CleanResult],
    thin_profile: np.ndarray,
    tmp_path,
    monkeypatch,
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


def test_plot_clean_residuals_highlights_all_offpulse_regions(
    clean_results: list[CleanResult], monkeypatch
) -> None:
    nbins = clean_results[0].nbins
    clean_result = replace(
        clean_results[0],
        off_bins=np.array([0, 1, 2, 8, 9]),
    )
    figures = []
    monkeypatch.setattr(
        plt,
        "savefig",
        lambda *args, **kwargs: figures.append(plt.gcf()),
    )

    assert plot_clean_residuals(
        clean_result.profile, [clean_result], period=500.0
    )

    assert len(figures) == 1
    bin_width = 500.0 / nbins
    for ax in figures[0].axes:
        spans = [patch for patch in ax.patches if patch.get_alpha() == 0.15]
        assert [
            (span.get_x(), span.get_x() + span.get_width())
            for span in spans
        ] == [
            (0.0, 3 * bin_width),
            (8 * bin_width, 10 * bin_width),
        ]
        assert "off-pulse region(s)" in [
            text.get_text() for text in ax.get_legend().get_texts()
        ]
        assert ax.get_xlim() == (0.0, 500.0)


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
