"""Integration tests for the canonical CLEAN API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tauclean.domain.clean_api import clean
from tauclean.domain.clean_run import CleanResult
from tauclean.domain.kernels import get_kernel


@pytest.mark.parametrize(
    ("filename", "tau", "pbftype", "onpulse"),
    [
        ("simulated_profile_tau20ms_thin.txt", 20.0, "thin", "440 900"),
        ("simulated_profile_tau1ms_thick.txt", 1.0, "thick", "128 700"),
        ("simulated_profile_tau3ms_uniform.txt", 3.0, "uniform", "128 700"),
    ],
)
def test_clean_returns_finite_typed_result(
    test_data_dir: Path,
    filename: str,
    tau: float,
    pbftype: str,
    onpulse: str,
) -> None:
    result = clean(
        np.loadtxt(test_data_dir / filename),
        tau,
        period=500.0,
        pbftype=pbftype,
        onpulse_estimator=onpulse,
        iter_limit=25,
    )

    assert isinstance(result, CleanResult)
    assert result.tau == tau
    assert result.pbftype == pbftype
    assert result.niter <= 25
    assert result.ncc > 0
    assert np.all(np.isfinite(result.recon))
    assert np.isfinite(result.init_off_rms)
    assert np.isfinite(result.off_rms)


def test_clean_honors_iteration_limit(thin_profile: np.ndarray) -> None:
    result = clean(
        thin_profile,
        20.0,
        period=500.0,
        onpulse_estimator="440 900",
        iter_limit=1,
    )

    assert result.niter == 1


def test_clean_tracks_component_history(thin_profile: np.ndarray) -> None:
    result = clean(
        thin_profile,
        20.0,
        period=500.0,
        onpulse_estimator="440 900",
        iter_limit=3,
        track_components=True,
    )

    assert result.component_history is not None
    assert len(result.component_history) == result.niter
    assert result.component_history.components[0].iteration == 1


def test_explicit_kernel_overrides_pbftype(thin_profile: np.ndarray) -> None:
    result = clean(
        thin_profile,
        20.0,
        period=500.0,
        pbftype="thin",
        kernel=get_kernel("thick"),
        onpulse_estimator="440 900",
        iter_limit=1,
    )

    assert result.pbftype == "thick"


def test_clean_rejects_unknown_kernel(thin_profile: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unknown kernel"):
        clean(thin_profile, 20.0, pbftype="unknown")