"""Integration tests for the canonical CLEAN API."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from tauclean.clean_api import clean
from tauclean.cleaner import CleanResult
from tauclean.kernels import get_kernel


@pytest.mark.parametrize(
    ("pbftype", "tau"),
    [
        ("thin", 20.0),
        ("thick", 1.0),
        ("uniform", 3.0),
    ],
)
def test_clean_returns_finite_typed_result(
    simulated_profile_factory: Callable[[str, float], np.ndarray],
    pbftype: str,
    tau: float,
) -> None:
    result = clean(
        simulated_profile_factory(pbftype, tau),
        tau,
        period=500.0,
        pbftype=pbftype,
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
        iter_limit=1,
    )

    assert result.niter == 1


def test_clean_tracks_component_history(thin_profile: np.ndarray) -> None:
    result = clean(
        thin_profile,
        20.0,
        period=500.0,
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
        iter_limit=1,
    )

    assert result.pbftype == "thick"


def test_clean_rejects_unknown_kernel(thin_profile: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unknown kernel"):
        clean(thin_profile, 20.0, pbftype="unknown")