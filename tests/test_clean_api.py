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


def _cyclic_scattered_profile(shift: int, nbins: int = 512) -> np.ndarray:
    from tauclean.circular import bin_grid, circ_convolve
    from tauclean.response import gaussian

    period = 500.0
    x = bin_grid(nbins, period)
    pulse = gaussian(x, 100.0, 6.0)
    kernel = get_kernel("thin")(x, 25.0)
    scattered = circ_convolve(pulse, kernel / kernel.sum())
    scattered = scattered / scattered.max()
    noise = np.random.default_rng(7).normal(0, 0.01, nbins)
    return np.roll(scattered + noise, shift)


@pytest.mark.parametrize("shift", [-20, 100, 250, 400])
def test_clean_is_shift_invariant(shift: int) -> None:
    kwargs = {"period": 500.0, "pbftype": "thin", "iter_limit": 300}
    base = clean(_cyclic_scattered_profile(0), 25.0, **kwargs)
    rolled = clean(_cyclic_scattered_profile(shift), 25.0, **kwargs)

    assert rolled.niter == base.niter
    np.testing.assert_allclose(
        rolled.recon, np.roll(base.recon, shift), atol=1e-6
    )
    np.testing.assert_allclose(rolled.cc, np.roll(base.cc, shift), atol=1e-9)


def test_clean_handles_pulse_straddling_profile_edge() -> None:
    # Pulse at phase ~0.2 with a long tail, rolled so the peak is near bin 0.
    profile = _cyclic_scattered_profile(-100)
    result = clean(profile, 25.0, period=500.0, pbftype="thin", iter_limit=300)

    assert result.ncc > 0
    assert np.all(np.isfinite(result.recon))
    # The recovered peak sits at the edge and its flanks wrap smoothly.
    peak = int(np.argmax(result.recon))
    assert min(peak, 512 - peak) <= 15
    assert result.recon[-1] > 0.5
    assert result.recon[0] > 0.5
