"""Shared pytest fixtures for tauclean tests."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tauclean.clean_api import clean
from tauclean.cleaner import CleanResult
from tauclean.scripts.cli_simulate import create_intrinsic_pulse, create_scattered_profile

SIMULATED_PERIOD_MS = 500.0
SIMULATED_NBINS = 1024


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a deterministic generator isolated to one test."""
    return np.random.default_rng(12345)


@pytest.fixture
def simulated_profile_factory() -> Callable[[str, float], np.ndarray]:
    """Build synthetic scattered pulse profiles on demand.

    Replaces the previous static tests/*.txt fixtures with profiles
    generated via the same simulate helpers used by the ``simulate`` CLI.
    """

    def build(
        pbftype: str,
        tau: float,
        nbins: int = SIMULATED_NBINS,
        period: float = SIMULATED_PERIOD_MS,
    ) -> np.ndarray:
        # Reseed so the generated profile is independent of test execution order.
        np.random.seed(12345)
        intrinsic = create_intrinsic_pulse(
            [nbins // 8], [nbins / 200], [1.0], nbins=nbins
        )
        _, _, observed = create_scattered_profile(
            intrinsic,
            tau=tau,
            rest_width=period / nbins,
            pbftype=pbftype,
            period=period,
            snr=200.0,
        )
        return observed

    return build


@pytest.fixture
def thin_profile(
    simulated_profile_factory: Callable[[str, float], np.ndarray]
) -> np.ndarray:
    """Return a synthetic thin-screen scattered profile."""
    return simulated_profile_factory("thin", 20.0)


@pytest.fixture
def clean_kwargs() -> dict[str, object]:
    """Return standard CLEAN arguments for integration tests."""
    return {
        "period": SIMULATED_PERIOD_MS,
        "gain": 0.05,
        "pbftype": "thin",
        "iter_limit": 400,
    }


@pytest.fixture
def clean_result_factory(
    thin_profile: np.ndarray, clean_kwargs: dict[str, object]
) -> Callable[[float], CleanResult]:
    """Build clean results on demand rather than during test collection."""

    def build(tau: float) -> CleanResult:
        return clean(thin_profile, tau, **clean_kwargs)

    return build


@pytest.fixture
def clean_results(
    clean_result_factory: Callable[[float], CleanResult]
) -> list[CleanResult]:
    """Return a small tau sweep for plotting and FOM integration tests."""
    return [clean_result_factory(tau) for tau in (15.0, 20.0, 25.0)]


@pytest.fixture(autouse=True)
def close_figures():
    """Prevent Matplotlib figures from leaking across test cases."""
    yield
    plt.close("all")


@pytest.fixture
def serial_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run CLI pool work synchronously while preserving callback behavior."""

    class Pool:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def apply_async(self, function, args, kwargs, callback=None):
            result = function(*args, **kwargs)
            if callback is not None:
                callback(result)

        def close(self) -> None:
            return None

        def join(self) -> None:
            return None

    monkeypatch.setattr("tauclean.scripts.cli_tauclean.mp.Pool", Pool)