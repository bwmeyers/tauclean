"""Shared pytest fixtures for tauclean tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tauclean.clean_api import clean
from tauclean.cleaner import CleanResult


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the directory containing static profile fixtures."""
    return Path(__file__).parent


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a deterministic generator isolated to one test."""
    return np.random.default_rng(12345)


@pytest.fixture
def thin_profile(test_data_dir: Path) -> np.ndarray:
    """Load the canonical thin-screen integration profile."""
    return np.loadtxt(test_data_dir / "simulated_profile_tau20ms_thin.txt")


@pytest.fixture
def clean_kwargs() -> dict[str, object]:
    """Return standard CLEAN arguments for integration tests."""
    return {
        "period": 500.0,
        "gain": 0.05,
        "pbftype": "thin",
        "onpulse_estimator": "440 900",
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