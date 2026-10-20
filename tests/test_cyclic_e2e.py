"""End-to-end tests that pulse profiles are handled as cyclic.

Each test simulates a scattered profile with the simulation helpers and
recovers it with the CLEAN API, with the pulse placed so that its scattering
tail wraps through the end of the profile.
"""

from __future__ import annotations

import numpy as np
import pytest

from tauclean.clean_api import clean
from tauclean.fom import TauSearchAnalyzer
from tauclean.scripts.cli_simulate import (
    create_intrinsic_pulse,
    create_scattered_profile,
)

PERIOD_MS = 500.0
NBINS = 512


def _simulate(
    position: int,
    tau: float,
    pbftype: str = "thin",
    snr: float = 300.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    intrinsic = create_intrinsic_pulse(
        [position], [3.0], [1.0], nbins=NBINS
    )
    _, _, observed = create_scattered_profile(
        intrinsic,
        tau=tau,
        rest_width=PERIOD_MS / NBINS,
        pbftype=pbftype,
        period=PERIOD_MS,
        snr=snr,
    )
    return intrinsic, observed


def _circular_distance(a: int, b: int, nbins: int = NBINS) -> int:
    d = abs(int(a) - int(b)) % nbins
    return min(d, nbins - d)


def test_tau_recovery_does_not_depend_on_pulse_position() -> None:
    tau = 20.0
    taus = np.linspace(8.0, 36.0, 15)
    _, observed = _simulate(NBINS // 4, tau)
    # The same noisy data, rolled so the scattering tail wraps the profile end.
    rolled = np.roll(observed, -(NBINS // 4) - 5)
    analyzer = TauSearchAnalyzer()

    estimates = []
    for profile in (observed, rolled):
        results = [
            clean(profile, t, period=PERIOD_MS, pbftype="thin", iter_limit=500)
            for t in taus
        ]
        estimates.append(analyzer.estimate_best_tau(results).best_tau)

    assert estimates[0] == pytest.approx(estimates[1], abs=1e-6)
    assert estimates[0] == pytest.approx(tau, rel=0.3)


def test_large_tau_pulse_near_edge_is_reconstructed_in_place() -> None:
    position = NBINS - 8
    intrinsic, observed = _simulate(position, tau=150.0, snr=500.0, seed=3)

    result = clean(
        observed, 150.0, period=PERIOD_MS, pbftype="thin", iter_limit=800
    )

    assert np.all(np.isfinite(result.recon))
    assert _circular_distance(
        int(np.argmax(result.recon)), position
    ) <= 12
    # Components are placed on both sides of the profile edge.
    placed = np.flatnonzero(result.cc)
    assert placed.min() < 50
    assert placed.max() > NBINS - 50


@pytest.mark.parametrize("periodic", [True, False])
def test_thick_screen_edge_cleaning_keeps_only_positive_components(
    periodic: bool,
) -> None:
    tau = 80.0
    _, observed = _simulate(NBINS - 8, tau, pbftype="thick", snr=500.0, seed=3)

    result = clean(
        observed,
        tau,
        period=PERIOD_MS,
        pbftype="thick",
        iter_limit=800,
        periodic_kernel=periodic,
        track_components=True,
    )

    assert result.ncc > 0
    assert np.all(result.cc >= 0)
    assert np.isfinite(result.gamma)
    assert result.component_history is not None
    assert all(
        component.peak_value > 0
        for component in result.component_history.components
    )
