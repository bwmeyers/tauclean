"""Tests for simulated-profile helpers and CLI parser."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from tauclean.scripts import cli_simulate
from tauclean.scripts.cli_simulate import (
    create_intrinsic_pulse,
    create_scattered_profile,
    write_data,
)


def test_simulation_helpers_generate_finite_profile_and_files(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    intrinsic = create_intrinsic_pulse([20], [3.0], [1.0], nbins=64)
    kernel, scattered, observed = create_scattered_profile(
        intrinsic,
        tau=2.0,
        rest_width=1.0,
        period=64.0,
        snr=100.0,
    )

    assert all(
        np.all(np.isfinite(values)) for values in (kernel, scattered, observed)
    )
    write_data(intrinsic, kernel, scattered, observed, "thin", 2.0)
    assert (tmp_path / "sim-profile_thin-tau2.txt").exists()


def test_simulate_main_parses_and_writes_data(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_simulate, "plot_simulated", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate",
            "-n",
            "64",
            "-m",
            "20",
            "-w",
            "3",
            "-a",
            "1",
            "-t",
            "2",
            "--write",
        ],
    )

    cli_simulate.main()

    assert (tmp_path / "sim-profile_thin-tau2.txt").exists()


def test_simulate_module_help_exits_successfully() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tauclean.scripts.cli_simulate", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "scattering time scale" in completed.stdout


def _noiseless(intrinsic, tau, period=500.0, **kwargs):
    _, scattered, _ = create_scattered_profile(
        intrinsic,
        tau=tau,
        rest_width=period / len(intrinsic),
        period=period,
        snr=1e12,
        **kwargs,
    )
    return scattered


def test_scattering_tail_wraps_past_profile_end() -> None:
    nbins = 256
    intrinsic = create_intrinsic_pulse([nbins - 10], [2.0], [1.0], nbins=nbins)
    scattered = _noiseless(intrinsic, tau=40.0)

    assert np.argmax(scattered) >= nbins - 10
    # Power from the tail re-emerges at the start of the profile.
    assert scattered[:20].max() > 0.1 * scattered.max()
    # Total flux is conserved by the cyclic convolution.
    assert scattered.sum() == pytest.approx(intrinsic.sum(), rel=1e-6)


@pytest.mark.parametrize("shift", [-30, 17, 128, 200])
def test_scattered_profile_is_shift_equivariant(shift: int) -> None:
    nbins = 256
    intrinsic = create_intrinsic_pulse([60], [3.0], [1.0], nbins=nbins)
    base = _noiseless(intrinsic, tau=30.0)
    rolled = _noiseless(np.roll(intrinsic, shift), tau=30.0)

    np.testing.assert_allclose(rolled, np.roll(base, shift), atol=1e-9)


def test_causal_kernel_does_not_move_peak_earlier() -> None:
    nbins = 256
    intrinsic = create_intrinsic_pulse([100], [3.0], [1.0], nbins=nbins)
    scattered = _noiseless(intrinsic, tau=30.0)

    assert np.argmax(scattered) >= np.argmax(intrinsic)


def test_periodic_kernel_changes_thick_screen_and_can_be_disabled() -> None:
    nbins = 256
    intrinsic = create_intrinsic_pulse([200], [2.0], [1.0], nbins=nbins)
    kwargs = {
        "tau": 150.0,
        "rest_width": 2.0,
        "period": 500.0,
        "snr": 1e12,
        "pbftype": "thick",
    }

    _, s_per, _ = create_scattered_profile(intrinsic, **kwargs)
    _, s_lin, _ = create_scattered_profile(
        intrinsic, periodic_kernel=False, **kwargs
    )

    assert s_per.sum() == pytest.approx(intrinsic.sum(), rel=1e-6)
    assert s_lin.sum() == pytest.approx(intrinsic.sum(), rel=1e-6)
    assert not np.allclose(s_per, s_lin, rtol=1e-2, atol=1e-3)
