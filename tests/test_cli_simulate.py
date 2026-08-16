"""Tests for simulated-profile helpers and CLI parser."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

import tauclean.cli_simulate as cli_simulate
from tauclean.cli_simulate import (
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
    monkeypatch.setattr(cli_simulate, "plot_simulated", lambda *args, **kwargs: None)
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
        [sys.executable, "-m", "tauclean.cli_simulate", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "scattering time scale" in completed.stdout