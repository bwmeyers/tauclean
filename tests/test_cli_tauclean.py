"""Tests for the tauclean command-line workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pytest

from tauclean.scripts.cli_tauclean import execute_tauclean, main


def _args(profile: str, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "profile": profile,
        "tau": 20.0,
        "search": None,
        "period": 500.0,
        "dm": 0.0,
        "freq": 1.4,
        "bw": 0.256,
        "nchan": 1024,
        "coherent": True,
        "native_dt": 100.0,
        "kernel": "thin",
        "onpulse": "440 900",
        "thresh": 3.0,
        "gain": 0.05,
        "iterlim": 2,
        "ncpus": 1,
        "noplot_r": True,
        "noplot_f": True,
        "nowrite": True,
        "truth": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_execute_tauclean_runs_single_tau_serially(
    thin_profile: np.ndarray, tmp_path, serial_pool
) -> None:
    profile = tmp_path / "profile.txt"
    np.savetxt(profile, thin_profile)

    execute_tauclean(_args(str(profile)))


def test_tauclean_main_parses_and_executes(
    thin_profile: np.ndarray, tmp_path, serial_pool, monkeypatch
) -> None:
    profile = tmp_path / "profile.txt"
    np.savetxt(profile, thin_profile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tauclean",
            str(profile),
            "--tau",
            "20",
            "--period",
            "500",
            "--onpulse",
            "440 900",
            "--iterlim",
            "2",
            "--ncpus",
            "1",
            "--nowrite",
            "--noplot_r",
            "--noplot_f",
        ],
    )

    main()


@pytest.mark.parametrize(
    "search",
    [(20.0, 10.0, 1.0), (1.0, 10.0, 0.0), (0.0, 10.0, 1.0)],
)
def test_execute_tauclean_rejects_invalid_search_ranges(
    thin_profile: np.ndarray, tmp_path, search
) -> None:
    profile = tmp_path / "profile.txt"
    np.savetxt(profile, thin_profile)

    with pytest.raises(SystemExit):
        execute_tauclean(_args(str(profile), tau=None, search=search))


def test_tauclean_module_help_exits_successfully() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tauclean.scripts.cli_tauclean", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Deconvolution options" in completed.stdout