"""Tests for the kernel registry and scattering-kernel contracts."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import simpson

from tauclean.circular import bin_grid
from tauclean.kernels import KernelRegistry, get_kernel


@pytest.mark.parametrize("name", KernelRegistry.choices())
def test_single_period_kernel_is_finite_causal_and_normalized(
    name: str,
) -> None:
    x = np.linspace(0.0, 500.0, 2048)

    kernel = get_kernel(name, periodic=False)(x, tau=10.0, x0=50.0)

    assert np.all(np.isfinite(kernel))
    assert np.all(kernel[x < 50.0] == 0.0)
    assert np.isclose(simpson(y=kernel, x=x), 1.0, rtol=1e-6)


def test_registry_choices_resolve_to_named_kernels() -> None:
    choices = KernelRegistry.choices()

    assert choices
    assert [get_kernel(name).name for name in choices] == choices


def test_unknown_kernel_reports_available_choices() -> None:
    with pytest.raises(ValueError, match="Unknown kernel 'not-a-kernel'"):
        get_kernel("not-a-kernel")


PERIOD = 500.0
NBINS = 1024


@pytest.mark.parametrize("name", KernelRegistry.choices())
@pytest.mark.parametrize("tau", [10.0, 25.0, 400.0, 1500.0])
def test_periodic_kernel_is_finite_nonnegative_and_unit_area(
    name: str, tau: float
) -> None:
    x = bin_grid(NBINS, PERIOD)

    kernel = get_kernel(name)(x, tau)

    assert np.all(np.isfinite(kernel))
    assert np.all(kernel >= 0)
    closed = np.append(kernel, kernel[0])
    assert np.trapz(closed, dx=PERIOD / NBINS) == pytest.approx(1.0, rel=3e-2)


def test_periodic_is_the_default_and_can_be_disabled() -> None:
    assert get_kernel("thin").periodic
    assert not get_kernel("thin", periodic=False).periodic


def test_periodic_kernel_matches_single_period_for_small_tau() -> None:
    x = bin_grid(NBINS, PERIOD)
    periodic = get_kernel("thin")(x, 3.0)
    single = get_kernel("thin", periodic=False)(x, 3.0)

    np.testing.assert_allclose(periodic, single, rtol=1e-3, atol=1e-6)


def test_periodic_thin_kernel_is_the_geometric_sum_of_wraps() -> None:
    tau = 300.0
    x = bin_grid(NBINS, PERIOD)

    kernel = get_kernel("thin")(x, tau)

    # sum_k exp(-(t + kP) / tau) / tau = exp(-t / tau) / (tau * (1 - r))
    ratio = np.exp(-PERIOD / tau)
    expected = np.exp(-x / tau) / (tau * (1 - ratio))
    np.testing.assert_allclose(kernel, expected, rtol=2e-3)


@pytest.mark.parametrize("name", ["thick", "uniform", "thick_exp"])
def test_periodic_kernel_differs_from_single_period_at_large_tau(
    name: str,
) -> None:
    x = bin_grid(NBINS, PERIOD)

    periodic = get_kernel(name)(x, 300.0)
    single = get_kernel(name, periodic=False)(x, 300.0)

    # The wrapped tail lifts the start of the profile relative to the peak.
    assert periodic[0] / periodic.max() > single[0] / single.max()
    assert np.max(np.abs(periodic - single)) > 0.05 * single.max()


def test_periodic_kernel_with_offset_wraps_tail_before_x0() -> None:
    x = bin_grid(NBINS, PERIOD)
    x0 = x[NBINS // 4]

    kernel = get_kernel("thin")(x, 100.0, x0=x0)

    assert kernel[NBINS // 4 - 1] > 0
    assert np.argmax(kernel) == NBINS // 4


def test_periodic_kernel_falls_back_on_nonuniform_grid() -> None:
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0])

    periodic = get_kernel("thin")(x, 2.0)
    single = get_kernel("thin", periodic=False)(x, 2.0)

    np.testing.assert_allclose(periodic, single)
