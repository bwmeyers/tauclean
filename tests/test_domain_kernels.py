"""Tests for the kernel registry and scattering-kernel contracts."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import simpson

from tauclean.domain.kernels import KernelRegistry, get_kernel


@pytest.mark.parametrize("name", KernelRegistry.choices())
def test_registered_kernel_is_finite_causal_and_normalized(name: str) -> None:
    x = np.linspace(0.0, 500.0, 2048)

    kernel = get_kernel(name)(x, tau=10.0, x0=50.0)

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