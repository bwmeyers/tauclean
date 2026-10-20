"""Kernel class hierarchy and registry for scattering models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.integrate import simpson as simps

logger = logging.getLogger(__name__)

# Number of scattering times of tail to include when wrapping a kernel around
# the pulse period (exp(-14) ~ 1e-6), and a cap on the number of extra periods.
_TAIL_TAUS = 14.0
_MAX_WRAPS = 500


def _smooth(
    fn1: np.ndarray,
    fn2: np.ndarray,
    scale: float,
    x_array: np.ndarray,
    x_offset: float = 0,
    peak_idx: int = 0,
) -> np.ndarray:
    k = (0.09 / scale) * 30
    blend = 0.5 * (1 + np.tanh(k * (x_array - x_offset)))
    smoothed = fn1 + blend * (fn2 - fn1)
    smoothed[:peak_idx] = (
        fn1[:peak_idx] / np.max(fn1[:peak_idx]) * smoothed[peak_idx]
    )
    return smoothed


@dataclass(frozen=True)
class Kernel(ABC):
    """Abstract scattering kernel with shared normalization/causality logic.

    Pulsar profiles are cyclic, so by default (``periodic=True``) the kernel
    is evaluated as the sum over wraps, ``sum_k h(t + k * period)``, which
    carries scattering tails that run past the end of the profile back to its
    start. The kernel is then normalized to unit area over one period. The
    period is inferred from a uniform grid as ``len(x) * dx``. Set
    ``periodic=False`` for the plain single-period kernel.
    """

    name: str
    periodic: bool = True

    @abstractmethod
    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        """Return unnormalized kernel values."""

    def _linear(self, x: np.ndarray, tau: float, x0: float) -> np.ndarray:
        t = x - x0
        h = self._evaluate(t, tau, x0)
        h[np.where(np.isnan(h) | np.isinf(h))] = 0
        h[x < x0] = 0

        integral = simps(x=t, y=h)
        if integral > 0:
            h = h / integral
        return h

    def _wrapped(self, x: np.ndarray, tau: float, x0: float) -> np.ndarray | None:
        nbins = len(x)
        if nbins < 2:
            return None
        dx = x[1] - x[0]
        if dx <= 0 or not np.allclose(np.diff(x), dx, rtol=1e-6, atol=0):
            return None
        period = nbins * dx

        n_wraps = int(np.ceil(_TAIL_TAUS * abs(tau) / period)) + 1
        if n_wraps > _MAX_WRAPS:
            logger.warning(
                "tau=%g is very large relative to the period (%g); limiting "
                "kernel wrapping to %d periods.",
                tau,
                period,
                _MAX_WRAPS,
            )
            n_wraps = _MAX_WRAPS

        extended = x[0] + dx * np.arange((n_wraps + 1) * nbins)
        h = self._linear(extended, tau, x0)
        # h is normalized to unit area over all wraps, so the folded kernel
        # has unit area over one period.
        return h.reshape(n_wraps + 1, nbins).sum(axis=0)

    def __call__(self, x: np.ndarray, tau: float, x0: float = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.periodic:
            folded = self._wrapped(x, tau, x0)
            if folded is not None:
                return folded
        return self._linear(x, tau, x0)


class ThinKernel(Kernel):
    def __init__(self, periodic: bool = True):
        super().__init__(name="thin", periodic=periodic)

    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        _ = x0
        return (1 / tau) * np.exp(-t / tau)


class ThickKernel(Kernel):
    def __init__(self, periodic: bool = True):
        super().__init__(name="thick", periodic=periodic)

    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        _ = x0
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        h = np.sqrt((np.pi * tau) / (4 * t**3)) * np.exp(
            -tau * np.pi**2 / (16 * t)
        )
        np.seterr(**old_settings)
        return h


class UniformKernel(Kernel):
    def __init__(self, periodic: bool = True):
        super().__init__(name="uniform", periodic=periodic)

    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        _ = x0
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        h = np.sqrt((np.pi**5 * tau**3) / (8 * t**5)) * np.exp(
            -tau * np.pi**2 / (4 * t)
        )
        np.seterr(**old_settings)
        return h


class ThickExpKernel(Kernel):
    def __init__(self, periodic: bool = True):
        super().__init__(name="thick_exp", periodic=periodic)

    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        expdelay = np.log(4 / np.pi)
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        h1 = np.sqrt((np.pi * tau) / (4 * t**3)) * np.exp(
            -tau * np.pi**2 / (16 * t)
        )
        h1[np.where(np.isnan(h1))] = 0
        np.seterr(**old_settings)

        pbfmax = x0 + np.pi**2 * tau / 24
        pbfmax_idx = np.where(t >= pbfmax)[0][0]
        decay_start_idx = np.where(t >= pbfmax + expdelay * tau)[0][0]
        decay_amp = h1[decay_start_idx]
        decay_time = t[decay_start_idx]

        h2 = np.exp(-t / tau)
        h2 = (h2 / h2[decay_start_idx]) * decay_amp
        h2[np.where(np.isnan(h2))] = 0
        return _smooth(
            h1, h2, tau, t, x_offset=decay_time, peak_idx=pbfmax_idx
        )


class UniformExpKernel(Kernel):
    def __init__(self, periodic: bool = True):
        super().__init__(name="uniform_exp", periodic=periodic)

    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        expdelay = np.log(2)
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        h1 = np.sqrt((np.pi**5 * tau**3) / (8 * t**5)) * np.exp(
            -tau * np.pi**2 / (4 * t)
        )
        h1[np.where(np.isnan(h1))] = 0
        np.seterr(**old_settings)

        pbfmax = x0 + np.pi**2 * tau / 10
        pbfmax_idx = np.where(t >= pbfmax)[0][0]
        decay_start_idx = np.where(t >= pbfmax + expdelay * tau)[0][0]
        decay_amp = h1[decay_start_idx]
        decay_time = t[decay_start_idx]

        h2 = np.exp(-t / tau)
        h2 = (h2 / h2[decay_start_idx]) * decay_amp
        h2[np.where(np.isnan(h2))] = 0
        return _smooth(
            h1, h2, tau, t, x_offset=decay_time, peak_idx=pbfmax_idx
        )


class KernelRegistry:
    """Kernel lookup registry."""

    _registry: ClassVar[dict[str, type[Kernel]]] = {
        "thin": ThinKernel,
        "thick": ThickKernel,
        "uniform": UniformKernel,
        "thick_exp": ThickExpKernel,
        "uniform_exp": UniformExpKernel,
    }

    @classmethod
    def choices(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def get(cls, name: str, periodic: bool = True) -> Kernel:
        try:
            return cls._registry[name](periodic=periodic)
        except KeyError as exc:
            options = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown kernel '{name}'. Available: {options}"
            ) from exc


def get_kernel(name: str, periodic: bool = True) -> Kernel:
    """Convenience kernel resolver.

    :param periodic: wrap the kernel around the pulse period (default), or
        use the plain single-period kernel if False.
    """
    return KernelRegistry.get(name, periodic=periodic)
