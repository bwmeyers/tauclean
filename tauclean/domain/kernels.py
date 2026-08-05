"""Kernel class hierarchy and registry for scattering models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.integrate import simpson as simps


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
    """Abstract scattering kernel with shared normalization/causality logic."""

    name: str

    @abstractmethod
    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        """Return unnormalized kernel values."""

    def __call__(self, x: np.ndarray, tau: float, x0: float = 0) -> np.ndarray:
        t = x - x0
        h = self._evaluate(t, tau, x0)
        h[np.where(np.isnan(h) | np.isinf(h))] = 0
        h[x < x0] = 0

        integral = simps(x=t, y=h)
        if integral > 0:
            h = h / integral
        return h


class ThinKernel(Kernel):
    def __init__(self):
        super().__init__(name="thin")

    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        _ = x0
        return (1 / tau) * np.exp(-t / tau)


class ThickKernel(Kernel):
    def __init__(self):
        super().__init__(name="thick")

    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        _ = x0
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        h = np.sqrt((np.pi * tau) / (4 * t**3)) * np.exp(
            -tau * np.pi**2 / (16 * t)
        )
        np.seterr(**old_settings)
        return h


class UniformKernel(Kernel):
    def __init__(self):
        super().__init__(name="uniform")

    def _evaluate(self, t: np.ndarray, tau: float, x0: float) -> np.ndarray:
        _ = x0
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        h = np.sqrt((np.pi**5 * tau**3) / (8 * t**5)) * np.exp(
            -tau * np.pi**2 / (4 * t)
        )
        np.seterr(**old_settings)
        return h


class ThickExpKernel(Kernel):
    def __init__(self):
        super().__init__(name="thick_exp")

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
    def __init__(self):
        super().__init__(name="uniform_exp")

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
    def get(cls, name: str) -> Kernel:
        try:
            return cls._registry[name]()
        except KeyError as exc:
            options = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown kernel '{name}'. Available: {options}"
            ) from exc


def get_kernel(name: str) -> Kernel:
    """Convenience kernel resolver."""
    return KernelRegistry.get(name)
