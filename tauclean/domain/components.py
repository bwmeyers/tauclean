"""Component models for iterative profile subtraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SubtractedComponent:
    """Represents one constructed component removed during CLEAN."""

    iteration: int
    peak_index: int
    peak_value: float
    component_amplitude: float
    alignment_offset: int
    clean_component: np.ndarray
    instrumental_response: np.ndarray
    kernel_response: np.ndarray
    impulse_response: np.ndarray
    raw_convolved_response: np.ndarray
    subtracted_response: np.ndarray
    residual_max: float
    residual_off_rms: float


@dataclass
class ComponentHistory:
    """Ordered collection of per-iteration components."""

    components: list[SubtractedComponent] = field(default_factory=list)

    def add(self, component: SubtractedComponent) -> None:
        self.components.append(component)

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)
