"""Tests for CLEAN component history models."""

from __future__ import annotations

import numpy as np

from tauclean.domain.components import ComponentHistory, SubtractedComponent


def test_component_history_preserves_component_order() -> None:
    response = np.array([0.0, 1.0, 0.0])
    first = SubtractedComponent(
        iteration=1,
        peak_index=1,
        peak_value=4.0,
        component_amplitude=0.2,
        alignment_offset=0,
        clean_component=response,
        instrumental_response=response,
        kernel_response=response,
        impulse_response=response,
        raw_convolved_response=response,
        subtracted_response=response,
        residual_max=3.0,
        residual_off_rms=0.1,
    )
    history = ComponentHistory()

    history.add(first)

    assert len(history) == 1
    assert list(history) == [first]