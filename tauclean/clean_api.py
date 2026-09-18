"""Top-level deconvolution API over the canonical Cleaner service."""

from __future__ import annotations

import logging

import numpy as np

from .cleaner import Cleaner, CleanResult
from .kernels import Kernel, get_kernel
from .profile import ProfileData


def clean(
    data: np.ndarray | ProfileData,
    tau: float,
    period: float = 100.0,
    rest_func: np.ndarray | None = None,
    inst_resp_func: np.ndarray | None = None,
    gain: float = 0.05,
    threshold: float = 3.0,
    pbftype: str = "thin",
    kernel: Kernel | None = None,
    iter_limit: int = 1000,
    onpulse_estimator: str = "auto",
    component_window: str | None = None,
    track_components: bool = False,
    logger: logging.Logger | None = None,
) -> CleanResult:
    """Run CLEAN deconvolution using the canonical service implementation.

    :param component_window: restrict where clean components may be placed.
        One of 'onpulse'/'on', 'offpulse'/'off', 'all'/'total', or a
        comma-separated list of 'START-END' bin ranges (end-exclusive).
        Defaults to no restriction.
    """
    cleaner = Cleaner(
        period=period,
        rest_func=rest_func,
        inst_resp_func=inst_resp_func,
        gain=gain,
        threshold=threshold,
        kernel=kernel if kernel is not None else get_kernel(pbftype),
        iter_limit=iter_limit,
        onpulse_estimator=onpulse_estimator,
        component_window=component_window,
        track_components=track_components,
        logger=logger,
    )
    return cleaner.run(data, tau)
