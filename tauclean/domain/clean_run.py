"""Result objects for deconvolution runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .components import ComponentHistory
from .figures_of_merit import FigureOfMeritSet
from .profile import ProfileData


@dataclass
class CleanResult:
    """Typed output for a single CLEAN run at one tau value."""

    profile: np.ndarray
    init_off_rms: float
    init_on_rms: float
    nbins: int
    nbins_on: int
    nbins_off: int
    rest_func: np.ndarray
    inst_resp_func: np.ndarray
    tau: float
    pbftype: str
    niter: int
    cc: np.ndarray
    ncc: int
    off_bins: np.ndarray
    on_bins: np.ndarray
    off_rms: float
    off_mean: float
    on_rms: float
    on_mean: float
    total_mean: float
    total_rms: float
    recon: np.ndarray
    threshold: float
    profile_data: ProfileData
    figures_of_merit: FigureOfMeritSet
    component_history: ComponentHistory | None = None

    @property
    def nf(self) -> int:
        return self.figures_of_merit.consistence

    @property
    def fr(self) -> float:
        return self.figures_of_merit.positivity

    @property
    def gamma(self) -> float:
        return self.figures_of_merit.skewness
