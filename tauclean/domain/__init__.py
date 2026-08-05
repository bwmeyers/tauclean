"""Domain models for the object-oriented tauclean refactor."""

from .components import ComponentHistory, SubtractedComponent
from .clean_api import clean
from .cleaner import Cleaner
from .clean_run import CleanResult
from .figures_of_merit import (
    FigureOfMeritEvaluator,
    FigureOfMeritSeries,
    FigureOfMeritSet,
    TauEstimate,
    TauSearchAnalyzer,
)
from .kernels import Kernel, KernelRegistry, get_kernel
from .noise import (
    AutoWindowNoiseEstimator,
    NoiseEstimator,
    NoiseStats,
    UserDefinedOnPulseNoiseEstimator,
)
from .profile import ProfileData
from .response import (
    dm_delay,
    gaussian,
    get_instrumental_response,
    get_restoring_function,
    reconstruct,
)

__all__ = [
    "AutoWindowNoiseEstimator",
    "Cleaner",
    "CleanResult",
    "ComponentHistory",
    "clean",
    "dm_delay",
    "FigureOfMeritEvaluator",
    "FigureOfMeritSeries",
    "FigureOfMeritSet",
    "gaussian",
    "get_instrumental_response",
    "Kernel",
    "KernelRegistry",
    "NoiseEstimator",
    "NoiseStats",
    "ProfileData",
    "get_restoring_function",
    "reconstruct",
    "SubtractedComponent",
    "TauEstimate",
    "TauSearchAnalyzer",
    "UserDefinedOnPulseNoiseEstimator",
    "get_kernel",
]
