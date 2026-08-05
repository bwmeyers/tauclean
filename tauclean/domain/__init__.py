"""Domain models for the object-oriented tauclean refactor."""

from .clean_api import clean
from .clean_run import CleanResult
from .cleaner import Cleaner
from .components import ComponentHistory, SubtractedComponent
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
    "CleanResult",
    "Cleaner",
    "ComponentHistory",
    "FigureOfMeritEvaluator",
    "FigureOfMeritSeries",
    "FigureOfMeritSet",
    "Kernel",
    "KernelRegistry",
    "NoiseEstimator",
    "NoiseStats",
    "ProfileData",
    "SubtractedComponent",
    "TauEstimate",
    "TauSearchAnalyzer",
    "UserDefinedOnPulseNoiseEstimator",
    "clean",
    "dm_delay",
    "gaussian",
    "get_instrumental_response",
    "get_kernel",
    "get_restoring_function",
    "reconstruct",
]
