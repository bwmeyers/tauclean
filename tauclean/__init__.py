"""
########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

----------------
tauclean module.
----------------

This module contains the functionality required by the 'tauclean' and 'simulate' scripts.
"""

__author__ = "Bradley W. Meyers"

try:
    from tauclean._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

from .clean_api import clean
from .cleaner import CleanResult, Cleaner
from .components import ComponentHistory, SubtractedComponent
from .fom import (
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
    ResponseComponent,
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
    "ResponseComponent",
    "SubtractedComponent",
    "TauEstimate",
    "TauSearchAnalyzer",
    "UserDefinedOnPulseNoiseEstimator",
    "__version__",
    "clean",
    "dm_delay",
    "gaussian",
    "get_instrumental_response",
    "get_restoring_function",
    "reconstruct",
]

__citation__ = """
If you make use of this tauclean implementation, please cite the following paper as appropriate:

% Original method description, implementation (in FORTRAN) and tests on simulated and real data.
@ARTICLE{2003ApJ...584..782B,
       author = {{Bhat}, N.~D.~R. and {Cordes}, J.~M. and {Chatterjee}, S.},
        title = "{A CLEAN-based Method for Deconvolving Interstellar Pulse Broadening from Radio Pulses}",
      journal = {The Astrophysical Journal},
     keywords = {ISM: Structure, Methods: Data Analysis, Stars: Pulsars: General, 
                Radio Continuum: General, Scattering, Astrophysics},
         year = "2003",
        month = "Feb",
       volume = {584},
        pages = {782-790},
          doi = {10.1086/345775},
archivePrefix = {arXiv},
       eprint = {astro-ph/0207451},
 primaryClass = {astro-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2003ApJ...584..782B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

% A recent Pythonic implementation with more robust uncertainty estimation and optimal value selection.
@ARTICLE{2024ApJ...962..131Y,
       author = {{Young}, Olivia and {Lam}, Michael T.},
        title = "{Redeveloping a CLEAN Deconvolution Algorithm for Scatter-broadened Radio Pulsar Signals}",
      journal = {The Astrophysical Journal},
     keywords = {Radio pulsars, Interstellar medium, Deconvolution, Interstellar scattering, 1353, 847, 1910, 854, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2024,
        month = feb,
       volume = {962},
       number = {2},
          eid = {131},
        pages = {131},
          doi = {10.3847/1538-4357/ad1ce7},
archivePrefix = {arXiv},
       eprint = {2306.06046},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...962..131Y},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
