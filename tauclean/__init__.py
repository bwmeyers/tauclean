#! /usr/bin/env python
"""
----------------
tauclean module.
----------------

This module contains the functionality required by the 'tauclean' and 'simulate' scripts.
"""

__all__ = ['fom', 'pbf', 'clean', 'plotting']
__author__ = 'Bradley Meyers'
__version__ = 1.0
__citation__ = """
If you make use of tauclean please cite the following paper as appropriate:

% Method description, implementation (in FORTRAN) and tests on simulated and real data
@ARTICLE{2003ApJ...584..782B,
       author = {{Bhat}, N.~D.~R. and {Cordes}, J.~M. and {Chatterjee}, S.},
        title = "{A CLEAN-based Method for Deconvolving Interstellar Pulse Broadening from Radio Pulses}",
      journal = {\apj},
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

"""

