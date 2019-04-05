# tauclean

Python 3 implementation of the pulse profile scattering deconvolution (CLEAN) code described by 
[Bhat et al. (2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...584..782B/abstract "Description paper").

Programs included:
* simulate - A simulation tool to artificially create scattered pulse profiles.
* tauclean - The deconvolution program which will attempt to reconstruct the intrinsic, unscattered pulse 
profile. It can: 
   * search the pulse broadening time scale (tau) parameter space and provide metrics to judge the best 
   model, or 
   * deconvolve a specified PBF model (with a known tau) from the data
 
