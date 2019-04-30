# tauclean
[![Build Status](https://travis-ci.com/bwmeyers/tauclean.svg?branch=master)](https://travis-ci.com/bwmeyers/tauclean) 
[![Coverage Status](https://coveralls.io/repos/github/bwmeyers/tauclean/badge.svg?branch=master)](https://coveralls.io/github/bwmeyers/tauclean?branch=master)

Python 3 implementation of the pulse profile scattering deconvolution (CLEAN) code described by 
[Bhat et al. (2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...584..782B/abstract "Description paper").
The technique is based on the extensively used CLEAN algorithms used for synthesis image reconstruction. 

Included in package are two scripts:
* `simulate` - A simulation tool to artificially create scattered pulse profiles
* `tauclean` - The deconvolution program which will attempt to reconstruct the intrinsic, unscattered pulse 
profile. It can: 
   * search the pulse broadening time scale (tau) parameter space and provide metrics to judge the best 
   model, or 
   * deconvolve a specified PBF model (with a known tau) from the data
 

## Usage
### `simulate`
Primarily for testing purposes, the `simulate` script will got through the nominal steps taken in the CLEAN code itself 
to produce a scattered pulsar profile by convolving the user-described intrinsic pulse profile with a specified 
scattering kernel. 

An example use case would be:
    
    simulate -n 1024 -p 500 -m 500 -a 12 -w 10 -k thin -t 20.0 
    
which will create a 1024-bin profile (500 ms period) with a single Gaussian component profile centered at bin 500 with 
an amplitude of 12 and a width of 10 bins. This "intrinsic" profile is then convolved with a thin screen scattering 
kernel with a pulse broadening time scale (tau) of 20 ms to produce the final scattered profile. There are also options 
to specify the desired signal-to-noise ratio of the output profile, as well as describe residual dispersion smearing 
incurred when using incoherent dedispersion (in the above case, the default is to assume coherently dedispersed data, 
thus there will be no residual smearing).

### `tauclean`
The `tauclean` script is the user interface to the deconvolution code. The primary output of this script is the 
reconstructed ("intrinsic") pulse profile. It follows the method laid out in Bhat et al. (2003). `tauclean` can operate 
in two modes:
* search (`-s`) - where the used does not _a priori_ know what the pulse broadening time scale is, but can estimate the range 
of possibilty. 
* deconvolve (`-t`) - where the user does know the pulse broadening time scale and just wants to deconvolve the profile to 
reconstruct the intrinsic pulse profile.

An example, where the user knows that the pulse broadening time scale is 20 ms, and has good evidence that the thin 
scattering screen is suitable, would be something like:

    tauclean -t 20.0 -k thin -p 500 -o 440 900 --coherent [file]
    
where the period (`-p`) is 500 ms, the "on-pulse" region ( `-o`, defining the bounds of where the scattered power 
extends) is between bins 440-900, and the data have been coherently dedispersed (`--coherent`). 

If the user did not know the pulse broadening time scale, then the above command would change to:

    tauclean -s 15.0 25.0 1.0 -k thin -p 500 -o 440 900 --coherent [file]
    
where now the code will trial tau values from 15 ms to 25 ms in steps of 1 ms, producing a reconstruction 
(plus other diagnostics) for each trial. Only in this case are the "figures of merit" produced, which can be used 
determine which of the trial values are a better representation of the data. `tauclean` will invoke multiple processes 
(with the `multiprocessing` module) when searching.
