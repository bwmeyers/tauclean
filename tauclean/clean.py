#!/usr/bin/env python3
"""
########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

from . import fom
from . import pbf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s :: %(name)s :: %(levelname)s - %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(fmt)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def keep_cleaning(on, off, threshold=3.0):
    """Compute residuals from data after clean component subtraction. The rms and baseline is determined from the
    "off-pulse" region of the profile, and the signal strength is calculated from the "on-pulse" region. Decides whether
    to keep cleaning or terminate the process.

    :param on: on-pulse time series [array-like]
    :param off: off-pulse time series [array-like]
    :param threshold: noise-threshold to determine when to stop cleaning [float]
    :return: whether to keep cleaning or not [boolean]
    """

    rms = np.std(off)
    mean = np.mean(off)

    datamax = np.max(on)

    limit = mean + threshold * rms

    return datamax > limit


def gaussian(x, mu, sigma):
    """A simple function that calculates a Gaussian shape over x

    :param x: independent variable [array-like]
    :param mu: the mean (position) of the Gaussian [float]
    :param sigma: the standard deviation (width) of the Gaussian [float]
    :return: a numerically evaluated Gaussian [array-like]
    """

    amp = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    g = amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    return g


def dm_delay(dm, lo, hi):
    """Calculate the dispersion delay between frequencies "lo" and "hi", both in GHz

    :param dm: dispersion measure of pulsar (in cm^-3 pc) [float]
    :param lo: the lowest frequency (in GHz) [float]
    :param hi: the highest frequency (in GHz) [float]
    :return: dispersion delay (in ms) [float]
    """

    k = 4.148808  # dispersion constant in ms
    delay = k * dm * (lo ** (-2) - hi ** (-2))

    return delay


def get_restoring_width(
    nbins, period=100.0, freq=1.4, bw=0.256, nchan=1024, dm=0.0, coherent=False
):
    """Estimate the restoring function width in milliseconds based on the time sampling and (possible) residual DM
    delay in channels

    :param nbins: number of bins in profile [int]
    :param period: pulsar rotation period in ms [float]
    :param freq: centre observing frequency in GHz [float]
    :param bw: observing bandwidth in GHz [float]
    :param nchan: number of frequency channels [int]
    :param dm: dispersion measure of pulsar (in cm^-3 pc) [float]
    :param coherent: boolean switch as to whether coherent dedispersion was used [boolean]
    :return: restoring function width in milliseconds [float]
    """
    time_sample = float(period) / nbins

    if coherent:
        # If coherently de-dispersed, then there will be no DM-smearing component to the response function
        dmdelay = 0.0
    else:
        # Figure out the dispersion smearing in the worst case (i.e. in the lowest channel), and then determine the
        # nominal width of the restoring function
        chan_bw = float(bw) / nchan
        lochan = freq - (bw / 2.0)
        hichan = lochan + chan_bw

        dmdelay = dm_delay(dm, lochan, hichan)

    # Restoring width in milliseconds
    restoring_width = np.sqrt(time_sample**2 + dmdelay**2)

    return restoring_width


def reconstruct(ccs, period=100.0, rest_width=1.0):
    """Attempt to reconstruct the intrinsic pulse shape based on the clean component positions and amplitudes

    :param ccs: the clean components amplitudes [array-like]
    :param period: pulsar period (in ms) [float]
    :param rest_width: telescope restoring function width (in ms) [float]
    :return: a reconstruction of the intrinsic pulse profile [array-like]
    """

    nbins = len(ccs)
    x = period * np.linspace(0, 1, nbins)

    # Calculate the nominal effective time sampling, including effects of dispersion smearing in channels
    impulse_response = gaussian(x, x[x.size // 2], rest_width)

    # Reconstruct the intrinsic pulse profile by convolving the clean components with the impulse response
    # The impulse response has unit area, thus the fluence of the pulse should be conserved
    recon = np.convolve(ccs, impulse_response, mode="full") / np.sum(impulse_response)
    recon = recon[nbins // 2 : -nbins // 2 + 1]

    return recon


def clean(
    data,
    tau,
    on_start=0,
    on_end=255,
    period=100.0,
    rest_width=1.0,
    gain=0.05,
    threshold=3.0,
    pbftype="thin",
    iter_limit=None,
):
    """The primary function of tauclean that actually does the deconvolution.

    :param data: original pulse profile [array-like]
    :param tau: scattering time scale to use when calculating the PBF [float]
    :param results: a list that is defined globally such that it can be written to by this function and remain available
    after completion (in the case of multiple processes, this will be a multiprocessing.Manager list object) [list]
    :param on_start: starting bin of the on-pulse region [int]
    :param on_end: end bin of the on-pulse region [int]
    :param period: pulsar period (in ms) [float]
    :param rest_width: telescope restoring function width (in ms) [float]
    :param gain: a "loop gain" that is sued to scale the clean component amplitudes (usually 0.01-0.05) [float]
    :param threshold: threshold defining when to terminate the clean procedure [float]
    :param pbftype: type of pbf to use in the deconvolution [str]
    :param iter_limit: number of iterations after which to terminate the clean procedure regardless of convergence [int]
    :return: a dictionary containing various parameters and output of the cleaning process [dictionary]
    """

    nbins = len(data)

    # Create an x-range that is much larger than the nominal pulsar period, so that the full effect of the PBF can be
    # modelled by evaluating  over the extended range and then folding the result on the pulsar period.
    x = np.linspace(0, 1, nbins) * period

    # Decide which PBF model to use based on the user input
    logger.debug(f"Computing PBF template for tau={tau} ms")
    if pbftype == "thin":
        filter_guess = pbf.thin(x, tau)
    elif pbftype == "thick":
        filter_guess = pbf.thick(x, tau)
    elif pbftype == "uniform":
        filter_guess = pbf.uniform(x, tau)
    elif pbftype == "thick_exp":
        filter_guess = pbf.thick_exp(x, tau)
    elif pbftype == "uniform_exp":
        filter_guess = pbf.uniform_exp(x, tau)
    else:
        logger.error(f"Invalid PBF type requested ({pbftype})")
        return None

    # Copy data into profile so that data is never actually touched in the process
    logger.debug(f"Estimating initial profile statistics for tau={tau} ms")
    profile = np.copy(data)
    on_pulse = profile[on_start:on_end]
    off_pulse = np.concatenate((profile[:on_start], profile[on_end:]))
    init_rms = np.std(off_pulse)

    # Pre-allocate an array the same size as profile where the clean component locations and amplitudes will be stored
    clean_components = np.zeros_like(profile)

    # Pre-calculate a delta-function
    delta = np.zeros_like(profile)
    delta[delta.size // 2] = 1.0

    # Initialise counters and boolean checks
    loop = True
    niter = 0

    # Start the clean procedure, terminating when either the iteration limit is reached,
    # or when the on-pulse residuals no longer hold data values above the 3-sigma off-pulse rms noise.
    logger.debug(f"Initiating clean loop for tau={tau} ms")
    while loop:

        if (iter_limit is not None) and (niter >= iter_limit):
            logger.warning(f"Reached iteration limit for tau={tau} ms")
            break
        else:
            niter += 1

        # Identify the location and value of the maximum data point in the profile
        imax = np.argmax(profile[on_start:on_end]) + on_start
        dmax = profile[imax]

        # Construct a single clean component on the same scale as the profile and assign its location and value
        temp_clean_comp = np.zeros_like(clean_components)
        temp_clean_comp[imax] = dmax * gain

        # Also add this component to the total list of clean components which will be used in the profile reconstruction
        clean_components[imax] += dmax * gain

        # Create a profile component from the model PBF and the clean component
        # (Normalised to have area = 1)
        component = np.convolve(temp_clean_comp, filter_guess, mode="full") / np.sum(
            filter_guess
        )

        # In this case, we have done the full convolution to accurately capture the shape of the PBF, now just grab
        # the profile-length
        component = component[:nbins]

        # Finally, subtract the component from the profile
        cleaned = profile - component

        # Calculate the on- and off-pulse regions and use them with the user-defined cleaning threshold to determine
        # whether the clean procedure should be terminated at this point
        on_pulse = cleaned[on_start:on_end]
        off_pulse = np.concatenate((cleaned[:on_start], cleaned[on_end:]))
        loop = keep_cleaning(on_pulse, off_pulse, threshold=threshold)

        # Now replace the profile with the newly cleaned profile for the next iteration
        profile = np.copy(cleaned)
    logger.debug(f"Clean loop terminated for tau={tau} ms")

    # After the clean procedure, calculate figures of merit and other information about the process
    logger.debug("Computing figures of merit and reconstructed profile")
    n_unique = np.count_nonzero(clean_components)
    off_rms = np.std(off_pulse)
    off_mean = np.mean(off_pulse)
    on_rms = np.std(on_pulse)
    nf = fom.consistence(profile, off_rms, off_mean=off_mean, onlims=(on_start, on_end))
    fr = fom.positivity(profile, off_rms)
    gamma = fom.skewness(clean_components, period=period)
    recon = reconstruct(clean_components, period=period, rest_width=rest_width)

    return dict(
        profile=profile,
        init_rms=init_rms,
        nbins=nbins,
        tau=tau,
        pbftype=pbftype,
        niter=niter,
        cc=clean_components,
        ncc=n_unique,
        nf=nf,
        off_rms=off_rms,
        off_mean=off_mean,
        on_rms=on_rms,
        fr=fr,
        gamma=gamma,
        recon=recon,
        threshold=threshold,
        on_start=on_start,
        on_end=on_end,
    )
