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
fmt = logging.Formatter(
    "%(asctime)s [pid %(process)d] :: %(name)-22s [%(lineno)d] :: %(levelname)s - %(message)s"
)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def keep_cleaning(
    on: np.ndarray,
    off: np.ndarray,
    threshold: float = 3.0,
) -> bool:
    """Compute residuals from data after clean component subtraction.
    The rms and baseline is determined from the "off-pulse" region of
    the profile, and the signal strength is calculated from the
    "on-pulse" region.

    Decides whether to keep cleaning or terminate the process.

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


def dm_delay(dm: float, lo: float, hi: float) -> float:
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
    nbins: int = 1024,
    period: float = 100.0,
    freq: float = 1.4,
    bw: float = 0.256,
    nchan: int = 1024,
    dm: float = 0.0,
    coherent: bool = False,
) -> float:
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


def get_offpulse_region(data: np.ndarray, windowsize: int = None) -> np.ndarray:
    """Determine the off-pulse window by minimising the integral over a range.
    i.e., because noise should integrate towards zero, finding the region that
    minimises the area mean it is representative of the noise level.

    Method taken from PyPulse (Lam, 2017. https://ascl.net/1706.011).

    :param data: original pulse profile [array-like]
    :param windowsize: window width (in bins) defining the trial regions to integrate [int]
    :return: a list of bins corresponding to the off-pulse region [array-like]
    """
    nbins = len(data)

    if windowsize is None:
        logger.debug("No off-pulse window size set, assuming 1/8 of profile.")
        windowsize = nbins // 8

    integral = np.zeros_like(data)
    for i in range(nbins):
        win = np.arange(i - windowsize // 2, i + windowsize // 2) % nbins
        integral[i] = np.trapz(data[win])

    minidx = np.argmin(integral)
    offpulse_win = np.arange(minidx - windowsize // 2, minidx + windowsize // 2) % nbins

    return offpulse_win


def reconstruct(
    ccs: np.ndarray,
    period: float = 100.0,
    rest_width: float = 1.0,
) -> np.ndarray:
    """Attempt to reconstruct the intrinsic pulse shape based on the clean component positions and amplitudes

    :param ccs: the clean components amplitudes [array-like]
    :param period: pulsar period (in ms) [float]
    :param rest_width: telescope restoring function width (in ms) [float]
    :return: a reconstruction of the intrinsic pulse profile [array-like]
    """

    nbins = len(ccs)
    x = period * np.linspace(0, 1, nbins)

    # Calculate the nominal effective time sampling, including effects of dispersion smearing in channels
    impulse_response = pbf.gaussian(x, x[x.size // 2], rest_width)

    # Reconstruct the intrinsic pulse profile by convolving the clean components with the impulse response
    # The impulse response has unit area, thus the fluence of the pulse should be conserved
    recon = np.convolve(ccs, impulse_response, mode="full") / np.sum(impulse_response)
    recon = recon[nbins // 2 : -nbins // 2 + 1]

    return recon


def clean(
    data: np.ndarray,
    tau: float,
    period: float = 100.0,
    rest_width: float = 1.0,
    gain: float = 0.05,
    threshold: float = 3.0,
    pbftype: str = "thin",
    iter_limit: int = None,
) -> dict:
    """The primary function of tauclean that actually does the deconvolution.

    :param data: original pulse profile [array-like]
    :param tau: scattering time scale to use when calculating the PBF [float]
    :param period: pulsar period (in ms) [float]
    :param rest_width: telescope restoring function width (in ms) [float]
    :param gain: a "loop gain" that is sued to scale the clean component amplitudes (usually 0.01-0.05) [float]
    :param threshold: threshold defining when to terminate the clean procedure [float]
    :param pbftype: type of pbf to use in the deconvolution [str]
    :param iter_limit: number of iterations after which to terminate the clean procedure regardless of convergence [int]
    :return: a dictionary containing various parameters and output of the cleaning process [dictionary]
    """

    nbins = len(data)
    bins = np.arange(nbins)

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

    # Determine the off-pulse by minimizing a windowed integrated quantity
    off_pulse_bins = get_offpulse_region(profile)

    # Define everything else of "on-pulse"
    on_pulse_bins = bins[np.logical_not(np.in1d(bins, off_pulse_bins))]

    # Calculate the noise level (assumes baseline removal happened already)
    off_pulse = profile[off_pulse_bins]
    on_pulse = profile[on_pulse_bins]
    init_off_rms = np.std(off_pulse)
    init_on_rms = np.std(on_pulse)

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
    logger.info(f"Initiating clean loop for tau={tau} ms")
    while loop:

        if (iter_limit is not None) and (niter >= iter_limit):
            logger.warning(f"Reached iteration limit for tau={tau} ms")
            break
        else:
            niter += 1

        # Identify the location and value of the maximum data point in the profile
        imax = np.argmax(profile)
        dmax = profile[imax]

        # Construct a single clean component on the same scale as the profile and assign its location and value
        temp_clean_comp = np.zeros_like(clean_components)
        temp_clean_comp[imax] = dmax * gain

        # Also add this component to the total list of clean components which will be used in the profile reconstruction
        logger.debug("Adding scaled cc to list")
        clean_components[imax] += dmax * gain

        # Create a profile component from the model PBF and the clean component
        # (Normalised to have area = 1)
        # TODO: do we need to have the restoring function convolved here, too?
        logger.debug("Constructing component to subtract from profile")
        component = (
            np.convolve(temp_clean_comp, filter_guess, mode="full") / filter_guess.sum()
        )

        # In this case, we have done the full convolution to accurately capture the shape of the PBF, now just grab
        # the profile-length
        component = component[:nbins]

        # Finally, subtract the component from the profile
        cleaned = profile - component

        # Calculate the on- and off-pulse regions and use them with the user-defined cleaning threshold to determine
        # whether the clean procedure should be terminated at this point
        logger.debug("Checking if cleaning should continue")
        on_pulse = cleaned[on_pulse_bins]
        off_pulse = cleaned[off_pulse_bins]
        loop = keep_cleaning(on_pulse, off_pulse, threshold=threshold)

        # Now replace the profile with the newly cleaned profile for the next iteration
        profile = np.copy(cleaned)

    logger.info(f"Clean loop terminated for tau={tau} ms")

    # After the clean procedure, calculate figures of merit and other information about the process
    logger.info("Computing figures of merit and reconstructing profile")
    n_unique = np.count_nonzero(clean_components)
    on_rms = on_pulse.std()
    off_rms = off_pulse.std()
    on_mean = on_pulse.mean()
    off_mean = off_pulse.mean()

    nf = fom.consistence(profile[on_pulse_bins], off_rms, off_mean=off_mean)
    fr = fom.positivity(profile, off_rms)
    gamma = fom.skewness(clean_components, period=period)
    recon = reconstruct(clean_components, period=period, rest_width=rest_width)

    return dict(
        profile=profile,
        init_off_rms=init_off_rms,
        init_on_rms=init_on_rms,
        nbins=nbins,
        nbins_on=len(on_pulse_bins),
        nbins_off=len(off_pulse_bins),
        tau=tau,
        pbftype=pbftype,
        niter=niter,
        cc=clean_components,
        ncc=n_unique,
        nf=nf,
        off_bins=off_pulse_bins,
        off_rms=off_rms,
        off_mean=off_mean,
        on_rms=on_rms,
        on_mean=on_mean,
        fr=fr,
        gamma=gamma,
        recon=recon,
        threshold=threshold,
    )
