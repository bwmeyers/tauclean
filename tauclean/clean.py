#!/usr/bin/env python3
"""
########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
"""
import logging
import numpy as np
from scipy.signal import convolve
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


def get_inst_resp(
    profile: np.ndarray,
    pulse_period: float,
    r_dm_width: float,
    r_pb_width: float,
    r_av_width: float,
    r_pd_width: float,
    fast: bool = False,
) -> np.ndarray:
    """As noted in Bhat et al. 2003 in section 2.3.3, the instrumental response
    is important to consider and has multiple components. In particular,
        r(t) = convolve( r_dm(t), convolve( r_pb(t), convolve( r_av(t), r_pd(t) ) ) )
    where
        - r_dm(t) describes the DM smearing,
        - r_pb(t) describes the profile binning effects,
        - r_av(t) describes back-end time averaging, and
        - r_pd(t) describes post-detection time averaging

    In principle, all of these are knowable before commencing deconvolution.
    A simplifying assumption (that is typically adequate) is that all of these
    contributions are approximately rectangular. Thus, the corresponding convolutions
    will be trapezoidal (if there are clearly dominant contributions), or nearly-Gaussian
    if the case that each contribution is similar.

    Given that many of these functions affect the signal at much finer time sampling than
    the integrated profile, this resolution function r(t) must be created with a
    time resolution much greater than the narrowest factor and then resampled.
    """
    upscale_factor = 10
    if fast:
        resp = np.zeros_like(profile)
        resp[np.argmax(profile)] = 1
    else:
        # Compute each of the components as rectangular functions and then convolve
        elements = [
            r for r in [r_dm_width, r_pb_width, r_av_width, r_pd_width] if r > 0
        ]
        logger.debug(f"Restoring elements = {elements} ms")
        narrowest_element = min(elements)
        oversamp_nbins = int(upscale_factor * (pulse_period / narrowest_element))
        oversamp_dt = pulse_period / oversamp_nbins
        logger.debug(f"Upsampled nbins={oversamp_nbins} & dt={oversamp_dt} ms")

        resp = np.zeros(oversamp_nbins)

        for el in elements:
            if np.isfinite(el):
                contrib = np.zeros(oversamp_nbins)
                width_bins = el // oversamp_dt
                slc = slice(
                    int(oversamp_nbins / 2 - width_bins / 2),
                    int(oversamp_nbins / 2 + width_bins / 2),
                )
                contrib[slc] = 1
                if resp.sum() <= 0:
                    resp = resp + contrib
                else:
                    resp = convolve(resp, contrib, mode="same")

        resp = resp / resp.sum()
    decimated_resp = resp[:: oversamp_nbins // profile.size]
    decimated_resp = decimated_resp / np.trapz(dx=1, y=decimated_resp)

    # At this point, we can approximate the width by assuming the total response
    # is a top-hat function so then the width is the total area divided by the peak
    resp_width = (np.trapz(dx=1, y=resp) / resp.max()) * oversamp_dt

    return decimated_resp, resp_width


def get_restoring_function(
    profile: np.ndarray,
    pulse_period: float,
    inst_resp_width: float,
) -> np.ndarray:
    nbins = len(profile)
    x = pulse_period * np.linspace(0, 1, nbins)
    return pbf.gaussian(x, mu=x[np.argmax(profile)], sigma=inst_resp_width)


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
    rest_func: np.ndarray = None,
) -> np.ndarray:
    """Attempt to reconstruct the intrinsic pulse shape based on the clean component positions and amplitudes

    :param ccs: the clean components amplitudes [array-like]
    :param rest_func: restoring function to reconstruct with [array-like]
    :return: a reconstruction of the intrinsic pulse profile [array-like]
    """

    nbins = len(ccs)

    # Calculate the nominal effective time sampling, including effects of dispersion smearing in channels
    if not rest_func:
        logger.warning("No valid restoring function provided, using a delta function")
        rest_func = np.zeros_like(ccs)
        rest_func[rest_func.size // 2] = 1

    # Reconstruct the intrinsic pulse profile by convolving the clean components with the impulse response
    # The impulse response has unit area, thus the fluence of the pulse should be conserved
    recon = convolve(ccs, rest_func, mode="same") / rest_func.sum()

    return recon


def clean(
    data: np.ndarray,
    tau: float,
    period: float = 100.0,
    rest_func: np.ndarray = None,
    inst_resp_func: np.ndarray = None,
    gain: float = 0.05,
    threshold: float = 3.0,
    pbftype: str = "thin",
    iter_limit: int = None,
) -> dict:
    """The primary function of tauclean that actually does the deconvolution.

    :param data: original pulse profile [array-like]
    :param tau: scattering time scale to use when calculating the PBF [float]
    :param period: pulsar period (in ms) [float]
    :param rest_func: the restoring function to use for profile reconstruction [array-like]
    :param inst_resp_func: the instrumental response function (effective time resolution)
        to use during deconvolution cycles [array-like]
    :param gain: a "loop gain" that is sued to scale the clean component amplitudes (usually 0.01-0.05) [float]
    :param threshold: threshold defining when to terminate the clean procedure [float]
    :param pbftype: type of pbf to use in the deconvolution [str]
    :param iter_limit: number of iterations after which to terminate the clean procedure
        regardless of convergence [int]
    :return: a dictionary containing various parameters and output of the cleaning process [dictionary]
    """

    nbins = len(data)
    bins = np.arange(nbins)
    prof_dt = period / nbins

    # Create an x-range that is much larger than the nominal pulsar period, so that the full effect of the PBF can be
    # modelled by evaluating  over the extended range and then folding the result on the pulsar period.
    x = np.linspace(0, 1, nbins) * period

    # Decide which PBF model to use based on the user input
    logger.debug(f"Computing PBF template for tau={tau:g} ms")
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
    logger.debug(f"Estimating initial profile statistics for tau={tau:g} ms")
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

    # Ensure that the restoring and instrumental response functions are set.
    logger.debug("Checking restoring and instrumental response functions...")
    if inst_resp_func is None:
        logger.warning(
            "No valid instrumental response function defined. Assuming a delta function."
        )
        # Just assume a delta function if nothing explicit is provided
        inst_resp_func = delta
    logger.debug("Generating instrumental+pbf convolution.")
    inst_pbf_conv = convolve(inst_resp_func, filter_guess, mode="same")
    inst_pbf_conv = inst_pbf_conv / inst_pbf_conv.sum()
    plt.plot(inst_resp_func)
    plt.plot(filter_guess)
    plt.show()

    if rest_func is None:
        logger.warning(
            "No valid restoring function defined. Assuming a Gaussian with FWHM = 2x profile time resolution."
        )
        # Assume a Gaussian with a FWHM corresponding to the profile time resolution
        rest_func = pbf.gaussian(x, x[x.size // 2], 2 * prof_dt / 2.355)

    # Initialise counters and boolean checks
    loop = True
    niter = 0

    # Start the clean procedure, terminating when either the iteration limit is reached,
    # or when the on-pulse residuals no longer hold data values above the 3-sigma off-pulse rms noise.
    logger.info(f"Initiating clean loop for tau={tau:g} ms")
    while loop:

        if (iter_limit is not None) and (niter >= iter_limit):
            logger.warning(f"Reached iteration limit for tau={tau:g} ms")
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
        logger.debug(
            f"Constructing component to subtract from profile for tau={tau:g} ms"
        )
        # The actual quantity to subtract is: convolve(scaled_clean_comp, convolve(inst_resp_func, pbf))
        # But the second term in the first convolution call is pre-computed
        component = convolve(temp_clean_comp, inst_pbf_conv, mode="valid")
        component = component / component.sum()
        plt.plot(component)
        plt.plot(data)
        plt.show()

        # In this case, we have done the full convolution to accurately capture the shape of the PBF, now just grab
        # the profile-length
        # component = component[:nbins]

        # Finally, subtract the component from the profile
        cleaned = profile - component

        # Calculate the on- and off-pulse regions and use them with the user-defined cleaning threshold to determine
        # whether the clean procedure should be terminated at this point
        logger.debug(f"Checking if cleaning should continue for tau={tau} ms")
        on_pulse = cleaned[on_pulse_bins]
        off_pulse = cleaned[off_pulse_bins]
        loop = keep_cleaning(on_pulse, off_pulse, threshold=threshold)

        # Now replace the profile with the newly cleaned profile for the next iteration
        profile = np.copy(cleaned)

    logger.debug(f"Clean loop terminated for tau={tau:g} ms")
    if niter <= 1:
        logger.warning(
            "Clean cycle only lasted 1 iteration - something probably went wrong!"
        )

    # After the clean procedure, calculate figures of merit and other information about the process
    logger.debug(f"Generating FOM and reconstructed profile for tau={tau:g} ms")
    n_unique = np.count_nonzero(clean_components)

    # Since `profile`, `on_pulse`, and `off_pulse` are all updated in the loop, we can access
    # them here to see the final result
    on_rms = on_pulse.std()
    off_rms = off_pulse.std()
    on_mean = on_pulse.mean()
    off_mean = off_pulse.mean()
    total_mean = profile.mean()
    total_rms = profile.std()

    nf = fom.consistence(
        profile[on_pulse_bins],
        off_rms,
        off_mean=off_mean,
        threshold=threshold,
    )
    fr = fom.positivity(profile, off_rms)
    gamma = fom.skewness(clean_components, period=period)
    recon = reconstruct(clean_components, rest_func=rest_func)

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
        total_mean=total_mean,
        total_rms=total_rms,
        fr=fr,
        gamma=gamma,
        recon=recon,
        threshold=threshold,
    )
