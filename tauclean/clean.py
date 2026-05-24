#!/usr/bin/env python3

import logging
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt

from . import fom
from . import pbf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter(
    "%(asctime)s [pid %(process)d] :: %(name)-22s [%(lineno)d] :: "
    "%(levelname)s - %(message)s"
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
    """Decide whether to keep cleaning or terminate the process.

    Computes residuals from data after clean component subtraction.
    The rms and baseline is determined from the "off-pulse" region of
    the profile, and the signal strength is calculated from the
    "on-pulse" region.

    :param on: On-pulse time series.
    :type on: np.ndarray
    :param off: Off-pulse time series.
    :type off: np.ndarray
    :param threshold: Noise threshold to determine when to stop cleaning.
    :type threshold: float
    :return: Whether to keep cleaning or not.
    :rtype: bool
    """
    rms = np.std(off)
    mean = np.mean(off)
    datamax = np.max(on)

    limit = mean + threshold * rms

    return datamax > limit


def dm_delay(dm: float, lo: float, hi: float) -> float:
    """Calculate the dispersion delay between frequencies "lo" and "hi",
    both in GHz.

    :param dm: Dispersion measure of pulsar (in cm^-3 pc).
    :type dm: float
    :param lo: The lowest frequency (in GHz).
    :type lo: float
    :param hi: The highest frequency (in GHz).
    :type hi: float
    :return: Dispersion delay (in ms).
    :rtype: float
    """

    k = 4.148808  # dispersion constant in ms
    delay = k * dm * (lo ** (-2) - hi ** (-2))

    return delay


def get_instrumental_response(
    profile: np.ndarray,
    pulse_period: float,
    r_dm_width: float,
    r_pb_width: float,
    r_av_width: float,
    r_pd_width: float,
    fast: bool = False,
) -> tuple[np.ndarray, float]:
    """Compute the instrumental response function.

    As noted in Bhat et al. 2003 in section 2.3.3, the instrumental response
    is important to consider and has multiple components. In particular:

    .. math::

        r(t) = r_{\\rm dm}(t) * r_{\\rm pb}(t) * r_{\\rm av}(t) * r_{\\rm pd}(t)

    where:
        * :math:`*` is the convolution operator
        * :math:`r_{\\rm dm}(t)` describes the DM smearing
        * :math:`r_{\\rm pb}(t)` describes the profile binning effects
        * :math:`r_{\\rm av}(t)` describes back-end time averaging
        * :math:`r_{\\rm pd}(t)` describes post-detection time averaging

    All of these are knowable before commencing deconvolution. A simplifying
    assumption (typically adequate) is that all contributions are approximately
    rectangular, resulting in trapezoidal or nearly-Gaussian convolutions.

    Since many of these functions affect the signal at much finer time sampling
    than the integrated profile, the resolution function must be created with
    high time resolution and then resampled.

    :param profile: The profile data for the original scattered pulse.
    :type profile: np.ndarray
    :param pulse_period: The pulsar period in ms.
    :type pulse_period: float
    :param r_dm_width: The DM smearing effective width.
    :type r_dm_width: float
    :param r_pb_width: The profile binning effective width.
    :type r_pb_width: float
    :param r_av_width: The effective width of back-end sampling/averaging.
    :type r_av_width: float
    :param r_pd_width: The effective width of post-detection averaging.
    :type r_pd_width: float
    :param fast: Use delta-function approximation for instrumental response.
        Defaults to False.
    :type fast: bool
    :return: The instrumental response function resampled at the given profile
        resolution, and the effective width of the response function.
    :rtype: tuple[np.ndarray, float]
    """
    upscale_factor = 10

    if fast:
        # Just use a delta function!
        decimated_resp = np.zeros_like(profile)
        decimated_resp[np.argmax(profile)] = 1
        resp_width = pulse_period / len(profile)
    else:
        # Compute each of the components as rectangular functions and
        # then convolve
        elements = [
            r
            for r in [r_dm_width, r_pb_width, r_av_width, r_pd_width]
            if r > 0
        ]
        logger.debug("Restoring elements = %s ms", elements)
        narrowest_element = min(elements)
        oversamp_nbins = int(
            upscale_factor * (pulse_period / narrowest_element)
        )
        oversamp_dt = pulse_period / oversamp_nbins
        x = np.linspace(0, 1, oversamp_nbins) * pulse_period
        logger.debug(
            "Upsampled nbins=%s & dt=%sms", oversamp_nbins, oversamp_dt
        )

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

        resp = resp / resp.sum()  # preserves fluence

        decimated_resp = resp[:: oversamp_nbins // profile.size]
        decimated_resp = decimated_resp / np.trapz(
            x=np.linspace(0, 1, len(decimated_resp)) * pulse_period,
            y=decimated_resp,
        )

        # At this point, we can approximate the width by assuming the total
        # response is a top-hat function so then the width is the total area
        # divided by the peak
        resp_width = (np.trapz(dx=1, y=resp) / resp.max()) * oversamp_dt

    return decimated_resp, resp_width


def get_restoring_function(
    profile: np.ndarray,
    pulse_period: float,
    inst_resp_width: float,
) -> np.ndarray:
    """Generate the restoring function for pulse shape reconstruction.

    This follows the logic as given in Bhat et al. (2003), section 2.3.4.

    :param profile: The profile data for the original scattered pulse.
    :type profile: np.ndarray
    :param pulse_period: The pulsar period in ms.
    :type pulse_period: float
    :param inst_resp_width: The instrumental response effective width in ms.
    :type inst_resp_width: float
    :return: A Gaussian pulse to use for profile reconstruction.
    :rtype: np.ndarray
    """
    upfact = 10
    nbins = upfact * len(profile)
    x = pulse_period * np.linspace(-0.5, 0.5, nbins)
    rest_func = pbf.gaussian(x, mu=0, sigma=inst_resp_width)
    # The above gaussian function is already normalised.

    return rest_func[::upfact]


def get_offpulse_region(
    data: np.ndarray, windowsize: int | None = None
) -> np.ndarray:
    """Determine the off-pulse window by minimizing the integral over a range.

    Because noise should integrate towards zero, finding the region that
    minimizes the area mean it is representative of the noise level.
    Method taken from PyPulse (Lam, 2017. https://ascl.net/1706.011).

    :param data: The original pulse profile.
    :type data: np.ndarray
    :param windowsize: Window width (in bins) defining the trial regions to
        integrate. Defaults to 1/8 of profile length.
    :type windowsize: int | None
    :return: Bins corresponding to the off-pulse region.
    :rtype: np.ndarray
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
    offpulse_win = (
        np.arange(minidx - windowsize // 2, minidx + windowsize // 2) % nbins
    )

    return offpulse_win


def reconstruct(
    ccs: np.ndarray,
    rest_func: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct the intrinsic pulse shape from clean components.

    :param ccs: The clean components amplitudes.
    :type ccs: np.ndarray
    :param rest_func: The restoring function to reconstruct with.
        If None, uses a delta function.
    :type rest_func: np.ndarray | None
    :return: A reconstruction of the intrinsic pulse profile.
    :rtype: np.ndarray
    """
    # Calculate the nominal effective time sampling, including effects
    # of dispersion smearing in channels
    if rest_func is None:
        logger.warning(
            "No valid restoring function provided, using a delta function"
        )
        rest_func = np.zeros_like(ccs)
        rest_func[rest_func.size // 2] = 1

    # Reconstruct the intrinsic pulse profile by convolving the clean
    # components with the restoring function. The restoring function has unit
    # area, thus the fluence of the pulse should be conserved.
    recon = convolve(ccs, rest_func, mode="same")

    return recon / recon.max()


def clean(
    data: np.ndarray,
    tau: float,
    period: float = 100.0,
    rest_func: np.ndarray | None = None,
    inst_resp_func: np.ndarray | None = None,
    gain: float = 0.05,
    threshold: float = 3.0,
    pbftype: str = "thin",
    iter_limit: int = 1000,
    onpulse_estimator: str = "auto",
) -> dict:
    """Perform CLEAN deconvolution on pulse profile data.

    This is the primary function for the deconvolution (CLEAN) procedure.
    It gathers the figures of merit for each converged cycle and returns
    a dictionary of results and supplemental information.

    :param data: The original pulse profile.
    :type data: np.ndarray
    :param tau: Scattering time scale for calculating the PBF (in ms).
    :type tau: float
    :param period: The pulsar spin period in ms. Defaults to 100.0.
    :type period: float
    :param rest_func: The restoring function for profile reconstruction.
        If None, uses a Gaussian with FWHM = 2x profile time resolution.
    :type rest_func: np.ndarray | None
    :param inst_resp_func: The instrumental response function during
        deconvolution. If None, uses a delta function.
    :type inst_resp_func: np.ndarray | None
    :param gain: Loop gain for scaling clean component amplitudes
        (typically << 1). Defaults to 0.05.
    :type gain: float
    :param threshold: Noise-level threshold for terminating the clean
        procedure. Defaults to 3.0.
    :type threshold: float
    :param pbftype: Type of pulse-broadening function to use. Options are
        'thin', 'thick', 'uniform', 'thick_exp', 'uniform_exp'.
        Defaults to 'thin'.
    :type pbftype: str
    :param iter_limit: Maximum iterations before terminating regardless of
        convergence. Defaults to 1000.
    :type iter_limit: int
    :param onpulse_estimator: A string of space-separated integers representing
        the range of profile bins defining an on-pulse region, or "auto" to
        automatically determine on/off-pulse regions. Defaults to "auto".
    :type onpulse_estimator: str
    :return: Dictionary containing various parameters and outputs from the
        cleaning process, including clean components, figures of merit, and
        convergence information.
    :rtype: dict
    """

    nbins = len(data)
    bins = np.arange(nbins)
    prof_dt = period / nbins

    # Create an x-range that is much larger than the nominal pulsar period,
    # so that the full effect of the PBF can be modelled by evaluating  over
    # the extended range and then folding the result on the pulsar period.
    x = np.linspace(0, 1, nbins) * period

    # Decide which PBF model to use based on the user input
    logger.debug("Computing PBF template for tau=%g ms", tau)
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
        logger.error("Invalid PBF type requested (%s)", pbftype)
        return {}

    # Copy data into profile so that data is never altered in the process
    logger.debug("Estimating initial profile statistics for tau=%g ms", tau)
    profile = np.copy(data)

    if onpulse_estimator == "auto":
        # Determine the off-pulse by minimizing a windowed integrated quantity
        logger.debug("Using 'auto' off-pulse estimator.")
        off_pulse_bins = get_offpulse_region(profile)
        on_pulse_bins = bins[np.logical_not(np.in1d(bins, off_pulse_bins))]
    else:
        # Accept the user-defined region as the on-pulse
        logger.debug("Using user-defined on-pulse region.")
        on_start, on_end = onpulse_estimator.split(" ")
        on_pulse_bins = bins[int(on_start) : int(on_end)]
        off_pulse_bins = bins[np.logical_not(np.in1d(bins, on_pulse_bins))]

    # Calculate the noise level (assumes baseline removal happened already)
    baseline = np.mean(profile[off_pulse_bins])
    profile = profile - baseline
    off_pulse = profile[off_pulse_bins]
    on_pulse = profile[on_pulse_bins]
    init_off_rms = np.std(off_pulse)
    init_on_rms = np.std(on_pulse)

    # Pre-allocate an array the same size as profile where the clean component
    # locations and amplitudes will be stored
    clean_components = np.zeros_like(profile)

    # Pre-calculate a delta-function
    delta = np.zeros_like(profile)
    delta[delta.size // 2] = 1.0

    # Ensure that the restoring and instrumental response functions are set.
    logger.debug("Checking restoring and instrumental response functions...")
    if inst_resp_func is None:
        logger.warning(
            "No valid instrumental response function defined. "
            "Assuming a delta function."
        )
        # Just assume a delta function if nothing explicit is provided
        inst_resp_func = delta

    logger.debug(
        "instr. response function area: %f (should be 1)",
        np.trapz(x=x, y=inst_resp_func),
    )

    if rest_func is None:
        logger.warning(
            "No valid restoring function defined. Assuming a Gaussian with "
            "FWHM = 2x profile time resolution."
        )
        # Assume a Gaussian with a std. dev. corresponding to the twice
        # profile time resolution
        rest_func = pbf.gaussian(x, x[x.size // 2], 2 * prof_dt)

    logger.debug(
        "restoring function area: %f (should be 1)", np.trapz(x=x, y=rest_func)
    )

    # Pre-compute one of the required convolutions required
    preconv = convolve(inst_resp_func, filter_guess, mode="full")
    preconv = preconv[nbins // 2 : -nbins // 2 + 1]
    preconv = preconv / np.trapz(preconv)

    # Initialise counters and boolean checks
    loop = True
    niter = 0

    # Start the clean procedure, terminating when either the iteration limit
    # is reached, or when the on-pulse residuals no longer hold data values
    # above the 3-sigma off-pulse rms noise.
    logger.debug("Initiating clean loop for tau=%g ms", tau)
    while loop:
        if (iter_limit is not None) and (niter >= iter_limit):
            logger.warning("Reached iteration limit for tau=%g ms", tau)
            break
        else:
            niter += 1

        # Identify the location and value of the maximum data point in
        # the profile
        imax = np.argmax(profile)
        dmax = profile[imax]
        cc_amp = dmax * gain

        # Construct a single clean component on the same scale as the profile
        # and assign its location and value
        temp_clean_comp = np.zeros_like(clean_components)
        temp_clean_comp[imax] = cc_amp

        # Also add this component to the total list of clean components which
        # will be used in the profile reconstruction
        logger.debug("Adding scaled cc to list")
        clean_components[imax] += cc_amp

        # Create a profile component from the model PBF and the clean component
        # (Normalised to have area = 1)
        logger.debug(
            "Constructing component to subtract from profile for tau=%g ms",
            tau,
        )
        # The actual quantity to subtract is:
        #   convolve(convolve(scaled_clean_comp, inst_resp_func), pbf))
        # The inst_resp_func and pbf components are pre-computed before the
        # CLEAN cycles.
        #
        # By the associativity of convolution, we can instead have
        #   convolve(scaled_clean_comp, convolve(inst_resp_func, pbf))
        # which means the interior convolution can occur outside of the CLEAN
        # cycle.

        # NOTE: Annoyingly, convolution with "same" appears to not exactly keep the index position
        # information in the way we need it to, so we do the full convolution and then cut the windows
        # as required. Even this, though, does not always ensure that the "pbf" part (preconv)
        # has its peak at index 0, which is what should be expected. Thus, we have an additional
        # "rolling" operation to shift the component to the correct location.
        component1 = convolve(temp_clean_comp, preconv, mode="full")
        component1 = component1[nbins // 2 : -nbins // 2 + 1]
        didx = imax - np.argmax(component1)
        component = np.roll(component1, didx)
        # Scale the component based on the off-pulse noise and loop-gain so that the S/N doesn't
        # dictate the run time quite as much (i.e., higher S/N = more required iterations without this)
        component = init_off_rms * (component / component.max())

        if component.size != profile.size:
            logger.error(
                "CLEAN component shape (%d) is not the same as the profile (%d)!",
                component.size,
                profile.size,
            )
            raise ValueError

        if np.argmax(component) != imax:
            logger.error(
                "Component alignment error - convolved subtraction component "
                "position (%d) does not match data maximum (%d)!",
                np.argmax(component),
                imax,
            )
            logger.error(
                "Error on Iteration number = %d/%d for tau=%g ms",
                niter,
                iter_limit,
                tau,
            )
            plt.plot(1e-5 + data, color="k", alpha=0.2)
            plt.plot(1e-5 + cleaned, label="profile")
            plt.plot(1e-5 + temp_clean_comp, label="cc")
            plt.plot(1e-5 + preconv, ls="--", label="inst_resp ^ pbf")
            plt.plot(1e-5 + component, ls="-.", label="component")
            plt.plot(1e-5 + component1, ls="-.", label="component1")
            plt.axvline(imax, ls="--", color="k")
            plt.axvline(np.argmax(component), ls=":", color="r")
            plt.axhline(threshold * init_off_rms, ls=":", color="k")
            plt.ylim(-1.1 * init_off_rms, 1.1 * cleaned.max())
            plt.xlim(0, len(data))
            plt.title(
                f"Iteration={niter}/{iter_limit}  "
                f"Current imax={imax}  "
                f"conv.comp idx={np.argmax(component1)}"
            )
            plt.legend()
            plt.show()
            raise ValueError

        # Finally, subtract the component from the profile
        cleaned = profile - component

        # if niter % 100 == 0 or niter == 1:  # or niter == 77981:
        #     plt.plot(1e-5 + data, color="k", alpha=0.2)
        #     plt.plot(1e-5 + cleaned, label="profile")
        #     plt.plot(1e-5 + temp_clean_comp, label="cc")
        #     plt.plot(1e-5 + preconv, ls="--", label="inst_resp ^ pbf")
        #     plt.plot(1e-5 + component, ls="-.", label="component")
        #     plt.plot(1e-5 + component1, ls="-.", label="component1")
        #     # plt.plot(filter_guess, ls=":", color="r", label="pbf")
        #     plt.axvline(imax, ls="--", color="k")
        #     plt.axvline(np.argmax(component), ls=":", color="r")
        #     plt.axhline(1e-5 + threshold * init_off_rms, ls=":", color="k")
        #     # plt.yscale("log")
        #     plt.ylim(-1.1 * init_off_rms, 1.1 * cleaned.max() + 1e-5)
        #     plt.xlim(0, len(data))
        #     plt.title(
        #         f"Iteration={niter}/{iter_limit}  Current imax={imax} conv. comp idx={np.argmax(component1)}"
        #     )
        #     plt.legend()
        #     plt.show()

        # Calculate the on- and off-pulse regions and use them with the user-defined cleaning threshold to determine
        # whether the clean procedure should be terminated at this point
        logger.debug("Checking if cleaning should continue for tau=%s ms", tau)
        on_pulse = cleaned[on_pulse_bins]
        off_pulse = cleaned[off_pulse_bins]
        loop = keep_cleaning(on_pulse, off_pulse, threshold=threshold)

        # Now replace the profile with the newly cleaned profile for the next iteration
        profile = cleaned

    if niter <= 1:
        logger.warning(
            "Clean cycle only lasted 1 iteration - something probably went wrong!"
        )
    elif niter >= iter_limit:
        logger.warning(
            f"Clean cycle terminated prematurely for tau={tau:g} ms"
        )
    else:
        logger.debug("Clean cycle terminated successfully for tau=%g ms", tau)

    # After the clean procedure, calculate figures of merit and other information about the process
    logger.debug(
        f"Generating FOM and reconstructed profile for tau={tau:g} ms"
    )
    n_unique = np.count_nonzero(clean_components)

    # Since `profile`, `on_pulse`, and `off_pulse` are all updated in the loop, we can access
    # them here to see the final result
    on_rms = on_pulse.std()
    off_rms = off_pulse.std()
    on_mean = on_pulse.mean()
    off_mean = off_pulse.mean()
    total_mean = profile.mean()
    total_rms = profile.std()

    # since the profile is already mean-subtracted, we can just use the defaults
    # of the consistence function
    nf = fom.consistence(
        profile=profile,
        off_rms=off_rms,
    )
    logger.debug("nf (tau=%g ms) = %s", tau, nf)

    fr = fom.positivity(profile, off_rms)
    logger.debug("fr (tau=%g ms) = %s", tau, fr)

    gamma = fom.skewness(clean_components, pulsar_period=period)
    logger.debug("gamma (tau=%g ms) = %s", tau, gamma)

    logger.debug("Reconstructing profile for tau=%g ms", tau)
    recon = reconstruct(clean_components, rest_func=rest_func)

    return dict(
        profile=profile,
        init_off_rms=init_off_rms,
        init_on_rms=init_on_rms,
        nbins=nbins,
        nbins_on=len(on_pulse_bins),
        nbins_off=len(off_pulse_bins),
        rest_func=rest_func,
        inst_resp_func=inst_resp_func,
        tau=tau,
        pbftype=pbftype,
        niter=niter,
        cc=clean_components,
        ncc=n_unique,
        nf=nf,
        off_bins=off_pulse_bins,
        on_bins=on_pulse_bins,
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
