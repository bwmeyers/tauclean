#!/usr/bin/env python3
"""
########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
"""

import logging
import numpy as np
from scipy.signal import savgol_filter, find_peaks

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter(
    "%(asctime)s [pid %(process)d] :: %(name)-22s [%(lineno)d] :: %(levelname)s - %(message)s"
)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def consistence(
    profile: np.ndarray,
    off_rms: float,
    off_mean: float = 0,
    threshold: float = 3.0,
) -> int:
    """The number of residual points in the deconvolved profile that are
    consistent with the off-pulse rms is another indicator of how well
    the CLEAN procedure has done. The number will should in principle be
    a plateau or start to grow until the optimal tau is reached, at which
    point it will start to decline as over-subtraction starts to become
    significant.

    Defined in Bhat et al. (2003) in the third-last paragraph of Section 2.5.3

    :param profile: The residual profile after the CLEAN process has terminated.
    :type profile: np.ndarray
    :param off_rms: The off-pulse rms noise.
    :type off_rms: float
    :param off_mean: The off-pulse mean value, defaults to 0.
    :type off_mean: float
    :param threshold: The threshold with which to compare for significance,
        defaults to 3.0.
    :type threshold: float
    :return: The number of points in the CLEANed profile that are consistent
        with the off-pulse noise.
    :rtype: int
    """

    # Calculate the number of profile points that are consistent with
    # the n-sigma noise level of the off-pulse region
    nf = np.sum(abs(profile - off_mean) <= threshold * off_rms)

    return nf


def positivity(
    res: np.ndarray,
    off_rms: float,
    m: float = 1.0,
    x: float = 1.5,
) -> float:
    """The positivity figure of merit used to help decide on the quality
    of the CLEAN procedure. Will start off reasonably low and begin to
    rapidly increase as over-subtraction starts to become significant.

    Defined by Bhat et al. 2003 in their eqs. 10 and 11.

    :param res: The residuals after the CLEAN process has terminated.
    :type res: np.ndarray
    :param off_rms: The off-pulse rms value to be used as a threshold.
    :type off_rms: float
    :param m: A scale-factor (or weight) that is of order unity, defaults to 1.0.
    :type m: float
    :param x: A threshold (units of off_rms) defined to penalise the positivity
        if there are residuals more negative than this, defaults to 1.5.
    :type x: float
    :return: The positivity figure of merit.
    :rtype float:
    """

    u = np.zeros_like(res)
    # When the residual is less than x * rms, turn on the step-function.
    # This means that only those points that have been over subtracted (due to a poor choice in PBF)
    # contribute to this parameter.
    u[res < -x * off_rms] = 1

    if np.all(res == 0):
        # Safe from a plotting perspective as NaNs are ignored.
        # Also easier to identify and deal with with other functions.
        return np.nan

    f_r = (m / (len(res) * off_rms**2)) * np.sum(u * res**2)

    return f_r


def skewness(
    ccs: np.ndarray,
    pulsar_period: float = 100.0,
) -> float:
    """The skewness of the clean components gives a figure of merit
    that describes how asymmetric the CLEANed profile is. For a
    well-matched PBF and high signal-to-noise data, the clean component
    distribution should be approximately symmetric (Gaussian-like).
    [NB: This metric is really only sensible for a simple pulse profile
     - those with multiple distribution components are not well described.]

    Defined by Bhat et al. 2003 in their eqs. 12, 13 and 14.

    :param ccs: a list of component (delta-function) amplitudes produced
        at the end of the CLEAN procedure.
    :type ccs: np.ndarray
    :param pulsar_period: The pulsar period (in ms), defaults to 100.0.
    :type pulsar_period:
    :return: The skewness figure of merit.
    :rtype: float
    """

    # Compute the times for each clean components based on the assumption
    # that the clean component array is the same shape as the CLEANed profile
    cc_times = pulsar_period * np.linspace(0, 1, len(ccs))

    # The moments defined are the equivalent to a weighted average, thus
    moment_1 = np.average(cc_times, weights=ccs)
    moment_2 = np.average((cc_times - moment_1) ** 2, weights=ccs)
    moment_3 = np.average((cc_times - moment_1) ** 3, weights=ccs)

    if np.count_nonzero(ccs) == 1:
        # The function is symmetric by definition at this stage, but
        # moment_2 = 0 so we'll get an error if we try to calculate the
        # skewness in the usual way
        logger.warning("Clean components skewness is undefined. Setting to 0.")
        gamma = 0
    else:
        gamma = moment_3 / (moment_2**1.5)

    return gamma


def get_best_tau_jerk(
    results: list[dict],
    norm_fom_peak_height: float | None = None,
    norm_fom_peak_prominance: float | None = None,
    smoothing_window_size: int | None = None,
    fom_weights: dict | None = None,
) -> tuple[float]:
    """Estimate the uncertainty of each tau trial value by determining
    the value of tau that results in the maximum peak of the FOM's 3rd derivative.

    Some of the FOMs are degenerate or less-reliable, and thus are weighted when
    combined to form an overall average estimate.

    :param results: A list of dictionaries containing the output from the
        deconvolution process, one per trial tau.
    :type results: list[dict]
    :param norm_fom_peak_height: A height parameter to determine peaks from
        normalised FOM derivatives. In the case of multiple peaks above this
        threshold, the first is taken as the best guess. Defaults to 0.8.
    :type norm_fom_peak_height: float
    :param smoothing_window_size: The window size to use when computing smoothed
        FOM derivatives. If no value is provided, a size is calculated based on the
        number of FOM measurements. Defaults to None.
    :type smoothing_window_size: int | None
    :param fom_weights: The corresponding weights for each FOM to use when computing
        the average best tau value. If no dictionary is given, or required keys are
        missing, a default weighting scheme is used. Defaults to None.
    :type fom_weights: dict | None
    :return: The best estimated tau and uncertainty based on FOMs.
    :rtype: tuple[float, float]
    """
    taus = np.array([a["tau"] for a in results])
    f_r = np.array([a["fr"] for a in results])
    gamma = np.array([a["gamma"] for a in results])
    f_c = (f_r + gamma) / 2.0
    r_sigma = np.array([a["total_rms"] for a in results]) / np.array(
        [a["init_off_rms"] for a in results]
    )
    r_phi = np.array([a["nf"] for a in results]) / np.array(
        [a["nbins"] for a in results]
    )

    foms = [
        {
            "name": "f_r",
            "values": np.array(f_r),
            "label": r"$f_r$",
            "use_jerk": True,
            "alt_operation": np.argmin,
        },
        {
            "name": "gamma",
            "values": np.array(gamma),
            "label": r"$\Gamma$",
            "use_jerk": True,
            "alt_operation": np.argmin,
        },
        {
            "name": "f_c",
            "values": np.array(f_c),
            "label": r"$f_c = (f_r + \Gamma)/2$",
            "use_jerk": False,
            "alt_operation": None,
        },
        {
            "name": "r_sigma",
            "values": np.array(r_sigma),
            "label": r"$r_\sigma = \sigma_{\rm offc}/\sigma_{\rm off}$",
            "use_jerk": False,
            "alt_operation": np.argmin,
        },
        {
            "name": "r_phi",
            "values": np.array(r_phi),
            "label": r"$r_\phi=N_f / N_{\rm tot}$",
            "use_jerk": False,
            "alt_operation": np.argmax,
        },
    ]
    fom_names = [f["name"] for f in foms]

    # Set FOMs to use for automatic best-fit guess and error approximation
    default_fom_weights = dict(f_r=1.0, gamma=0.2, f_c=0.0, r_sigma=0.5, r_phi=0.5)
    if fom_weights is None:
        fom_weights = default_fom_weights
    else:
        # Check to make sure required keys are present, if not, adding them.
        for key in default_fom_weights.keys():
            if key in fom_weights.keys():
                if not (
                    isinstance(fom_weights[key], float) and 0 <= fom_weights[key] <= 1
                ):
                    logger.warning(
                        f"Weight for FOM={key} is not a float in the range 0 <= x <= 1! Setting to 0."
                    )
                    fom_weights[key] = 0
            else:
                fom_weights.update({f"{key}": default_fom_weights[key]})

    # Smoothing and derivative kernel order
    savgol_polyorder = 3
    savgol_derorder = 3

    # Some useful check markers
    multi_peak_flag = 0

    # Use the FOM and their derivatives to estimate the best match
    fom_tau_estimates = []
    for fom in foms:
        # For the cases of r_sigma and r_phi, we actually want to use a heuristic
        # functional evaluation of the values to determine the "best fit"
        if fom["name"] in ["r_phi", "r_sigma"]:
            fn = fom["alt_operation"]
            best_tau_fom = taus[fn(fom["values"])]
            fom_tau_estimates.append(best_tau_fom)
            logger.info(
                f"Best tau from metric={fom['name']:7s} is: {np.squeeze(best_tau_fom):.2f} ms"
            )
        else:

            logger.debug(f"Finding 'best' tau from FOM={fom['name']} via 3rd deriv.")

            # The smoothing window size must be greater than the polynomial order used in the filter
            if smoothing_window_size is None:
                logger.debug(
                    "No savgol_filter window size provided, choosing sensible value based on FOM series length."
                )
                smoothing_window_size = len(fom["values"]) // 8
                if smoothing_window_size <= savgol_polyorder:
                    smoothing_window_size = savgol_polyorder + 1
                logger.debug(f"Window size set to {smoothing_window_size} bins")

            # The smoothing window must be smaller than the total number of measurements
            if smoothing_window_size > fom["values"].size:
                logger.error(
                    f"Smoothing window size ({smoothing_window_size}) is greater "
                    f"than the number of FOM values ({fom['values'].size})!"
                )

            # Compute the smoothed derivative to use for best-tau determination and peak-finding
            deriv = savgol_filter(
                fom["values"],
                window_length=smoothing_window_size,
                polyorder=savgol_polyorder,
                deriv=savgol_derorder,
            )
            norm_deriv = deriv / deriv.max()
            pidx, peak_props = find_peaks(
                np.abs(norm_deriv),
                prominence=(0.05, norm_fom_peak_prominance),
                height=(None, norm_fom_peak_height),
            )

            # If there's only one peak, take that as the best estimate.
            if len(pidx) == 1:
                best_tau_fom = taus[pidx]
                fom_tau_estimates.append(np.squeeze(best_tau_fom))
                logger.info(
                    f"Best tau from metric={fom['name']:7s} is: {np.squeeze(best_tau_fom):.2f} ms"
                )
            # If there are multiple peaks (possible for complex profiles or poorly sampled trials),
            # the first instance of a significant peak is likely the best guess
            elif len(pidx) > 1:
                logger.debug(
                    "Multiple peaks in FOM found, taking mean of first two instances..."
                )
                multi_peak_flag += 1
                best_tau_fom = np.mean(taus[pidx[:1]])
                fom_tau_estimates.append(np.squeeze(best_tau_fom))
                logger.info(
                    f"Best tau from metric={fom['name']:7s} is: {np.squeeze(best_tau_fom):.2f} ms"
                )
            # If there are zero peaks (again, possible for complex profiles or poorly samples trials),
            # try to use a heuristic measure, otherwise we can't use that information and so the
            # weights/values need to be excluded.
            else:
                logger.warning(
                    f"Unable to find peaks in the FOM ({fom['name']}) derivative."
                )
                logger.debug(
                    "This could be due to profile complexity, or maybe you need to increase the number of trial taus."
                )
                logger.warning(
                    "Resorting to heuristic selection (generally, under-estimates)."
                )
                # Use heuristic method based on the specific FOM
                if fom["alt_operation"] != None:
                    fn = fom["alt_operation"]
                    best_tau_fom = taus[fn(fom["values"])]
                    fom_tau_estimates.append(best_tau_fom)
                else:
                    logger.debug(f"Excluding FOM={fom['name']} from further analysis.")
                    # Remove that FOM from the weighting scheme
                    fom_weights.pop(fom["name"])

    if multi_peak_flag > 0:
        logger.info(
            f"There were {multi_peak_flag} FOMs with >1 peaks, so the mean "
            f"of the first two peaks was used in each instance."
        )

    fom_tau_estimates = np.array(fom_tau_estimates)
    weights = np.array(
        [fom_weights[name] for name in fom_names if name in fom_weights.keys()]
    )
    fom_wt_mean_tau = np.average(fom_tau_estimates, weights=np.array(weights))
    fom_wt_std_tau = np.sqrt(
        np.average((fom_tau_estimates - fom_wt_mean_tau) ** 2, weights=weights)
    )
    d_tau = taus[1] - taus[0]

    # Now to approximate some kind of error
    wt_err = np.sqrt(fom_wt_std_tau**2 + d_tau**2)

    logger.info(
        f"Best overall tau = {fom_wt_mean_tau:g} +/- {wt_err:g} ms  (weighted mean, weighted error)"
    )

    return fom_wt_mean_tau, wt_err
