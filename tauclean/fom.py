#!/usr/bin/env python3
"""
########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
"""

import logging
import matplotlib.pyplot as plt
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


def consistence(onpulse, off_rms, off_mean=0, threshold=3.0):
    """The number of residual points in the on-pulse region that are
    consistent with the off-pulse rms is another indicator of how well
    the CLEAN procedure has done.

    Defined in Bhat et al. (2004) in the third-last paragraph of Section 2.5.3

    :param onpulse: the residual profile after the CLEAN process has terminated [array-like]
    :param off_rms: the off-pulse rms noise [float]
    :param off_mean: the off-pulse mean value [float]
    :param threshold: the threshold with which to compare for significance [float]
    :return: the number of points in the cleaned on-pulse region that
        are consistent with the off-pulse noise [int]
    """

    # Calculate the number of on-pulse points that are consistent with the 3-sigma noise of the off-pulse
    nf = np.sum(abs(onpulse - off_mean) <= threshold * off_rms)

    return nf


def positivity(res, off_rms, m=1.0, x=1.5):
    """The positivity figure of merit used to help decide on the quality
    of the CLEAN procedure.

    Defined by Bhat et al. 2004 in their eqs. 10 and 11.

    :param res: residuals after the CLEAN process has terminated [array-like]
    :param off_rms: off-pulse rms value to be used as a threshold [float]
    :param m: a scale-factor (or weight) that is of order unity [float]
    :param x: threshold (units of off_rms) defined to penalise the positivity if
        there residuals more negative than this [float]
    :return: the positivity figure of merit [float]
    """

    u = np.zeros_like(res)
    # When the residual is less than x * rms, turn on the step-function.
    # This means that only those points that have been over subtracted (due to a poor choice in PBF)
    # contribute to this parameter.
    u[res < -x * off_rms] = 1

    if np.all(res == 0):
        # safe from a plotting perspective as NaNs are ignored
        return np.nan

    f_r = (m / (len(res) * off_rms**2)) * np.sum(u * res**2)

    return f_r


def skewness(ccs, period=100.0):
    """The skewness of the clean components gives a figure of merit
    that describes how asymmetric the clean profile is. For a well-matched
    PBF and high signal-to-noise data, the clean component distribution
    should be approximately
    symmetric (Gaussian-like).

    Defined by Bhat et al. 2004 in their eqs. 12, 13 and 14.

    :param ccs: a list of component (delta-function) amplitudes produced
        at the end of the CLEAN procedure [array-like]
    :param period: pulsar period (in ms) [float]
    :return: the skewness figure of merit [float]
    """

    # Compute the times for each clean components based on the assumption
    # that the clean component array is the same shape as the CLEANED profile
    cc_times = period * np.linspace(0, 1, len(ccs))

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


def get_best_tau_jerk(results, norm_fom_peak_height=0.8, smoothing_window_size=None):
    """Estimate the uncertainty of each tau trial value by determining
    the value of tau that results in an increase of `dchi` units in the
    f_c or f_r metrics (typically we would expect dchi=1).

    Here, we use a second-order finite difference in an attempt to figure
    out where the maximum inflection begins, which nominally corresponds to
    the best-fitting tau.

    :param results: a list of dictionaries, one per trial tau [array-like]
    :param norm_fom_peak_height: height parameter to determine peaks from
        normalised FOM derivatives, default=0.8 [float]
    :param smoothing_window_size: window size to use when computing smoothed
        FOM derivatives, optional, default set based on FOM series length [int]
    :returns tuple (best_tau, approx_err) [float, float]
    """
    taus = np.array([a["tau"] for a in results])
    fr = np.array([a["fr"] for a in results])
    gamma = np.array([a["gamma"] for a in results])
    fc = (fr + gamma) / 2.0
    sigma_c = np.array([a["total_rms"] for a in results]) / np.array(
        [a["init_off_rms"] for a in results]
    )
    nf_frac = np.array([a["nf"] for a in results]) / np.array(
        [a["nbins"] for a in results]
    )

    fom = [fr, gamma, fc, sigma_c, nf_frac]
    names = ["fr", "gamma", "fc", "sigma_c", "nf_frac"]
    weights = [1, 0.8, 0.1, 1, 1]
    savgol_polyorder = 3
    savgol_derorder = 3

    # Use the FOM and their derivatives to estimate the best match
    fom_tau_estimates = []
    for f, lab in zip(fom, names):
        logger.debug(f"Finding 'best' tau from FOM={lab} via 3rd deriv.")

        if not smoothing_window_size:
            logger.debug(
                "No savgol_filter window size provided, choosing sensible value based on FOM series length."
            )
            # The smoothing window size must be greater than the polynomial order used in the filter
            smoothing_window_size = len(f) // 8
            if smoothing_window_size <= savgol_polyorder:
                smoothing_window_size = savgol_polyorder + 1
            logger.debug(f"Window size set to {smoothing_window_size} bins")

        deriv = savgol_filter(
            f,
            window_length=smoothing_window_size,
            polyorder=savgol_polyorder,
            deriv=savgol_derorder,
        )
        norm_deriv = deriv / deriv.max()
        pidx, _ = find_peaks(norm_deriv, height=norm_fom_peak_height)

        if len(pidx) == 1:
            best_tau_fom = taus[pidx]
            fom_tau_estimates.append(np.squeeze(best_tau_fom))
            logger.info(
                f"Best tau from metric={lab} is: {np.squeeze(best_tau_fom):.2f} ms"
            )
        elif len(pidx) > 1:
            logger.info("Multiple peaks in FOM found, taking median...")
            best_tau_fom = np.median(taus[pidx])
            fom_tau_estimates.append(np.squeeze(best_tau_fom))
            logger.info(
                f"Best tau from metric={lab} is: {np.squeeze(best_tau_fom):.2f} ms"
            )
        else:
            logger.error("Something went wrong finding peaks in the FOM derivatives...")

    fom_median_tau = np.average(np.array(fom_tau_estimates), weights=np.array(weights))

    # Now to approximate some kind of error
    base_err = np.sqrt(np.std(fom_tau_estimates) ** 2 + (taus[1] - taus[0]) ** 2)

    logger.info(
        f"Best overall tau = {fom_median_tau:g} +/- {base_err:g} ms  (median from all FOM)"
    )

    return fom_median_tau, base_err
