"""
Copyright 2019 Bradley Meyers
Licensed under the Academic Free License version 3.0
"""

import numpy as np
import matplotlib.pyplot as plt


__all__ = ["consistence", "positivity", "skewness"]


def consistence(residuals, off_rms, off_mean=0, onlims=(0, 255), thresh=3.0):
    """The number of residual points in the on-pulse region that are consistent with the off-pulse rms is another
    indicator of how well the CLEAN procedure has done.
    Defined in Bhat et al. (2004) in the third-last paragraph of Section 2.5.3

    :param residuals: the residual profile after the CLEAN process has terminated [array-like]
    :param off_rms: the off-pulse rms noise [float]
    :param off_mean: the off-pulse mean value [float]
    :param onlims: a tuple containing the on-pulse region in terms of phase [(float, float) between 0 and 1 inclusive]
    :return: the number of points in the cleaned on-pulse region that are consistent with the off-pulse noise [int]
    """

    start = onlims[0]
    end = onlims[1]

    onpulse = residuals[start:end]

    # Calculate the number of on-pulse points that are consistent with the 3-sigma noise of the off-pulse
    nf = len(onpulse[abs(onpulse - off_mean) <= thresh * off_rms])

    return nf


def positivity(res, off_rms, m=1.0, x=1.5):
    """The positivity figure of merit used to help decide on the quality of the CLEAN procedure.
    Defined by Bhat et al. 2004 in their eqs. 10 and 11.

    :param res: residuals after the CLEAN process has terminated [array-like]
    :param off_rms: off-pulse rms value to be used as a threshold [float]
    :param m: a scale-factor (or weight) that is of order unity [float]
    :param x: threshold (units of off_rms) defined to penalise the positivity if there residuals more negative
    than this [float]
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

    f_r = (m / (len(res) * off_rms ** 2)) * np.sum(u * res ** 2)

    return f_r


def skewness(ccs, period=100.0):
    """The skewness of the clean components gives a figure of merit that describes how asymmetric the clean profile is.
    For a well-matched PBF and high signal-to-noise data, the clean component distribution should be approximately
    symmetric (Gaussian-like).
    Defined by Bhat et al. 2004 in their eqs. 12, 13 and 14.

    :param ccs: a list of component (delta-function) amplitudes produced at the end of the CLEAN procedure [array-like]
    :param period: pulsar period (in ms) [float]
    :return: the skewness figure of merit [float]
    """

    # Compute the times for each clean components based on the assumption that the clean component array is the
    # same shape as the CLEANED profile
    cc_times = period * np.linspace(0, 1, len(ccs))

    # The moments defined are the equivalent to a weighted average, thus
    moment_1 = np.average(cc_times, weights=ccs)
    moment_2 = np.average((cc_times - moment_1) ** 2, weights=ccs)
    moment_3 = np.average((cc_times - moment_1) ** 3, weights=ccs)

    if np.count_nonzero(ccs) == 1:
        # The function is symmetric by definition at this stage, but moment_2 = 0 so we'll get an error if we try to
        # calculate the skewness in the usual way
        gamma = 0
    else:
        gamma = moment_3 / (moment_2 ** 1.5)

    return gamma


def get_error(results, dchi=1.0, plot=False):
    """Estimate the uncertainty of each tau trial value by determining the value of tau that results in an increase
    in f_c of unity (or in this case, by whatever value is specified in `dchi` (typically we would expect 1). This is
    basically equivalent to a reduced chi-square approach.

    Here, we use a second-order finite difference in an attempt to figure out where the maximum inflection begins, which
    nominally corresponds to the best-fitting tau.

    :param results: a list of dictionaries, one per trial tau [array-like]
    :param dchi: the offset from the minimum of either fr or fc that is used to constrain the error [float]
    :param plot: switch that will produce a diagnostic plot of how the errors were calculated if True [boolean]
    :returns tuple (fr_tau, fr_err, fc_tau, fc_err)
            WHERE
            fr_tau: the best-fitting tau estimated from the fr figure of merit [float]
            fr_err: the error based on fr [float]
            fc_tau: the best-fitting tau estimated from the fc figure of merit [float]
            fc_err: the error based on fc [float]
    """

    taus = np.array([a["tau"] for a in results])
    fr = np.array([a["fr"] for a in results])
    gamma = np.array([a["gamma"] for a in results])
    fc = (fr + gamma) / 2.0

    # rather than the minimum of fr, work out the point of greatest inflection using finite differences
    d2 = [(fr[i] - 2 * fr[i - 1] + fr[i - 2]) for i in range(2, len(taus) - 1)]  # backwards 2nd order difference
    imin = np.argmax(d2) - 2  # i.e. we don't actually want the max, we want i-2 which will be the "minimum point"
    imax = np.argmax(fr)

    # estimate the slope and intercept of the linear line drawn between the min and max values of fr
    frm = (fr[imax] - fr[imin]) / (taus[imax] - taus[imin])
    frc = fr[imin] - frm * taus[imin]

    fr_tauval = ((fr[imin] + dchi) - frc) / frm

    # the nominal uncertainty is then the difference between this and the best-fit tau
    fr_err = abs(fr_tauval - taus[imin])
    fr_tau = taus[imin]

    # Also try to get error estimates from fc
    d2 = [(fc[i] - 2 * fc[i - 1] + fc[i - 2]) for i in range(2, len(taus) - 1)]  # backwards 2nd order difference
    imin = np.argmax(d2) - 2  # ie we don't actually want the max, we want i-2 which will be the "minimum point"
    imax = np.argmax(fc)

    # estimate the slope and intercept of the linear line drawn between the min and max values of fc
    fcm = (fc[imax] - fc[imin]) / (taus[imax] - taus[imin])
    fcc = fc[imin] - fcm * taus[imin]

    fc_tauval = ((fc[imin] + dchi) - fcc) / fcm

    # the nominal uncertainty is then the difference between this and the best-fit tau
    fc_err = abs(fc_tauval - taus[imin])
    fc_tau = taus[imin]

    if plot:
        fig, (ax, ax2) = plt.subplots(1, 2)
        ax.plot(taus, fr, ls="none", marker="o")
        x = np.linspace(0.9 * taus[imin], 1.5 * fr_tauval, 10)
        ax.plot(x, frm * x + frc, ls=":")
        ax.axhline(fr[imin] + dchi)
        ax.axvline(fr_tauval)
        ax.set_title("fr")
        ax.set_ylim(None, 1.1*fr.max())

        ax2.plot(taus, fc, ls="none", marker="o")
        x = np.linspace(0.9 * taus[imin], 1.5 * fc_tauval, 10)
        ax2.plot(x, fcm * x + fcc, ls=":")
        ax2.axhline(fc[imin] + dchi)
        ax2.axvline(fc_tauval)
        ax2.set_title("fc")
        ax2.set_ylim(None, 1.1*fc.max())

        plt.savefig("tauclean_err_diag.png")
        plt.close(fig)

    return fr_tau, fr_err, fc_tau, fc_err
