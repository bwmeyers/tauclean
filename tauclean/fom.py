import numpy as np


__all__ = ["consistence", "positivity", "skewness"]


def consistence(residuals, off_rms, off_mean=0, onlims=(0, 255)):
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
    nf = len(onpulse[abs(onpulse - off_mean) <= 3 * off_rms])

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
