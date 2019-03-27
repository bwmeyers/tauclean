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
    :param x: threshold (units of off_rms) defined to penalise the positivity if
    there residuals more negative than this [float]
    :return: the positivity figure of merit [float]
    """
    u = np.zeros_like(res)
    # When the residual is less than x * rms, turn on the step-function.
    # This means that only those points that have been over subtracted (due to a poor choice in PBF)
    # contribute to this parameter.
    u[res < x * off_rms] = 1

    f_r = (m / (len(res) * off_rms ** 2)) * np.sum(u * res ** 2)

    return f_r


def moment(a, t, n=1):
    """Calculate the moment of clean components as:
            <x> = sum{ a * t } / sum{ a }

            or

            <x^n> = sum{ a * (t - <t>)^n } / sum{ a }

    :param a: clean component amplitudes [array-like]
    :param t: corresponding clean component times [array-like]
    :param n: order [int]
    :return: moment of order n
    """
    first = np.sum(a * t) / np.sum(a)

    if n == 1:
        return first
    else:
        return np.sum(a * (t - first) ** n) / np.sum(a)


def skewness(ccs, dt=1.0):
    """The skewness of the clean components gives a figure of merit that describes how asymmetric the clean profile is.
    For a well-matched PBF and high signal-to-noise data, the clean component distribution should be approximately
    symmetric (Gaussian-like).
    Defined by Bhat et al. 2004 in their eqs. 12, 13 and 14.

    :param ccs: a list of component (delta-function) amplitudes produced at the end of the CLEAN procedure [array-like]
    :param dt: the time step per profile bin [float]
    :return: the skewness figure of merit [float]
    """

    # Compute the times for each clean components based on the assumption that the clean component array is the
    # same shape as the CLEANED profile
    cc_times = dt * np.linspace(0, 1, len(ccs))

    # The first moment, <t> = sum{ti * Ci} / sum{Ci}, is required to calculate others
    moment_1 = moment(ccs, cc_times, n=1)

    # Second and third moments defined by: <x^n> = sum{ (ti - <t>)^n * Ci } / sum{ Ci }
    moment_2 = moment(ccs, cc_times, n=2)
    moment_3 = moment(ccs, cc_times, n=3)

    # Absolute value doesn't really matter in this case as we are just trying to minimise the skewness
    gamma = moment_3 / (moment_2 ** 1.5)

    return gamma
