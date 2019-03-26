import numpy as np


def moment(a, t, n=1):
    """Calculate the moment of clean components as:
            <x> = sum{ a * t } / sum{ a }

            or

            <x^n> = sum{ a * (t - <t>)^n } / sum{ a }

    :param a: clean component amplitudes [numpy array]
    :param t: corresponding clean component times [numpy array]
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

    :param ccs: a list of component (delta-function) amplitudes produced at the end of the CLEAN procedure [numpy array]
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
