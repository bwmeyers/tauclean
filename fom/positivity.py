import numpy as np


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
