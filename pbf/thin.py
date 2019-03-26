import numpy as np


def thin_screen(x, tau, x0=0):
    """The classical, square-law structure media thin screen approximation for a pulse broadening function.
    See e.g. Cordes & Rickett (1998) and Lambert & Rickett (1999).

    :param x: time over which to evaluate the PBF [numpy array]
    :param tau: pulse broadening time scale [float]
    :param x0: where the PBF turns on [float, in range of x]
    :return: evaluated thin screen PBF [numpy array]
    """
    t = x - x0
    h = (1 / tau) * np.exp(-t / tau)  # normalised to unit area

    # Turn on a unit step function at the given x0 offset.
    h[x <= x0] = 0

    return h
