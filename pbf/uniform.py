import numpy as np


def thick_screen(x, tau, x0=0):
    """The uniform media pulse broadening function as presented in Williamson (1972, 1973).

    :param x: time over which to evaluate the PBF [numpy array]
    :param tau: pulse broadening time scale [float]
    :param x0: where the PBF turns on [float, in range of x]
    :return: evaluated PBF for a uniform scattering medium [numpy array]
    """
    t = x - x0
    h = np.sqrt((np.pi**5 * tau**3) / (8 * t**5)) * np.exp(-tau * np.pi**2 / (4 * t))  # normalised to unit area

    # In some cases (for negative t), h will not be well-defined (nan).
    # However, this is not important in this case, thus we can simply
    # replace those values with 0.
    h[np.isnan(h)] = 0

    return h
