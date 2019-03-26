def consistence(residuals, off_rms, off_mean=0, onlims=(0, 1)):
    """The number of residual points in the on-pulse region that are consistent with the off-pulse rms is another
    indicator of how well the CLEAN procedure has done.
    Defined in Bhat et al. (2004) in the third-last paragraph of Section 2.5.3

    :param residuals: the residual profile after the CLEAN process has terminated [array-like]
    :param off_rms: the off-pulse rms noise [float]
    :param off_mean: the off-pulse mean value [float]
    :param onlims: a tuple containing the on-pulse region in terms of phase [(float, float) between 0 and 1 inclusive]
    :return: the number of points in the cleaned on-pulse region that are consistent with the off-pulse noise [int]
    """

    nbins = len(residuals)
    start = int(onlims[0] * nbins)
    end = int(onlims[1] * nbins)

    onpulse = residuals[start:end]

    # Calculate the number of on-pulse points that are consistent with the 3-sigma noise of the off-pulse
    nf = len(onpulse[abs(onpulse - off_mean) <= 3 * off_rms])

    return nf
