"""
Copyright 2019 Bradley Meyers
Licensed under the Academic Free License version 3.0
"""

import numpy as np
from scipy.integrate import simps

from . import fom
from . import pbf


def keep_cleaning(on, off, threshold=3.0):
    """Compute residuals from data after clean component subtraction. The rms and baseline is determined from the
    "off-pulse" region of the profile, and the signal strength is calculated from the "on-pulse" region. Decides whether
    to keep cleaning or terminate the process.

    :param on: on-pulse time series [array-like]
    :param off: off-pulse time series [array-like]
    :param threshold: noise-threshold to determine when to stop cleaning [float]
    :return: whether to keep cleaning or not [boolean]
    """

    rms = np.std(off)
    mean = np.mean(off)

    datamax = np.max(on)

    limit = mean + threshold * rms

    return datamax > limit


def gaussian(x, mu, sigma):
    """A simple function that calculates a Gaussian shape over x

    :param x: independent variable [array-like]
    :param mu: the mean (position) of the Gaussian [float]
    :param sigma: the standard deviation (width) of the Gaussian [float]
    :return: a numerically evaluated Gaussian [array-like]
    """

    amp = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    g = amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    return g


def dm_delay(dm, lo, hi):
    """Calculate the dispersion delay between frequencies "lo" and "hi", both in GHz

    :param dm: dispersion measure of pulsar (in cm^-3 pc) [float]
    :param lo: the lowest frequency (in GHz) [float]
    :param hi: the highest frequency (in GHz) [float]
    :return: dispersion delay (in ms) [float]
    """

    k = 4.148808  # dispersion constant in ms
    delay = k * dm * (lo**(-2) - hi**(-2))

    return delay


def get_response(nbins, period=100.0, freq=1.4, bw=0.256, nchan=1024, dm=0.0, coherent=False):
    """Estimate the restoring function width in milliseconds based on the time sampling and (possible) residual DM
    delay in channels

    :param nbins: number of bins in profile [int]
    :param period: pulsar rotation period in ms [float]
    :param freq: centre observing frequency in GHz [float]
    :param bw: observing bandwidth in GHz [float]
    :param nchan: number of frequency channels [int]
    :param dm: dispersion measure of pulsar (in cm^-3 pc) [float]
    :param coherent: boolean switch as to whether coherent dedispersion was used [boolean]
    :return: (rest_width, inst_resp) [tuple]
            WHERE
            rest_width: equivalent restoring function width in ms [float]
            inst_resp: the instrumental response [array-like]
    """
    time_sample = float(period) / nbins

    if coherent:
        # If coherently de-dispersed, then there will be no DM-smearing component to the response function
        dmdelay = 0.0
    else:
        # Figure out the dispersion smearing in the worst case (i.e. in the lowest channel), and then determine the
        # nominal width of the restoring function
        chan_bw = float(bw) / nchan
        lochan = freq - (bw / 2.0)
        hichan = lochan + chan_bw

        dmdelay = dm_delay(dm, lochan, hichan)

    # Restoring width in milliseconds
    restoring_width = np.sqrt(time_sample ** 2 + dmdelay ** 2)

    if dmdelay == 0:
        # Don't bother doing anything else, the instrumental response is identical to the time sampling
        instrumental_response = np.zeros(nbins)
        instrumental_response[nbins // 2] = 1.0
    else:
        # Of the two time scales, figure out which is smaller so that we can upsample to ensure that it is resolved
        narrowest_factor = min([time_sample, dmdelay])

        desired_tres = 10.0e-3  # 10 us (since we're operating in ms)

        if narrowest_factor < desired_tres:
            # The narrowest factor is less than 10 microseconds, so just increase resolution by a factor of 16
            ratio = 16
        else:
            # Determine the up/down sampling ration such that the convolution happens at a time resolution
            # of < 10 microseconds
            ratio = 1 << (int(np.ceil(time_sample / desired_tres)) - 1).bit_length()

        # Figure out how many upsampled bins (rounding up) each time scale corresponds to
        upsampled_tsamp_bins = int(np.ceil(ratio * (time_sample / narrowest_factor)))
        upsampled_tdisp_bins = int(np.ceil(ratio * (dmdelay / narrowest_factor)))

        # Make rectangular functions for each of these time scales
        length = ratio * nbins
        tpb = np.zeros(length)
        tdm = np.zeros_like(tpb)

        offset = length // 2  # offset the "signal" to roughly the centre

        tpb[offset:offset + upsampled_tsamp_bins] = 1.0
        tdm[offset:offset + upsampled_tdisp_bins] = 1.0

        # Convolve the above response functions to form the total instrument response
        r = np.convolve(tpb, tdm, mode="full")
        r = r[offset:-offset + 1]
        r = r / simps(y=r, dx=1)  # normalise to unit area

        # Re-sample the outcome to have the same shape as the input data
        instrumental_response = r[::ratio]

    return restoring_width, instrumental_response


def reconstruct(profile, ccs, period=100.0, rest_width=1.0):
    """Attempt to reconstruct the intrinsic pulse shape based on the clean component positions and amplitudes

    :param profile: the initial pulse profile [array-like]
    :param ccs: the clean components amplitudes [array-like]
    :param period: pulsar period (in ms) [float]
    :param rest_width: telescope restoring function width (in ms) [float]
    :return: a reconstruction of the intrinsic pulse profile [array-like]
    """

    nbins = len(ccs)
    x = period * np.linspace(0, 1, nbins)

    # Calculate the nominal effective time sampling, including effects of dispersion smearing in channels
    restoring_func = gaussian(x, x[x.size // 2], rest_width)

    # Reconstruct the intrinsic pulse profile by convolving the clean components with the impulse response
    # The impulse response has unit area, thus the fluence of the pulse should be conserved
    recon = np.convolve(ccs, restoring_func, mode="full") / np.sum(restoring_func)
    recon = recon[nbins // 2:-(nbins // 2) + 1]  # actually just want the middle bit of this

    # Roll this such that the maximum value corresponds to the maximum value of the initial profile
    offset = np.argmax(profile) - np.argmax(recon)
    recon = np.roll(recon, offset)

    return recon, restoring_func


def clean(data, tau, inst_resp,
          on_start=0, on_end=255, period=100.0, rest_width=1.0,
          gain=0.05, threshold=3.0, pbftype="thin", iter_limit=None):
    """The primary function of tauclean that actually does the deconvolution.

    :param data: original pulse profile [array-like]
    :param tau: scattering time scale to use when calculating the PBF [float]
    :param inst_resp: the instrumental response with the same shape as the data [array-like]
    :param on_start: starting bin of the on-pulse region [int]
    :param on_end: end bin of the on-pulse region [int]
    :param period: pulsar period (in ms) [float]
    :param rest_width: telescope restoring function width (in ms) [float]
    :param gain: a "loop gain" that is sued to scale the clean component amplitudes (usually 0.01-0.05) [float]
    :param threshold: threshold defining when to terminate the clean procedure [float]
    :param pbftype: type of pbf to use in the deconvolution [str]
    :param iter_limit: number of iterations after which to terminate the clean procedure regardless of convergence [int]
    :return: a dictionary containing various parameters and output of the cleaning process [dictionary]
    """

    nbins = len(data)

    # Create an x-range that is much larger than the nominal pulsar period, so that the full effect of the PBF can be
    # modelled by evaluating  over the extended range and then folding the result on the pulsar period.
    x = np.linspace(0, 1, nbins) * period

    # Decide which PBF model to use based on the user input
    if pbftype == "thin":
        filter_guess = pbf.thin(x, tau)
    elif pbftype == "thick":
        filter_guess = pbf.thick(x, tau)
    elif pbftype == "uniform":
        filter_guess = pbf.uniform(x, tau)
    elif pbftype == "thick_exp":
        filter_guess = pbf.thick_exp(x, tau)
    elif pbftype == "uniform_exp":
        filter_guess = pbf.uniform_exp(x, tau)
    else:
        print("Invalid PBF type requested ({0})".format(pbftype))
        return None

    # Copy data into profile so that data is never actually touched in the process
    profile = np.copy(data)
    on_pulse = profile[on_start:on_end]
    off_pulse = np.concatenate((profile[:on_start], profile[on_end:]))
    init_rms = np.std(off_pulse)

    # Pre-allocate an array the same size as profile where the clean component locations and amplitudes will be stored
    clean_components = np.zeros_like(profile)

    # Initialise counters and boolean checks
    loop = True
    niter = 0

    # Start the clean procedure, terminating when either the iteration limit is reached, or when the on-pulse residuals
    # no longer hold data values above the 3-sigma off-pulse rms noise.
    while loop:

        if (iter_limit is not None) and (niter >= iter_limit):
            print("Reached iteration limit for tau={0:g}".format(tau))
            break
        else:
            niter += 1

        # Identify the location and value of the maximum data point in the profile
        imax = np.argmax(profile[on_start:on_end]) + on_start
        dmax = profile[imax]

        # Construct a single clean component on the same scale as the profile and assign its location and value
        temp_clean_comp = np.zeros_like(clean_components)
        temp_clean_comp[imax] = dmax * gain

        # Also add this component to the total list of clean components which will be used in the profile reconstruction
        clean_components[imax] += dmax * gain

        # Create a profile component from the model PBF, the instrumental response, and the clean component
        component = np.convolve(temp_clean_comp, inst_resp, mode="full") / np.sum(inst_resp)
        component = np.convolve(component, filter_guess, mode="full") / np.sum(filter_guess)

        # Roll the component so that it is re-aligned with the clean component as this dictates from where in the
        # profile the constructed component is removed
        offset = np.argmax(temp_clean_comp) - np.argmax(component)
        component = np.roll(component, offset)

        # In this case, we have done the full convolution to accurately capture the shape of the PBF, now just grab
        # the profile-length
        component = component[:nbins]  # this is ok because we moved to be back in the correct position already

        # Finally, subtract the component from the profile
        cleaned = profile - component

        # Calculate the on- and off-pulse regions and use them with the user-defined cleaning threshold to determine
        # whether the clean procedure should be terminated at this point
        on_pulse = cleaned[on_start:on_end]
        off_pulse = np.concatenate((cleaned[:on_start], cleaned[on_end:]))
        loop = keep_cleaning(on_pulse, off_pulse, threshold=threshold)

        # Now replace the profile with the newly cleaned profile for the next iteration
        profile = np.copy(cleaned)

    # After the clean procedure, calculate figures of merit and other information about the process
    n_unique = np.count_nonzero(clean_components)
    off_rms = np.std(off_pulse)
    off_mean = np.mean(off_pulse)
    on_rms = np.std(on_pulse)
    nf = fom.consistence(profile, off_rms, off_mean=off_mean, onlims=(on_start, on_end))
    fr = fom.positivity(profile, off_rms)
    gamma = fom.skewness(clean_components, period=period)
    recon, rest_func = reconstruct(data, clean_components, period=period, rest_width=rest_width)

    return dict(
            profile=profile, init_rms=init_rms, nbins=nbins, tau=tau, pbftype=pbftype,
            niter=niter, cc=clean_components, ncc=n_unique, inst_resp=inst_resp, rest_func=rest_func,
            nf=nf, off_rms=off_rms, off_mean=off_mean, on_rms=on_rms, fr=fr, gamma=gamma,
            recon=recon, threshold=threshold, on_start=on_start, on_end=on_end
        )
