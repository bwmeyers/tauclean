import numpy as np
from scipy.integrate import simps
import pbf
import fom


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

    amp = 1.0 / (np.sqrt(2) * sigma)
    g = amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    return g


def reconstruction(profile, ccs, dt=1.0, norm=True):
    """Attempt to reconstruct the intrinsic pulse shape based on the clean component positions and amplitudes

    :param profile: the initial pulse profile [array-like]
    :param ccs: the clean components amplitudes [array-like]
    :param dt: time sample duration (i.e. amount of time per bin) [float]
    :param norm: whether to normalise the output to unit area [boolean]
    :return: a reconstruction of the intrinsic pulse profile [array-like]
    """
    nbins = len(ccs)
    x = dt * np.linspace(0, nbins, nbins)
    width = 3 * dt  # TODO: need to evaluate the approximate instrumental response more robustly
    impulse_response = gaussian(x, x[x.size // 2], width)
    impulse_response /= impulse_response.max()

    # Reconstruct the intrinsic pulse profile by convolving the clean components with the impulse response
    recon = np.convolve(ccs, impulse_response, mode="full")[:nbins]

    # Roll this such that the maximum value corresponds to the maximum value of the initial profile
    offset = np.argmax(profile) - np.argmax(recon)
    recon = np.roll(recon, offset)

    if norm:
        # Normalise the reconstructed profile such that it has unit area
        return recon / simps(y=recon, x=x)
    else:
        return recon


def clean(data, tau, results,
          on_start=0, on_end=255, dt=1.0, gain=0.01, threshold=3.0,
          pbftype="thin", iter_limit=None):
    """The primary function of tauclean that actually does the deconvolution.

    :param data: original pulse profile [array-like]
    :param tau: scattering time scale to use when calculating the PBF [float]
    :param results: a list that is defined globally such that it can be written to by this function and remain available
    after completion (in the case of multiple processes, this will be a multiprocessing.Manager list object) [list]
    :param on_start: starting bin of the on-pulse region [int]
    :param on_end: end bin of the on-pulse region [int]
    :param dt: time sample duration (i.e. amount of time per bin) [float]
    :param gain: a "loop gain" that is sued to scale the clean component amplitudes (usually 0.01-0.05) [float]
    :param threshold: threshold defining when to terminate the clean procedure [float]
    :param pbftype: type of pbf to use in the deconvolution [str]
    :param iter_limit: number of iterations after which to terminate the clean procedure regardless of convergence [int]
    :return:
    """
    nbins = len(data)
    nrot = 10

    # Create an x-range that is much larger than the nominal pulsar period, so that the full effect of the PBF can be
    # modelled by evaluating  over the extended range and then folding the result on the pulsar period.
    pbf_x = np.linspace(0, nrot, nrot * nbins) * dt

    # Decide which PBF model to use based on the user input
    if pbftype == "thin":
        filter_guess = pbf.thin(pbf_x, tau)
    elif pbftype == "thick":
        filter_guess = pbf.thick(pbf_x, tau)
    elif pbftype == "uniform":
        filter_guess = pbf.uniform(pbf_x, tau)
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

    # Pre-calculate a delta-function
    delta = np.zeros_like(profile)
    delta[delta.size // 2] = 1.0

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

        # Construct a clean component on the same scale as profile and assign its location and value
        temp_clean_comp = np.zeros_like(clean_components)
        temp_clean_comp[imax] = dmax * gain

        # Also add this component to the total list of clean components
        clean_components[imax] += dmax * gain

        # Create a restoring function with which to construct a model PBF to remove
        rest_func = np.convolve(temp_clean_comp, delta, mode="same")  # TODO: is this actually doing anything?
        rest_func = np.convolve(rest_func, filter_guess, mode="full")

        # Normalise restoring function so that we can ensure its amplitude is the same as the newest clean component.
        # We also calculate an offset (introduced by the convolution) that will re-align the component to the clean
        # component position.
        rest_func = rest_func / np.max(rest_func)
        component = rest_func * temp_clean_comp[imax]
        offset = np.argmax(temp_clean_comp) - np.argmax(rest_func)
        component = np.roll(component, offset)

        # In this case, we have done the full convolution to accurately capture the shape of the PBF
        # and here we fold that so that it matches the profile data size and will better represent the effect of the
        # PBF on the folded data.
        component = component[:nrot * nbins]
        component = np.sum(np.split(component, nrot), axis=0)

        # Finally, subtract the component from the profile
        cleaned = profile - component

        # Calculate the on- and off-pulse regions and use them with the user-defined cleaning threshold to determine
        # whether the clean procedure should be terminated
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
    gamma = fom.skewness(clean_components, dt=dt)
    recon = reconstruction(data, clean_components, dt=dt)

    results.append(
        dict(
            profile=profile, init_rms=init_rms, nbins=nbins, tau=tau, pbftype=pbftype,
            niter=niter, cc=clean_components, ncc=n_unique,
            nf=nf, off_rms=off_rms, off_mean=off_mean, on_rms=on_rms, fr=fr, gamma=gamma,
            recon=recon, threshold=threshold, on_start=on_start, on_end=on_end
        )
    )


