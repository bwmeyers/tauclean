#! /usr/bin/env python
"""
########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
"""

import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as simps

from . import pbf
from .clean import dm_delay

logger = logging.getLogger(__name__)
# Set the seed for numpy's random functions so that the same result can be retrieved each time
np.random.seed(12345)


def create_intrinsic_pulse(position, width, amps, nbins=2048):
    """Simulate an intrinsic pulse shape (made of Gaussian components)

    :param position: where the centroid (mean) of the Gaussian is to be placed (units: bins) [array-like]
    :param width: width (standard deviation) of the Gaussian (units: bins) [array-like]
    :param amps: peak amplitudes of the Gaussian [array-like]
    :param nbins: desired number of bins in the profile
    :return: the intrinsic emission profile [array-like]
    """

    x = np.linspace(0, nbins, nbins)

    f = np.zeros_like(x)

    # If a list of positions have been given, but only one value for the width, then create an array of widths that
    # are all the same
    if len(position) > 1 and len(width) == 1:
        logger.warning("Will use same width for each component")
        width = np.repeat(width, len(position))

    # Similarly for the amplitudes
    if len(position) > 1 and len(amps) == 1:
        logger.warning("Will use same amplitude for each component")
        amps = np.repeat(amps, len(position))

    # For each (position, width, amplitude) set, create and add a component to the intrinsic pulse profile
    for p, w, a in zip(position, width, amps):
        logger.debug("added gaussian comp.")
        g = pbf.gaussian(x, float(p), float(w))
        f += a * (g / g.max())

    return f


def create_scattered_profile(
    intrinsic, tau, rest_width, pbftype="thin", period=100.0, snr=500.0
):
    """Take the intrinsic emission profile and apply the effects of a scattering screen to it, then add noise

    :param intrinsic: intrinsic emission profile [array-like]
    :param tau: desired pulse broadening time scale (units: ms) [float]
    :param rest_width: width of the telescope restoring function/impulse response (units: ms) [float]
    :param pbftype: the type of PBF to use [string]
    :param period: pulsar period (units: ms)
    :param snr: nominal desired signal-to-noise ration
    :return: pbf [array-like], scattered profile [array-like], scattered profile with noise added [array-like]
    """

    nbins = len(intrinsic)

    x = period * np.linspace(0, 1, nbins)

    # Decide which PBF model to use
    if pbftype == "thin":
        h = pbf.thin(x, tau)
    elif pbftype == "thick":
        h = pbf.thick(x, tau)
    elif pbftype == "uniform":
        h = pbf.uniform(x, tau)
    elif pbftype == "thick_exp":
        h = pbf.thick_exp(x, tau)
    elif pbftype == "uniform_exp":
        h = pbf.uniform_exp(x, tau)
    else:
        logger.error("Invalid PBF type requested ({0})".format(pbftype))
        logger.warning("Defaulting to thin screen...")
        h = pbf.thin(x, tau)

    restoring_function = pbf.gaussian(x, x[x.size // 2], rest_width)

    # The observed pulse shape is the convolution of:
    # - the true signal,
    # - the telescope response function,
    # - the scattering kernel, and
    # - some Gaussian radiometer noise
    # Here we do the mode="full" convolution so that the complete shape is convolved and we don't end up with sharp edge
    # effects in the final profile that depend on where the shapes are defined (as in the case of mode="same")
    response = np.convolve(intrinsic, restoring_function, mode="full") / np.sum(
        restoring_function
    )
    response = response[nbins // 2 : -(nbins // 2) + 1]

    scattered = np.convolve(response, h, mode="full") / np.sum(h)

    offset = np.argmax(intrinsic) - np.argmax(scattered)
    scattered = np.roll(scattered, offset)[:nbins]
    scattered = scattered[:nbins]

    # And do the same for the PBF, then re-normalise it to unit area
    h = h / simps(x=x, y=h)

    # Add noise to produce a profile with approximately the signal-to-noise ratio desired
    observed = np.copy(scattered) + np.random.normal(
        0, scattered.max() / snr, scattered.size
    )

    return h, scattered, observed


def plot_simulated(
    intrinsic,
    kernel,
    scattered,
    observed,
    tau,
    pbftype,
    snr,
    period=100.0,
    xunit="time",
    save=False,
):
    """Plot the simulated data in the desired units

    :param intrinsic: intrinsic emission profile [array-like]
    :param kernel: scattering kernel (PBF) used [array-like]
    :param scattered: scattered profile (convolution of intrinsic and kernel) [array-like]
    :param observed: a scattered profile with noise added (i.e. the observed profile) [array-like]
    :param tau: pulse broadening time scale (units: ms) [float]
    :param pbftype: pulse broadening function type [string]
    :param snr: nominal signal-to-noise ratio of observed profile [float]
    :param period: pulsar period (units: ms) [float]
    :param xunit: what units to plot along the x-axis [string]
    :param save: whether to save the plot to disk or note [boolean]
    :return: None
    """

    nbins = len(intrinsic)

    if xunit == "phase":
        x = np.linspace(0, 1, nbins)
        xlab = "Phase"
    elif xunit == "time":
        x = period * np.linspace(0, 1, nbins)
        xlab = "Time (ms)"
    elif xunit == "bins":
        x = np.linspace(0, nbins, nbins).astype(int)
        xlab = "Bins"
    else:
        logger.error("Unknown x-unit: {0}".format(xunit))
        sys.exit(1)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3)

    ax_int = fig.add_subplot(gs[0, 0])
    ax_ker = fig.add_subplot(gs[0, 1], sharex=ax_int)
    ax_sim = fig.add_subplot(gs[0, 2], sharex=ax_int)
    ax_obs = fig.add_subplot(gs[1, :])

    ax_int.plot(x, intrinsic, color="C0")
    ax_int.set_title("Intrinsic pulse")
    ax_int.set_ylabel("Intensity")
    ax_int.set_xlabel(xlab)
    ax_int.set_xlim(0, x.max())
    step = x.max() / 4.0
    ax_int.set_xticks(np.arange(0, x.max() + step, step))

    ax_ker.plot(x, kernel, color="C1", label=r"$\rm \tau = {0:g} ms$".format(tau))
    ax_ker.set_title("Scattering kernel")
    ax_ker.set_xlabel(xlab)
    ax_ker.legend()

    ax_sim.plot(x, scattered, color="C2")
    ax_sim.set_title("Scattered profile")
    ax_sim.set_xlabel(xlab)

    ax_obs.plot(x, observed, color="k")
    ax_obs.set_title(
        r"Observed pulse profile (noise added, $\rm SNR \approx {0}$)".format(snr)
    )
    ax_obs.axhline(0, ls="--", color="r", lw=1)
    step = x.max() / 16.0
    ax_obs.set_xticks(np.arange(0, x.max() + step, step))
    ax_obs.set_xlim(0, x.max())
    ax_obs.grid(True)
    ax_obs.set_xlabel(xlab)

    plt.subplots_adjust(wspace=0.25)
    if save:
        plt.savefig(
            "simulated-profile_{0}-tau{1:g}.png".format(pbftype, tau),
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show()


def write_data(intrinsic, kernel, scattered, observed, pbftype, tau):
    """Small function to just write all of the simulated results to text files

    :param intrinsic: intrinsic emission profile [array-like]
    :param kernel: scattering kernel (PBF) used [array-like]
    :param scattered: scattered profile (convolution of intrinsic and kernel) [array-like]
    :param observed: a scattered profile with noise added (i.e. the observed profile) [array-like]
    :param pbftype: pulse broadening function type [string]
    :param tau: pulse broadening time scale (units: ms) [float]
    :return: None
    """

    np.savetxt("sim-intrinsic_{0}-tau{1:g}.txt".format(pbftype, tau), intrinsic)
    np.savetxt("sim-kernel_{0}-tau{1:g}.txt".format(pbftype, tau), kernel)
    np.savetxt("sim-scattered_{0}-tau{1:g}.txt".format(pbftype, tau), scattered)
    np.savetxt("sim-profile_{0}-tau{1:g}.txt".format(pbftype, tau), observed)

    logger.info(
        "Wrote final scattered profile to: sim-profile_{0}-tau{1:g}.txt".format(
            pbftype, tau
        )
    )


def main():
    parser = argparse.ArgumentParser(
        prog="simulate", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-n", type=int, default=2048, help="number of profile bins")
    parser.add_argument(
        "-m",
        nargs="+",
        type=int,
        help="centre positions of gaussian components (in bins)",
    )
    parser.add_argument(
        "-w",
        nargs="+",
        type=float,
        help="widths (std. dev.) of gaussian components (in bins)",
    )
    parser.add_argument(
        "-a", nargs="+", type=float, help="amplitudes of gaussian components"
    )
    parser.add_argument(
        "-k",
        metavar="pbf",
        default="thin",
        choices=["thin", "thick", "uniform", "thick_exp", "uniform_exp"],
        help="The type of PBF kernel to use during the deconvolution."
        "A '_exp' suffix implies a modified PBF that asymptotes to a thin-screen approximation at large times.",
    )
    parser.add_argument(
        "-t", type=float, default=5.0, help="scattering time scale (in ms)"
    )
    parser.add_argument("-p", type=float, default=100.0, help="pulsar period (in ms)")
    parser.add_argument(
        "-d",
        "--dm",
        metavar="DM",
        type=float,
        default=0.0,
        help="pulsar dispersion measure (in pc/cm^3) - use zero to simulate coherent de-dispersion",
    )
    parser.add_argument(
        "-f",
        "--freq",
        metavar="freq",
        type=float,
        default=1.4,
        help="centre observing frequency (in GHz)",
    )
    parser.add_argument(
        "-b",
        "--bw",
        metavar="BW",
        type=float,
        default=0.256,
        help="observing bandwidth (in GHz)",
    )
    parser.add_argument(
        "--nchan", type=int, default=1024, help="number of frequency channels"
    )
    parser.add_argument(
        "-s", type=float, default=500.0, help="Desired signal-to-noise ratio"
    )
    parser.add_argument(
        "-x",
        default="bins",
        choices=["time", "phase", "bins"],
        help="plot x-axis units",
    )
    parser.add_argument(
        "--saveplot",
        action="store_true",
        default=False,
        help="Switch to save plot to disk rather than just show",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help="Write the data (intrinsic, kernel, convolution and noisy) to files",
    )

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s :: %(name)s :: %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if len(args.w) != len(args.m):
        logger.warning(
            "Provided different number of widths than positions, selecting first width"
        )
        args.w = [args.w[0]]

    if len(args.a) != len(args.m):
        logger.warning(
            "Provided different number of amplitudes than positions, selecting first amplitude"
        )
        args.a = [args.a[0]]

    time_sample = args.p / args.n
    logger.info("Time sample: {0:g} ms".format(time_sample))
    # Figure out the dispersion smearing in the worst case (i.e. in the lowest channel), and then determine the
    # nominal width of the restoring function
    chan_bw = args.bw / args.nchan

    logger.debug("Frequency channel size: {0:g} MHz".format(chan_bw * 1000))
    lochan = args.freq - (args.bw / 2)
    hichan = lochan + chan_bw

    logger.debug(
        "Lowest channel edges: {0:g}-{1:g} MHz".format(lochan * 1000, hichan * 1000)
    )
    dmdelay = dm_delay(args.dm, lochan, hichan)
    logger.info("Dispersion smearing in lowest channel: {0:g} ms".format(dmdelay))

    restoring_width = np.sqrt(time_sample**2 + dmdelay**2)
    logger.info("Restoring function width: {0:g} ms".format(restoring_width))

    i = create_intrinsic_pulse(args.m, args.w, args.a, nbins=args.n)
    k, s, o = create_scattered_profile(
        i, args.t, restoring_width, pbftype=args.k, period=args.p, snr=args.s
    )

    plot_simulated(
        i,
        k,
        s,
        o,
        args.t,
        args.k,
        snr=args.s,
        period=args.p,
        xunit=args.x,
        save=args.saveplot,
    )

    if args.write:
        write_data(i, k, s, o, args.k, args.t)


if __name__ == "__main__":
    main()
