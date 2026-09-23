import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as simps

from ..kernels import KernelRegistry, get_kernel
from ..response import dm_delay, gaussian

logger = logging.getLogger(__name__)
# Set the seed for numpy's random functions so that the same result can be
# retrieved each time
np.random.seed(12345)


def create_intrinsic_pulse(position, width, amps, nbins=2048):
    """Simulate an intrinsic pulse shape (made of Gaussian components)

    :param position: where the centroid (mean) of the Gaussian is to be
        placed (units: bins) [array-like]
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
        g = gaussian(x, float(p), float(w))
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
    try:
        kernel = get_kernel(pbftype)
    except ValueError:
        logger.error("Invalid PBF type requested (%s)", pbftype)
        logger.warning("Defaulting to thin screen...")
        kernel = get_kernel("thin")
    h = kernel(x, tau)

    restoring_function = gaussian(x, x[x.size // 2], rest_width)

    # The observed pulse shape is the convolution of:
    # - the true signal,
    # - the telescope response function,
    # - the scattering kernel, and
    # - some Gaussian radiometer noise
    # Here we do the mode="full" convolution so that the complete shape is convolved and we don't end up with sharp edge
    # effects in the final profile that depend on where the shapes are defined (as in the case of mode="same")
    response = np.convolve(
        intrinsic, restoring_function, mode="full"
    ) / np.sum(restoring_function)
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
    dm=None,
    freq=None,
    bw=None,
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
    :param dm: pulsar dispersion measure (units: pc/cm^3) [float]
    :param freq: centre observing frequency (units: GHz) [float]
    :param bw: observing bandwidth (units: GHz) [float]
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
        logger.error("Unknown x-unit: %s", xunit)
        sys.exit(1)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.5)

    # Extra headroom for the top row titles when a secondary bin axis is shown
    title_pad = 10 if xunit != "bins" else None

    ax_int = fig.add_subplot(gs[0, 0])
    ax_ker = fig.add_subplot(gs[0, 1], sharex=ax_int)
    ax_sim = fig.add_subplot(gs[0, 2], sharex=ax_int)
    ax_obs = fig.add_subplot(gs[1, :])

    # Bins are linear in both "time" and "phase" units, so a simple scale
    # factor converts between them for the secondary (top) bin axis
    if xunit == "time":
        bins_per_unit = nbins / period
    elif xunit == "phase":
        bins_per_unit = nbins
    else:
        bins_per_unit = None

    def add_bin_axis(ax):
        if bins_per_unit is None:
            return
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda v: v * bins_per_unit,
                lambda v: v / bins_per_unit,
            ),
        )
        # secax tick positions are in its own (bin) units, so the bottom
        # axis' ticks must be converted before being applied here
        bin_ticks = ax.get_xticks() * bins_per_unit
        secax.set_xticks(bin_ticks)
        secax.set_xticklabels([f"{round(b):d}" for b in bin_ticks])
        secax.set_xlabel("Bins")

    ax_int.plot(x, intrinsic, color="C0")
    ax_int.set_title("Intrinsic pulse", pad=title_pad)
    ax_int.set_ylabel("Intensity")
    ax_int.set_xlabel(xlab)
    ax_int.set_xlim(0, x.max())
    step = x.max() / 4.0
    ax_int.set_xticks(np.arange(0, x.max() + step, step))
    add_bin_axis(ax_int)

    ax_ker.plot(x, kernel, color="C1", label=rf"$\rm \tau = {tau:g} ms$")
    ax_ker.set_title("Scattering kernel", pad=title_pad)
    ax_ker.set_xlabel(xlab)
    ax_ker.legend()
    add_bin_axis(ax_ker)

    ax_sim.plot(x, scattered, color="C2")
    ax_sim.set_title("Scattered profile", pad=title_pad)
    ax_sim.set_xlabel(xlab)
    add_bin_axis(ax_sim)

    ax_obs.plot(x, observed, color="k")
    ax_obs.set_title(
        rf"Observed pulse profile (noise added, $\rm SNR \approx {snr}$)",
        pad=title_pad,
    )
    ax_obs.axhline(0, ls="--", color="r", lw=1)
    step = x.max() / 8.0
    ax_obs.set_xticks(np.arange(0, x.max() + step, step))
    ax_obs.set_xlim(0, x.max())
    ax_obs.grid(True)
    ax_obs.set_xlabel(xlab)
    add_bin_axis(ax_obs)

    info_lines = [rf"Period = {period:g} ms"]
    if dm is not None:
        info_lines.append(rf"DM = {dm:g} pc cm$^{{-3}}$")
    if freq is not None:
        info_lines.append(rf"Freq = {freq:g} GHz")
    if bw is not None:
        info_lines.append(rf"BW = {bw:g} GHz")
    ax_obs.text(
        0.98,
        0.95,
        "\n".join(info_lines),
        transform=ax_obs.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )

    plt.subplots_adjust(wspace=0.25)
    if save:
        plt.savefig(
            f"simulated-profile_{pbftype}-tau{tau:g}.png",
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

    np.savetxt(f"sim-intrinsic_{pbftype}-tau{tau:g}.txt", intrinsic)
    np.savetxt(f"sim-kernel_{pbftype}-tau{tau:g}.txt", kernel)
    np.savetxt(f"sim-scattered_{pbftype}-tau{tau:g}.txt", scattered)
    np.savetxt(f"sim-profile_{pbftype}-tau{tau:g}.txt", observed)

    logger.info(
        "Wrote final scattered profile to: sim-profile_%s-tau%g.txt",
        pbftype,
        tau,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="simulate", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-n", type=int, default=2048, help="number of profile bins"
    )
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
        choices=KernelRegistry.choices(),
        help="The type of PBF kernel to use during the deconvolution."
        "A '_exp' suffix implies a modified PBF that asymptotes to a thin-screen approximation at large times.",
    )
    parser.add_argument(
        "-t", type=float, default=5.0, help="scattering time scale (in ms)"
    )
    parser.add_argument(
        "-p", type=float, default=100.0, help="pulsar period (in ms)"
    )
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
        default="time",
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
    logger.info("Time sample: %g ms", time_sample)
    # Figure out the dispersion smearing in the worst case (i.e. in the lowest channel), and then determine the
    # nominal width of the restoring function
    chan_bw = args.bw / args.nchan

    logger.debug("Frequency channel size: %g MHz", chan_bw * 1000)
    lochan = args.freq - (args.bw / 2)
    hichan = lochan + chan_bw

    logger.debug(
        "Lowest channel edges: %g-%g MHz", lochan * 1000, hichan * 1000
    )
    dmdelay = dm_delay(args.dm, lochan, hichan)
    logger.info("Dispersion smearing in lowest channel: %g ms", dmdelay)

    restoring_width = np.sqrt(time_sample**2 + dmdelay**2)
    logger.info("Restoring function width: %g ms", restoring_width)

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
        dm=args.dm,
        freq=args.freq,
        bw=args.bw,
        save=args.saveplot,
    )

    if args.write:
        write_data(i, k, s, o, args.k, args.t)


if __name__ == "__main__":
    main()
