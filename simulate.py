import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import pbf
from scipy.integrate import simps
from clean import gaussian

# Set the seed for numpy's random functions so that the same result can be retrieved each time
np.random.seed(12345)


def create_intrinsic_pulse(position, width, amps, nbins=2048, dt=1.0):
    x = dt * np.linspace(0, nbins, nbins)

    f = np.zeros_like(x)

    # If a list of positions have been given, but only one value for the width, then create an array of widths that
    # are all the same
    if isinstance(position, list) and (not isinstance(width, list) or len(width) == 1):
        print("Will use same width for each component")
        width = np.repeat(width, len(position))

    if isinstance(position, list) and (not isinstance(amps, list) or len(amps) == 1):
        print("Will use same amplitude for each component")
        amps = np.repeat(amps, len(position))

    # For each position and width pair, add these components to the intrinsic pulse profile
    for p, w, a in zip(position, width, amps):
        print("added gaussian comp.")
        g = gaussian(x, p * dt, w * dt)
        f += a * (g / g.max())

    # Normalise the output to have a maximum value of unity
    f = f / f.max()

    return f


def create_scattered_profile(intrinsic, tau, pbftype="thin", dt=1.0, nrot=10, snr=500.0):
    nbins = len(intrinsic)

    x = dt * np.linspace(0, nbins, nbins)
    pbf_x = dt * np.linspace(0, nrot, nrot * nbins)

    # Decide which PBF model to use
    if pbftype == "thin":
        h = pbf.thin(pbf_x, tau)
    elif pbftype == "thick":
        h = pbf.thick(pbf_x, tau)
    elif pbftype == "uniform":
        h = pbf.uniform(pbf_x, tau)
    else:
        print("Invalid PBF type requested ({0})".format(pbftype))
        print("Defaulting to thin screen...")
        h = pbf.thin(pbf_x, tau)

    h = np.roll(h, int(0.05 * nbins))

    # The observed pulse shape is the convolution of:
    # - the true signal,
    # - the scattering kernel, and
    # - some Gaussian radiometer noise
    # Here we do the mode="full" convolution so that the complete shape is convolved and we don't end up with sharp edge
    # effects in the final profile that depend on where the shapes are defined (as in the case of mode="same")
    scattered = np.convolve(intrinsic, h, mode="full")

    offset = np.argmax(intrinsic) - np.argmax(scattered)
    scattered = np.roll(scattered, offset)[:nrot*nbins]

    # Now, split the convolved pulse into 'nrot' profiles of 'nbins' and sum them to create the pulse profile.
    # This effectively mimics the idea of folding a pulse profile where the scattering kernel is not necessarily
    # contained within one pulsar rotation.
    scattered = np.sum(np.split(scattered, nrot), axis=0)
    scattered = scattered / simps(y=scattered, x=x)

    # And do the same for the PBF
    h = np.sum(np.split(h, nrot), axis=0)
    h = h / simps(y=h, x=x)  # re-normalise to unit area

    # Add noise to produce a profile with approximately the signal-to-noise ratio desired
    observed = np.copy(scattered) + np.random.normal(0, scattered.max() / snr, scattered.size)

    return h, scattered, observed


def plot_simulated(intrinsic, kernel, scattered, observed, tau, pbftype, snr, dt=1.0, xunit="time", save=False):

    nbins = len(intrinsic)

    if xunit == "phase":
        x = np.linspace(0, 1, nbins)
        xlab = "Phase"
    elif xunit == "time":
        x = dt * np.linspace(0, nbins, nbins)
        xlab = "Time (ms)"
    elif xunit == "bins":
        x = np.linspace(0, nbins, nbins).astype(int)
        xlab = "Bins"
    else:
        print("Unknown x-unit: {0}".format(xunit))
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
    ax_obs.set_title(r"Observed pulse profile (noise added, $\rm SNR \approx {0}$)".format(snr))
    ax_obs.axhline(0, ls="--", color="r", lw=1)
    ax_obs.set_xlim(0, x.max())
    step = x.max() / 16.0
    ax_obs.set_xticks(np.arange(0, x.max() + step, step))
    ax_obs.grid(True)
    ax_obs.set_xlabel(xlab)

    if save:
        plt.savefig("simulated-profile_{0}-tau{1:g}.png".format(pbftype, tau), dpi=300, bbox_inches="tight")
    else:
        plt.show()


def write_data(intrinsic, kernel, scattered, observed, pbftype, tau):

    np.savetxt("sim-intrinsic_{0}-tau{1:g}.txt".format(pbftype, tau), intrinsic)
    np.savetxt("sim-kernel_{0}-tau{1:g}.txt".format(pbftype, tau), kernel)
    np.savetxt("sim-scattered_{0}-tau{1:g}.txt".format(pbftype, tau), scattered)
    np.savetxt("sim-noisy_{0}-tau{1:g}.txt".format(pbftype, tau), observed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="simulate", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-n", type=int, default=2048, help="number of profile bins")
    parser.add_argument("-p", nargs="+", type=int, help="centre positions of gaussian components (in bins)")
    parser.add_argument("-w", nargs="+", type=float, help="widths (std. dev.) of gaussian components (in bins)")
    parser.add_argument("-a", nargs="+", type=float,
                        help="amplitudes of gaussian components (will be normalised so that the maximum value is 1)")
    parser.add_argument("-k", default="thin",
                        help="PBF kernel type", choices=pbf.__all__)
    parser.add_argument("-t", type=float, default=0.05, help="scattering time scale (in ms)")
    parser.add_argument("--dt", type=float, default=1.0, help="time sample per bin (in ms)")
    parser.add_argument("-s", type=float, default=500.0, help="Desired signal-to-noise ratio")
    parser.add_argument("-x", default="bins", choices=["time", "phase", "bins"], help="plot x-axis units")
    parser.add_argument("--saveplot", action="store_true", default=False,
                        help="Switch to save plot to disk rather than just show")
    parser.add_argument("--write", action="store_true", default=False,
                        help="Write the data (intrinsic, kernel, convolution and noisy) to files")

    args = parser.parse_args()

    if len(args.w) != len(args.p):
        print("Provided different number of widths than positions, selecting first width")
        args.w = args.w[0]

    if len(args.a) != len(args.p):
        print("Provided different number of amplitudes than positions, selecting first amplitude")
        args.a = args.a[0]

    i = create_intrinsic_pulse(args.p, args.w, args.a, nbins=args.n, dt=args.dt)
    k, s, o = create_scattered_profile(i, args.t, pbftype=args.k, dt=args.dt, nrot=10, snr=args.s)

    plot_simulated(i, k, s, o, args.t, args.k, snr=args.s, dt=args.dt, xunit=args.x, save=args.saveplot)

    if args.write:
        write_data(i, k, s, o, args.k, args.t)
