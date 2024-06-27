#! /usr/bin/env python
"""
########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
"""

import argparse
import logging
import multiprocessing as mp
import sys

import numpy as np

from . import pbf, plotting, clean, fom

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        prog="tauclean", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    obs_group = parser.add_argument_group("Observing and de-dispersion details")

    parser.add_argument(
        "profile",
        help="The data file containing the folded pulse profile. Expects a single column, one value per line.",
    )

    # Option group for observation and processing details
    obs_group.add_argument(
        "-p",
        "--period",
        metavar="P",
        type=float,
        default=100.0,
        help="Pulsar period (in ms)",
    )

    obs_group.add_argument(
        "--coherent",
        action="store_true",
        default=False,
        help="Whether the data are coherently de-dispersed (affects calculation of effective time "
        "sampling for reconstruction). If yes, DM and frequency options are not required.",
    )

    obs_group.add_argument(
        "-d",
        "--dm",
        metavar="DM",
        type=float,
        default=0.0,
        help="Pulsar dispersion measure (in pc/cm^3).",
    )

    obs_group.add_argument(
        "-f",
        "--freq",
        metavar="FREQ",
        type=float,
        default=1.4,
        help="Centre frequency of central channel (in GHz).",
    )

    obs_group.add_argument(
        "-b",
        "--bw",
        metavar="BW",
        type=float,
        default=0.256,
        help="Observing bandwidth (in GHz).",
    )

    obs_group.add_argument(
        "-n",
        "--nchan",
        metavar="NCHAN",
        type=int,
        default=1024,
        help="Number of frequency channels across the bandwidth.",
    )

    # Option group specifying configuration of deconvolution, and how to perform it (i.e. to search or not)
    clean_group = parser.add_argument_group("Deconvolution options")
    tau_group = clean_group.add_mutually_exclusive_group(required=True)
    tau_group.add_argument(
        "-t",
        "--tau",
        metavar="tau",
        type=float,
        default=None,
        help="Nominal pulse broadening time scale to use when deconvolving profile (in ms)",
    )

    tau_group.add_argument(
        "-s",
        "--search",
        metavar=("MIN", "MAX", "STEP_SIZE"),
        nargs=3,
        type=float,
        default=None,
        help="Pulse broadening time scale search parameters: "
        "minimum tau (in ms), maximum tau (in ms), step (in ms)",
    )

    clean_group.add_argument(
        "-k",
        "--kernel",
        metavar="pbf",
        default="thin",
        choices=["thin", "thick", "uniform", "thick_exp", "uniform_exp"],
        help="The type of PBF kernel to use during the deconvolution."
        "A '_exp' suffix implies a modified PBF that asymptotes to a thin-screen approximation at large times.",
    )

    clean_group.add_argument(
        "-o",
        "--onpulse",
        nargs=2,
        metavar=("start", "end"),
        type=int,
        default=(0, 255),
        help="Boundaries of the on-pulse region.",
    )

    clean_group.add_argument(
        "--thresh",
        metavar="sigma",
        type=float,
        default=3.0,
        help="On-pulse data threshold (units of off-pulse rms noise) to stop cleaning.",
    )

    clean_group.add_argument(
        "-g",
        "--gain",
        metavar="gain",
        type=float,
        default=0.05,
        help="Loop gain (scaling factor << 1) used to weight component subtraction. Values around 0.05 are empirically good.",
    )

    clean_group.add_argument(
        "--iterlim",
        metavar="N",
        type=int,
        default=None,
        help="Limit the number of iterations for each trial value, regardless of convergence factors.",
    )

    clean_group.add_argument(
        "--ncpus",
        type=int,
        default=mp.cpu_count(),
        help="Number of CPUs to use for parallel trial deconvolution.",
    )

    other_group = parser.add_argument_group("Other options")
    other_group.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Increase verbosity and print DEBUG info.",
    )

    other_group.add_argument(
        "--nowrite",
        action="store_true",
        default=False,
        help="Do not write reconstructed profiles or clean components disk.",
    )

    other_group.add_argument(
        "--noplot", action="store_true", default=False, help="Do not produce any plots."
    )

    other_group.add_argument(
        "--truth", type=float, default=None, help="Truth value (for debugging)."
    )

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    if args.debug:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s :: %(name)s :: %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    execute_tauclean(args, logger)


def execute_tauclean(args, logger):
    # Load the data (assumes single column, 1 bin per line)
    data = np.loadtxt(args.profile)
    nbins = len(data)

    # Check tau values and adjust if necessary
    if args.tau is None:
        tau_min = args.search[0]
        tau_max = args.search[1]
        step = args.search[2]

        if tau_max <= tau_min:
            logger.error("Maximum tau is <= minimum tau")
            sys.exit()
        elif tau_min <= 0:
            logger.error("Minimum tau is <= 0 ms")
            sys.exit()
        elif step <= 0:
            logger.error("Step size must be >= 0 ms")
            sys.exit()
        else:
            taus = np.arange(tau_min, tau_max + step, step)
    else:
        taus = [args.tau]
    ntaus = len(taus)
    if ntaus > 1:
        logger.info(
            f"Will search {ntaus} scattering time scales, {tau_min}-{tau_max} ms, inclusive"
        )

    # Calculate the restoring function width
    restoring_width = clean.get_restoring_width(
        nbins,
        period=args.period,
        freq=args.freq,
        bw=args.bw,
        nchan=args.nchan,
        dm=args.dm,
        coherent=False,
    )
    chan_bw = args.bw / args.nchan
    chan_cntr_low = args.freq - args.bw / 2
    chan_ledge_lo = chan_cntr_low - chan_bw / 2
    chan_ledge_hi = chan_cntr_low + chan_bw / 2
    worst_intrachan_smear = clean.dm_delay(args.dm, chan_ledge_lo, chan_ledge_hi)
    logger.info(f"Native profile time resolution: {args.period/nbins:g} ms")
    if not args.coherent:
        logger.info(f"DM smearing within lowest channel: {worst_intrachan_smear:g} ms")
    logger.info(f"Restoring function width: {restoring_width:g} ms")

    # Check the on-pulse boundaries
    onpulse_start = args.onpulse[0]
    onpulse_end = args.onpulse[1]
    if onpulse_end <= onpulse_start:
        logger.error("On-pulse end boundary must be > start boundary")
        sys.exit()
    logger.info(f"Deconvolving data only from bins: {onpulse_start}-{onpulse_end}")

    # Setup for the deconvolution (potentially distributed across multiple processes)
    clean_kwargs = dict(
        period=args.period,
        gain=args.gain,
        pbftype=args.kernel,
        iter_limit=args.iterlim,
        threshold=args.thresh,
        on_start=onpulse_start,
        on_end=onpulse_end,
        rest_width=restoring_width,
    )

    # Create a master list that will contain the output for each trial
    result_list = []

    # Define a small callback function that simply appends output from Pool workers to "master" list
    def log_results(worker_results):
        logger.debug(f"    finished work for tau={worker_results['tau']}")
        result_list.append(worker_results)

    logger.info("Starting deconvolution cycles...")
    # Create worker pool, where the number of workers is given by the user, or based on the number of CPUs available
    pool = mp.Pool(processes=args.ncpus)
    logger.debug("Created pool of {0} workers".format(args.ncpus))

    for t in taus:
        logger.debug("Started async. job for tau={0:g} ms".format(t))
        pool.apply_async(clean.clean, (data, t), clean_kwargs, callback=log_results)

    pool.close()
    pool.join()

    # Sort the results based on the trial value of tau
    logger.info("Done. Sorting output...")
    sorted_results = sorted(result_list, key=lambda r: r["tau"])

    frbest, frerr, fcbest, fcerr = fom.get_error(sorted_results, plot=True)
    logger.info("Attempting to estimate best fit tau and errors")
    logger.info(
        "   based on f_r: {0:.3f} +/- {1:.3f} ({2:.2f}%) ms".format(
            frbest, frerr, 100 * frerr / frbest
        )
    )
    logger.info(
        "   based on f_c: {0:.3f} +/- {1:.3f} ({2:.2f}%) ms".format(
            fcbest, fcerr, 100 * fcerr / fcbest
        )
    )

    if np.isnan(frerr) or np.isnan(fcerr):
        logger.warning(
            "Undefined uncertainties. "
            "Review figures of merit - perhaps expand your search bounds?"
        )

    if (frerr > frbest) or (fcerr > fcbest):
        logger.warning(
            "Uncertainties are larger than nominal values. "
            "Review figures of merit - perhaps expand your search bounds?"
        )

    # Make all of the diagnostic plots and write relevant files to disk
    if not args.noplot:
        if len(taus) > 1:
            logger.debug("Plotting figures of merit...")
            plotting.plot_figures_of_merit(sorted_results, args.truth)

        logger.debug("Plotting clean residuals...")
        plotting.plot_clean_residuals(data, sorted_results, period=args.period)

        logger.debug("Plotting clean components...")
        plotting.plot_clean_components(sorted_results, period=args.period)

        logger.debug("Plotting profile reconstruction...")
        plotting.plot_reconstruction(sorted_results, data, period=args.period)

    if not args.nowrite:
        logger.debug("Writing output products (reconstruction + clean component list")
        plotting.write_output(sorted_results)


if __name__ == "__main__":
    main()
