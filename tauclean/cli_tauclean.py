#! /usr/bin/env python

import argparse
import logging
import multiprocessing as mp
import sys

import numpy as np

from . import plotting
from .domain.clean_api import clean
from .domain.figures_of_merit import TauSearchAnalyzer
from .domain.kernels import KernelRegistry, get_kernel
from .domain.response import dm_delay, get_instrumental_response, get_restoring_function

# Set up the logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter(
    "%(asctime)s [pid %(process)d] :: %(name)-22s [%(lineno)d] :: "
    "%(levelname)s - %(message)s"
)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tau_search_analyzer = TauSearchAnalyzer(logger=logger)


def main():
    parser = argparse.ArgumentParser(
        prog="tauclean", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    obs_group = parser.add_argument_group(
        "Observing and de-dispersion details"
    )

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

    obs_group.add_argument(
        "-r",
        "--native_dt",
        metavar="NATIVE_TIME_RES",
        type=float,
        default=100.0,
        help="Native time resolution of output data from back-end (pre-folding), in microseconds.",
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
        choices=KernelRegistry.choices(),
        help="The type of PBF kernel to use during the deconvolution."
        "A '_exp' suffix implies a modified PBF that asymptotes to a thin-screen approximation at large times.",
    )

    clean_group.add_argument(
        "-o",
        "--onpulse",
        metavar="'START END' or 'auto'",
        type=str,
        default="auto",
        help="""Boundaries of the on-pulse region. Format as 'START END', or if set to 'auto',
        will automatically compute on- and off-pulse regions.""",
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
        default=0.01,
        help="Loop gain (scaling factor << 1) used to weight component subtraction. Values around 0.01 are empirically good.",
    )

    clean_group.add_argument(
        "--iterlim",
        metavar="N",
        type=int,
        default=100000,
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
        "--nowrite",
        action="store_true",
        default=False,
        help="Do not write reconstructed profiles or clean components disk.",
    )

    other_group.add_argument(
        "--noplot_r",
        action="store_true",
        default=False,
        help="Do not produce reconstruction or residual plots.",
    )

    other_group.add_argument(
        "--noplot_f",
        action="store_true",
        default=False,
        help="Do not produce figure-of-merit plots.",
    )

    other_group.add_argument(
        "--truth",
        type=float,
        default=None,
        help="Truth value (for debugging).",
    )

    args = parser.parse_args()

    execute_tauclean(args)


def execute_tauclean(args):
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
            "Will search %s scattering time scales, %s-%s ms, inclusive",
            ntaus,
            tau_min,
            tau_max,
        )
    logger.info("Loop gain factor = %s", args.gain)

    chan_bw = args.bw / args.nchan
    chan_cntr_low = args.freq - args.bw / 2
    chan_ledge_lo = chan_cntr_low - chan_bw / 2
    chan_ledge_hi = chan_cntr_low + chan_bw / 2
    dm_smear_width = dm_delay(
        args.dm, chan_ledge_lo, chan_ledge_hi
    )  # in ms
    prof_bin_width = args.period / nbins  # in ms
    backend_dt_width = args.native_dt / 1000  # in ms
    post_dt_width = 0  # in ms

    logger.info("Native profile time resolution: %g ms", prof_bin_width)
    if not args.coherent:
        logger.info("DM smearing within lowest channel: %g ms", dm_smear_width)

    inst_resp_fn, inst_resp_width = get_instrumental_response(
        data,
        args.period,
        r_dm_width=dm_smear_width,
        r_pb_width=prof_bin_width,
        r_av_width=backend_dt_width,
        r_pd_width=post_dt_width,
        fast=True,
    )
    restoring_fn = get_restoring_function(
        data, args.period, inst_resp_width
    )

    logger.info(
        "Effective instrumental response width: %g ms", inst_resp_width
    )
    logger.info("Restoring function (Gaussian) width: %g ms", inst_resp_width)

    kernel = get_kernel(args.kernel)

    # Setup for the deconvolution (potentially distributed across multiple processes)
    clean_kwargs = dict(
        period=args.period,
        gain=args.gain,
        kernel=kernel,
        iter_limit=args.iterlim,
        threshold=args.thresh,
        inst_resp_func=inst_resp_fn,
        rest_func=restoring_fn,
        onpulse_estimator=args.onpulse,
    )

    # Create a master list that will contain the output for each trial
    result_list = []

    # Define a small callback function that simply appends output from Pool workers to "master" list
    def log_results(worker_results):
        logger.debug("Finished work for tau=%s", worker_results.tau)
        result_list.append(worker_results)

    logger.info("Starting deconvolution cycles...")
    # Create worker pool, where the number of workers is given by the user, or based on the number of CPUs available
    logger.debug("Creating a pool of %s workers", args.ncpus)
    with mp.Pool(processes=args.ncpus) as pool:
        for tau in taus:
            logger.debug("Started async. job for tau=%g ms", tau)
            pool.apply_async(
                clean, (data, tau), clean_kwargs, callback=log_results
            )
        pool.close()
        pool.join()
    logger.debug("Worker pool closed.")

    # Sort the results based on the trial value of tau
    logger.info("Done. Sorting output...")
    sorted_results = sorted(result_list, key=lambda r: r.tau)

    logger.info("Attempting to determine best tau from figures-of-merit...")
    if ntaus > 1:
        tau_estimate = tau_search_analyzer.estimate_best_tau(sorted_results)
        best = tau_estimate.best_tau
        err = tau_estimate.uncertainty
        if not np.isfinite(err):
            logger.warning(
                "Undefined uncertainty. "
                "Review figures of merit - perhaps adjust your search bounds?"
            )
        if err > best:
            logger.warning(
                "Uncertainty is larger than nominal value. "
                "Review figures of merit - perhaps adjust your search bounds?"
            )
    else:
        fom_set = sorted_results[0].figures_of_merit
        logger.info("f_r ~ positivity: %s", fom_set.positivity)
        logger.info("gamma ~ skewnesss: %s", fom_set.skewness)
        logger.info(
            f"f_c = f_r / gamma: {fom_set.combined}"
        )
        logger.info(
            f"nf ~ consistence: {fom_set.consistence} ({100 * fom_set.consistence / sorted_results[0].nbins_on}%)"
        )

    # Make all of the diagnostic plots and write relevant files to disk
    if not args.noplot_f:
        if len(taus) > 1:
            logger.info("Plotting figures of merit...")
            plotting.plot_figures_of_merit(
                sorted_results,
                true_tau=args.truth,
                best_tau=best,
                best_tau_err=err,
            )

    if not args.noplot_r:
        logger.info("Plotting clean residuals...")
        plotting.plot_clean_residuals(data, sorted_results, period=args.period)

        logger.info("Plotting clean components...")
        plotting.plot_clean_components(sorted_results, period=args.period)

        logger.info("Plotting profile reconstruction...")
        plotting.plot_reconstruction(sorted_results, data, period=args.period)

        logger.info("Done plotting reconstruction.")

    if not args.nowrite:
        logger.debug(
            "Writing output products (reconstruction + clean component list"
        )
        plotting.write_output(sorted_results)


if __name__ == "__main__":
    main()
