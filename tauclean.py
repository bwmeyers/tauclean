import argparse
import sys
import numpy as np
import multiprocessing as mp
import pbf
import clean

import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="tauclean", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("profile", help="The data file containing the folder pulse profile (single column)")

    tau_group = parser.add_mutually_exclusive_group(required=True)

    tau_group.add_argument("-t", "--tau", metavar="tau", type=float, default=None,
                           help="tau value (in ms) to use when deconvolving")

    tau_group.add_argument("-s", "--search", metavar=("min", "max", "N"), nargs=3, type=float, default=None,
                           help="search parameters to use: minimum tau (in ms), maximum tau (in ms), number of trials")

    parser.add_argument("-k", "--kernel", metavar="pbf", default="thin", choices=pbf.__all__,
                        help="type of PBF kernel to use during the deconvolution")

    parser.add_argument("-b", "--begin", metavar="B", type=int, default=0,
                        help="starting phase bin of on-pulse region")

    parser.add_argument("-e", "--end", metavar="E", type=int, default=255,
                        help="end phase bin of on-pulse region")

    parser.add_argument("--dt", metavar="dt", type=float, default=1.0, help="Time resolution (in ms) of profile")

    parser.add_argument("--thresh", metavar="sigma", type=float, default=3.0,
                        help="on-pulse data threshold (units of off-pulse rms noise) to stop cleaning")

    parser.add_argument("-l", "--loop_gain", metavar="L", type=float, default=0.01,
                        help="loop gain (scaling factor << 1) used to weight component subtraction")

    parser.add_argument("--iterlim", metavar="N", type=int, default=None,
                        help="limit of the number of iterations for each trial value, regardless of convergence")

    args = parser.parse_args()

    data = np.loadtxt(args.profile)

    if args.tau is None:
        tau_min = float(args.search[0])
        tau_max = float(args.search[1])
        ntrials = int(args.search[2])

        if tau_max <= tau_min:
            sys.exit("Maximum tau is <= minimum tau")
        elif tau_min <= 0:
            sys.exit("Minimum tau is <= 0")
        elif ntrials <= 1:
            sys.exit("Require >= 2 trial values")
        else:
            taus = np.linspace(tau_min, tau_max, ntrials)
    else:
        taus = [args.tau]

    clean_kwargs = dict(dt=args.dt, gain=args.loop_gain, pbftype=args.kernel, iter_limit=args.iterlim,
                        threshold=args.thresh, on_start=args.begin, on_end=args.end)

    with mp.Manager() as manager:
        # Create a list that can be seen by all processes
        results = manager.list()
        processes = []

        # Spawn processes, one per trial value of tau
        for t in taus:
            print("Spawning process to clean with tau={0:g}".format(t))
            p = mp.Process(target=clean.clean, args=(data, t, results), kwargs=clean_kwargs)
            p.start()
            processes.append(p)

        # Wait for processes to finish and rejoin the master process
        print("Waiting for processes to finish...")
        for p in processes:
            p.join()
            print("\tProcess {0} done".format(p.pid))
        print("Done")

        # Sort the results based on the trial value of tau
        print("Sorting output...")
        sorted_results = sorted(results, key=lambda r: r['tau'])
        print("Done")

    plt.plot(data)
    plt.plot(sorted_results[0]["profile"])
    plt.plot(sorted_results[0]["recon"])
    plt.show()
