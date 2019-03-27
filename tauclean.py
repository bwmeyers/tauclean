import argparse
import pbf
import clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="tauclean", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("profile", help="The data file containing the folder pulse profile (single column)")

    parser.add_argument("--taumin", type=float, default=1.0,
                        help="The minimum value of tau (in ms) to use in the search")

    parser.add_argument("--taumax", type=float, default=10.0,
                        help="The maximum value of tau (in ms) to use in the search")

    parser.add_argument("-n", "--ntrials", metavar="N", type=int, default=10,
                        help="The number of values to trial, from minimum to maximum tau")

    parser.add_argument("-p", metavar="pbf", default="thin", choices=pbf.__all__,
                        help="The type of PBF to use during the deconvolution. Choices are: {0}".format(pbf.__all__))

    parser.add_argument("-s", "--start", metavar="S", type=float, default=0.0,
                        help="Starting phase (in turns) of on-pulse region")

    parser.add_argument("-e", "--end", metavar="E", type=float, default=1.0,
                        help="End phase (in turns) of on-pulse region")

    parser.add_argument("--dt", metavar="dt", type=float, default=1.0, help="Time resolution (in ms) of profile")

    parser.add_argument("--thresh", metavar="sigma", type=float, default=3.0,
                        help="On-pulse data threshold (units of off-pulse rms noise) to stop cleaning")

    parser.add_argument("-l", "--loop_gain", metavar="L", type=float, default=0.01,
                        help="Loop gain (scaling factor << 1) used to weight component subtraction")

    parser.add_argument("--iterlim", metavar="N", type=int, default=None,
                        help="Limit of the number of iterations for each trial value, regardless of convergence")

    args = parser.parse_args()


