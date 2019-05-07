"""
Copyright 2019 Bradley Meyers
Licensed under the Academic Free License version 3.0
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_figures_of_merit(results):

    taus = np.array([a["tau"] for a in results])

    params = [np.array([a["fr"] for a in results]),
              np.array([a["gamma"] for a in results]),
              np.array([a["ncc"] for a in results]),
              np.array([a["on_rms"] for a in results]),
              np.array([a["off_rms"] for a in results]) / np.array([a["init_rms"] for a in results]),
              np.array([a["nf"] for a in results]) / np.array([a["nbins"] for a in results])
              ]

    labels = [r"$f_r$", r"$\Gamma$", r"$N_{cc}$", r"$\sigma_{\rm on}$",
              r"$\sigma_{\rm offc}/\sigma_{\rm off}$", r"$N_f / N_{\rm total}$"
              ]

    fig, axs = plt.subplots(ncols=3, nrows=2, sharex="all", figsize=(20, 6))

    for y, ylab, ax in zip(params, labels, axs.flatten()):
        ax.plot(taus, y, marker="o")
        ax.set_ylabel(ylab, fontsize=20)

        if min(y) < 0:
            ax.axhline(0, lw=1, ls=":", color="k")

        if abs(min(y)) > 0 and abs(max(y)/min(y)) > 100:
            ax.set_yscale("symlog")

    for ax in axs.flatten()[3:]:
        ax.set_xlabel(r"$\rm \tau\ (ms)$", fontsize=20)
        ax.set_xlim(0.9*min(taus), 1.1*max(taus))

    plt.subplots_adjust(hspace=0.1, wspace=0.25)
    plt.savefig("tauclean_fom.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # For the purposes of testing, return whether the figure was closed successfully (implying nothing broke)
    return not plt.fignum_exists(fig.number)


def plot_clean_residuals(initial_data, results, period=100.0):

    nbins = len(initial_data)
    x = period * np.linspace(0, 1, nbins)
    dt = period / nbins
    taus = np.array([a["tau"] for a in results])
    residuals = np.array([a["profile"] for a in results])
    off_rms = np.array([a["off_rms"] for a in results])
    off_mean = np.array([a["off_mean"] for a in results])
    thresh = np.array([a["threshold"] for a in results])
    on_start = np.array([a["on_start"] for a in results])
    on_end = np.array([a["on_end"] for a in results])
    pbftype = np.array([a["pbftype"] for a in results])

    pos_thresh = off_mean + thresh * off_rms
    neg_thresh = off_mean - thresh * off_rms

    for i, t in enumerate(taus):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex="all", figsize=plt.figaspect(0.25))
        fig.suptitle(r"Residuals ($\rm \tau = {0:g}\ ms$)".format(t))

        ax1.plot(x, initial_data, label="initial data")
        ax1.plot(x, residuals[i], label="post-clean residuals")
        ax1.fill_between(x, neg_thresh[i], pos_thresh[i], color="0.8")
        ax1.axvline(on_start[i] * dt, color="k")
        ax1.axvline(on_end[i] * dt, color="k")
        ax1.set_xlabel("Time (ms)")
        ax1.legend()

        ax2.plot(x, initial_data, label="initial data")
        ax2.plot(x, residuals[i], label="post-clean residuals")
        ax2.fill_between(x, neg_thresh[i], pos_thresh[i], color="0.8")
        ax2.axvline(on_start[i] * dt, color="k")
        ax2.axvline(on_end[i] * dt, color="k")
        ax2.axhline(0, color="k", ls=":", lw=1)
        ax2.set_xlabel("Time (ms)")
        ax2.legend()
        ax2.set_ylim(-10 * off_rms[i], 10 * off_rms[i])

        plt.savefig("clean_residuals_{0}-tau{1:g}.png".format(pbftype[i], t), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # For the purposes of testing, return whether the figure was closed successfully (implying nothing broke)
    return not plt.fignum_exists(fig.number)


def plot_clean_components(results, period=100.0):

    taus = np.array([a["tau"] for a in results])
    clean_components = np.array([a["cc"] for a in results])
    nunique = np.array([a["ncc"] for a in results])
    ntotal = np.array([a["niter"] for a in results])
    pbftype = np.array([a["pbftype"] for a in results])

    nbins = len(clean_components[0])
    x = period * np.linspace(0, 1, nbins)

    for i, t in enumerate(taus):
        fig, ax = plt.subplots(1, 1)

        ax.vlines(x, ymin=0, ymax=clean_components[i], color="k")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Clean component amplitude")
        title = r"""$\rm \tau = {0:g}\ ms$
total components added = {1}, unique components = {2}""".format(t, ntotal[i], nunique[i])
        ax.set_title(title)

        plt.savefig("clean_components_{0}-tau{1:g}.png".format(pbftype[i], t), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # For the purposes of testing, return whether the figure was closed successfully (implying nothing broke)
    return not plt.fignum_exists(fig.number)


def plot_reconstruction(results, original, period=100.0):

    taus = np.array([a["tau"] for a in results])
    recons = np.array([a["recon"] for a in results])
    residuals = np.array([a["profile"] for a in results])
    pbftype = np.array([a["pbftype"] for a in results])

    nbins = len(recons[0])
    x = period * np.linspace(0, 1, nbins)

    for i, t in enumerate(taus):
        fig, ax = plt.subplots(1, 1)

        ax.plot(x, recons[i] + residuals[i], color="k")
        ax.plot(x, original, color="C1", alpha=0.6)
        ax.axhline(0, ls=":", lw=1, color="k")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Intensity")
        ax.set_title(r"Profile reconstruction for $\rm \tau = {0:g}\ ms$".format(t))

        plt.savefig("reconstruction_{0}-tau{1:g}.png".format(pbftype[i], t), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # For the purposes of testing, return whether the figure was closed successfully (implying nothing broke)
    return not plt.fignum_exists(fig.number)


def write_output(results):

    taus = np.array([a["tau"] for a in results])
    clean_components = np.array([a["cc"] for a in results])
    pbftype = np.array([a["pbftype"] for a in results])
    recons = np.array([a["recon"] for a in results])

    for i, t in enumerate(taus):
        np.savetxt("clean_components_{0}-tau{1:g}.txt".format(pbftype[i], t), clean_components[i])
        np.savetxt("reconstruction_{0}-tau{1:g}.txt".format(pbftype[i], t), recons[i])
