#!/usr/bin/env python3
"""
########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
"""
import logging
import matplotlib.pyplot as plt
from matplotlib.scale import SymmetricalLogScale
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from . import pbf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter(
    "%(asctime)s [pid %(process)d] :: %(name)-22s [%(lineno)d] :: %(levelname)s - %(message)s"
)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def plot_figures_of_merit(results, true_tau=None, best_tau=None, best_tau_err=None):
    taus = np.array([a["tau"] for a in results])
    f_r = np.array([a["fr"] for a in results])
    gamma = np.array([a["gamma"] for a in results])
    f_c = (f_r + gamma) / 2.0
    niter = [a["niter"] for a in results]
    nuniq = [a["ncc"] for a in results]
    r_sigma = np.array([a["total_rms"] for a in results]) / np.array(
        [a["init_off_rms"] for a in results]
    )
    r_phi = np.array([a["nf"] for a in results]) / np.array(
        [a["nbins"] for a in results]
    )

    foms = [
        {
            "name": "f_r",
            "values": np.array(f_r),
            "label": r"$f_r$",
            "use_jerk": True,
            "alt_operation": np.argmin,
            "ylims": None,
        },
        {
            "name": "gamma",
            "values": np.array(gamma),
            "label": r"$\Gamma$",
            "use_jerk": True,
            "alt_operation": np.argmin,
            "ylims": None,
        },
        {
            "name": "f_c",
            "values": np.array(f_c),
            "label": r"$f_c = (f_r + \Gamma)/2$",
            "use_jerk": False,
            "alt_operation": None,
            "ylims": None,
        },
        {
            "name": "r_sigma",
            "values": np.array(r_sigma),
            "label": r"$r_\sigma = \sigma_{\rm offc}/\sigma_{\rm off}$",
            "use_jerk": False,
            "alt_operation": np.argmin,
            "ylims": (1, 3),
        },
        {
            "name": "niter",
            "values": np.array(niter),
            "label": r"$N_{\rm iter}$",
            "use_jerk": False,
            "alt_operation": None,
            "ylims": None,
        },
        {
            "name": "r_phi",
            "values": np.array(r_phi),
            "label": r"$r_\phi=N_f / N_{\rm tot}$",
            "use_jerk": False,
            "alt_operation": np.argmax,
            "ylims": None,
        },
    ]

    min_tau_step = min(np.diff(taus))

    fig, axs = plt.subplots(ncols=3, nrows=2, sharex="all", figsize=(24, 6))
    for fom, ax in zip(foms, axs.flatten()):
        ax.plot(taus, fom["values"], marker="o", color="C0", label="FOM")
        ax.set_ylabel(fom["label"], fontsize=20)

        if fom["use_jerk"]:
            tax = ax.twinx()
            wlen = fom["values"].size // 8 if fom["values"].size // 8 > 3 else 4
            der3 = savgol_filter(
                fom["values"],
                window_length=wlen,
                polyorder=3,
                deriv=3,
            )
            norm_abs_der3 = np.abs(der3) / np.abs(der3).max()
            pidx, _ = find_peaks(
                np.abs(norm_abs_der3),
                prominence=(0.05, None),
                height=(None, None),
            )
            tax.plot(
                taus,
                norm_abs_der3,
                ls=":",
                color="k",
                label="|norm. 3rd deriv.|",
            )
            tax.scatter(
                taus[pidx], norm_abs_der3[pidx], marker="x", color="C1", label="peaks"
            )
            tax.set_ylabel("abs(normalsed 3rd deriv.)")
        elif fom["alt_operation"] != None:
            fn = fom["alt_operation"]
            idx = fn(fom["values"])
            ax.plot(taus[idx], fom["values"][idx], marker="*", ms=10, color="C1")

        if fom["ylims"] != None:
            ax.set_ylim(fom["ylims"])

        if true_tau is not None:
            ax.axvline(true_tau, color="k", ls="--", label="truth", zorder=0.5)

        if best_tau is not None:
            ax.axvline(best_tau, color="r", ls="-.", label="best tau", zorder=0.5)

        if best_tau_err is not None:
            ylims = ax.get_ylim()
            ax.fill_betweenx(
                ylims,
                best_tau - best_tau_err,
                best_tau + best_tau_err,
                color="0.1",
                alpha=0.3,
                label="best tau err.",
                zorder=0.49,
            )
            ax.set_ylim(ylims)

    for ax in axs.flatten()[3:]:
        ax.set_xlabel(r"$\tau\ {\rm (ms)}$", fontsize=20)
        ax.set_xlim(min(taus) - min_tau_step, max(taus) + min_tau_step)
    axs.flatten()[1].set_title("Figures of Merit summary")
    axs.flatten()[0].legend(loc="upper left")

    plt.subplots_adjust(hspace=0.05, wspace=0.4)
    plt.savefig("tauclean_fom.png", bbox_inches="tight")
    plt.close(fig)

    # In this case, also write the figures of merit to a file
    header_fmt = "{0:<7}  {1:<8}  {2:<8} {3:<8} {4:<8}  {5:<5}  {6:<5} {7:<7}\n"
    line_fmt = (
        "{0:7.5f} {1: 8.6f} {2: 8.6f} {3: 8.6f} {4: 8.6f} {5:<5d} {6:<5d} {7:7.5f}\n"
    )
    with open("tauclean_fom.txt", "w") as f:
        f.write(
            header_fmt.format(
                "#tau", "f_r", "gamma", "f_c", "sigma_c", "niter", "nuniq", "nf_frac"
            )
        )
        for i, t in enumerate(taus):
            f.write(
                line_fmt.format(
                    t,
                    f_r[i],
                    gamma[i],
                    f_c[i],
                    r_sigma[i],
                    niter[i],
                    nuniq[i],
                    r_phi[i],
                )
            )

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
    pbftype = np.array([a["pbftype"] for a in results])

    pos_thresh = off_mean + thresh * off_rms
    neg_thresh = off_mean - thresh * off_rms

    for i, t in enumerate(taus):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex="all", figsize=(20, 8))
        fig.suptitle(r"Residuals ($\rm \tau = {0:g}\ ms$)".format(t))

        ax1.plot(x, initial_data, label="initial data")
        ax1.plot(x, residuals[i], label="post-clean residuals")
        ax1.fill_between(
            x,
            neg_thresh[i],
            pos_thresh[i],
            color="0.8",
            label="CLEAN threshold",
        )
        ax1.set_xlabel("Time (ms)")
        ax1.set_xlim(x.min(), x.max())
        ax1.legend()

        ax2.plot(x, initial_data, label="initial data")
        ax2.plot(x, residuals[i], label="post-clean residuals")
        ax2.fill_between(
            x,
            neg_thresh[i],
            pos_thresh[i],
            color="0.8",
            label="CLEAN threshold",
        )
        ax2.axhline(0, color="k", ls=":", lw=1)
        ax2.set_xlabel("Time (ms)")
        ax2.legend()
        ax2.set_ylim(-2 * thresh[i] * off_rms[i], 2 * thresh[i] * off_rms[i])

        plt.savefig(
            "clean_residuals_{0}-tau{1:g}.png".format(pbftype[i], t),
            bbox_inches="tight",
        )
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
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        ax.vlines(x, ymin=0, ymax=clean_components[i], color="k")
        ax.set_ylim(0, None)
        ax.set_xlim(x.min(), x.max())
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Clean component amplitude")
        title = rf"$\tau$ = {t:g} ms,  total components added = {ntotal[i]},  unique components = {nunique[i]}"
        ax.set_title(title)

        plt.savefig(
            "clean_components_{0}-tau{1:g}.png".format(pbftype[i], t),
            bbox_inches="tight",
        )
        plt.close(fig)

    # For the purposes of testing, return whether the figure was closed successfully (implying nothing broke)
    return not plt.fignum_exists(fig.number)


def plot_reconstruction(results, original, period=100.0):
    taus = np.array([a["tau"] for a in results])
    recons = np.array([a["recon"] for a in results])
    restoring = np.array([a["rest_func"] for a in results])
    residuals = np.array([a["profile"] for a in results])
    pbftype = np.array([a["pbftype"] for a in results])

    nbins = len(recons[0])
    x = period * np.linspace(0, 1, nbins)

    for i, t in enumerate(taus):
        try:
            pbffn = getattr(pbf, pbftype[i])
        except AttributeError as e:
            logger.error(e)
            logger.warning(
                f"Cannot find pbf function '{pbftype[i]}'! Assuming 'thin' model."
            )
            pbffn = pbf.thin
        rest = np.roll(restoring[i], -np.argmax(restoring[i]) + len(x) // 40)
        norm_rest = rest / rest.max()
        norm_pbf = pbffn(x, t, x0=x[len(x) // 20])
        norm_pbf = norm_pbf / norm_pbf.max()

        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        ax.plot(
            x,
            (recons[i] * original.max() + residuals[i]),
            color="k",
            label="reconstruction (scaled)",
        )
        ax.plot(
            x,
            original,
            color="k",
            alpha=0.3,
            label="original",
        )
        ax.plot(
            x,
            original.max() * norm_pbf,
            color="C0",
            alpha=0.5,
            ls=":",
            label="PBF (scaled, shifted)",
        )
        ax.plot(
            x,
            original.max() * norm_rest,
            color="C1",
            alpha=0.5,
            ls="--",
            label="restoring function (shifted)",
        )
        ax.axhline(0, ls=":", lw=1, color="k")
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("Time (ms)", fontsize=15)
        ax.set_ylabel("Intensity", fontsize=15)
        ax.set_title(r"Profile reconstruction for $\rm \tau = {0:g}\ ms$".format(t))
        ax.legend(fontsize=15)

        plt.savefig(
            "reconstruction_{0}-tau{1:g}.png".format(pbftype[i], t), bbox_inches="tight"
        )
        plt.close(fig)

    # For the purposes of testing, return whether the figure was closed successfully (implying nothing broke)
    return not plt.fignum_exists(fig.number)


def write_output(results):
    taus = np.array([a["tau"] for a in results])
    clean_components = np.array([a["cc"] for a in results])
    pbftype = np.array([a["pbftype"] for a in results])
    # stack the reconstructed profile with the residuals
    recon_resid = np.array(
        [np.column_stack((a["recon"], a["profile"])) for a in results]
    )

    for i, t in enumerate(taus):
        np.savetxt(
            "clean_components_{0}-tau{1:g}.txt".format(pbftype[i], t),
            clean_components[i],
        )
        np.savetxt(
            "reconstruction_{0}-tau{1:g}.txt".format(pbftype[i], t),
            recon_resid[i],
            header="Recon Residuals",
        )
