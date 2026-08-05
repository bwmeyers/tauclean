"""Object-oriented figures-of-merit models and analyzers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter


@dataclass(frozen=True)
class FigureOfMeritSet:
    """Figures of merit derived from a single CLEAN run."""

    consistence: int
    positivity: float
    skewness: float
    total_rms: float
    initial_off_rms: float
    nbins: int

    @property
    def combined(self) -> float:
        return (self.positivity + self.skewness) / 2.0

    @property
    def residual_ratio(self) -> float:
        return self.total_rms / self.initial_off_rms

    @property
    def consistence_fraction(self) -> float:
        return self.consistence / self.nbins


@dataclass(frozen=True)
class TauEstimate:
    """Selected best tau and uncertainty for a search run."""

    best_tau: float
    uncertainty: float


@dataclass(frozen=True)
class FigureOfMeritSeries:
    """Search-space values for one FOM across tau trials."""

    name: str
    values: np.ndarray
    label: str
    use_jerk: bool
    alt_operation: object | None


class FigureOfMeritEvaluator:
    """Evaluate per-run figures of merit."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def consistence(
        self,
        profile: np.ndarray,
        off_rms: float,
        off_mean: float = 0,
        threshold: float = 3.0,
    ) -> int:
        consistent_points = abs(profile - off_mean) <= threshold * off_rms
        return int(np.sum(consistent_points.astype(int)))

    def positivity(
        self,
        residuals: np.ndarray,
        off_rms: float,
        m: float = 1.0,
        x: float = 1.5,
    ) -> float:
        mask = np.zeros_like(residuals)
        mask[residuals < -x * off_rms] = 1

        if np.all(residuals == 0):
            return np.nan

        return float((m / (len(residuals) * off_rms**2)) * np.sum(mask * residuals**2))

    def skewness(
        self, clean_components: np.ndarray, pulsar_period: float = 100.0
    ) -> float:
        component_times = pulsar_period * np.linspace(0, 1, len(clean_components))
        moment_1 = np.average(component_times, weights=clean_components)
        moment_2 = np.average(
            (component_times - moment_1) ** 2, weights=clean_components
        )
        moment_3 = np.average(
            (component_times - moment_1) ** 3, weights=clean_components
        )

        if np.count_nonzero(clean_components) == 1:
            self.logger.warning("Clean components skewness is undefined. Setting to 0.")
            return 0.0
        return float(moment_3 / (moment_2**1.5))

    def evaluate(
        self,
        profile: np.ndarray,
        clean_components: np.ndarray,
        pulsar_period: float,
        total_rms: float,
        initial_off_rms: float,
        nbins: int,
        off_rms: float,
        off_mean: float = 0.0,
        threshold: float = 3.0,
    ) -> FigureOfMeritSet:
        consistence = self.consistence(
            profile=profile,
            off_rms=off_rms,
            off_mean=off_mean,
            threshold=threshold,
        )
        positivity = self.positivity(profile, off_rms)
        skewness = self.skewness(clean_components, pulsar_period=pulsar_period)
        return FigureOfMeritSet(
            consistence=consistence,
            positivity=positivity,
            skewness=skewness,
            total_rms=total_rms,
            initial_off_rms=initial_off_rms,
            nbins=nbins,
        )


class TauSearchAnalyzer:
    """Determine best tau from a sequence of CLEAN results."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def build_series(self, results) -> tuple[np.ndarray, list[FigureOfMeritSeries]]:
        taus = np.array([result.tau for result in results])
        f_r = np.array([result.figures_of_merit.positivity for result in results])
        gamma = np.array([result.figures_of_merit.skewness for result in results])
        f_c = (f_r + gamma) / 2.0
        r_sigma = np.array(
            [result.figures_of_merit.residual_ratio for result in results]
        )
        r_phi = np.array(
            [result.figures_of_merit.consistence_fraction for result in results]
        )

        return taus, [
            FigureOfMeritSeries("f_r", f_r, r"$f_r$", True, np.argmin),
            FigureOfMeritSeries("gamma", gamma, r"$\Gamma$", True, np.argmin),
            FigureOfMeritSeries("f_c", f_c, r"$f_c = (f_r + \Gamma)/2$", False, None),
            FigureOfMeritSeries(
                "r_sigma",
                r_sigma,
                r"$r_\sigma = \sigma_{\rm offc}/\sigma_{\rm off}$",
                False,
                np.argmin,
            ),
            FigureOfMeritSeries(
                "r_phi",
                r_phi,
                r"$r_\phi=N_f / N_{\rm tot}$",
                False,
                np.argmax,
            ),
        ]

    def estimate_best_tau(
        self,
        results,
        norm_fom_peak_height: float | None = None,
        norm_fom_peak_prominance: float | None = None,
        smoothing_window_size: int | None = None,
        fom_weights: dict | None = None,
    ) -> TauEstimate:
        taus, fom_series = self.build_series(results)
        fom_names = [series.name for series in fom_series]

        default_fom_weights = {
            "f_r": 1.0,
            "gamma": 0.2,
            "f_c": 0.0,
            "r_sigma": 0.5,
            "r_phi": 0.5,
        }
        if fom_weights is None:
            fom_weights = default_fom_weights
        else:
            for key, default_value in default_fom_weights.items():
                if key in fom_weights:
                    if not (
                        isinstance(fom_weights[key], float)
                        and 0 <= fom_weights[key] <= 1
                    ):
                        self.logger.warning(
                            "Weight for FOM=%s is not a float in the range 0 <= x <= 1! Setting to 0.",
                            key,
                        )
                        fom_weights[key] = 0
                else:
                    fom_weights[key] = default_value

        savgol_polyorder = 3
        savgol_derorder = 3
        multi_peak_flag = 0
        fom_tau_estimates = []

        for series in fom_series:
            if series.name in ["r_phi", "r_sigma"]:
                fn = series.alt_operation
                best_tau_fom = taus[fn(series.values)]
                fom_tau_estimates.append(best_tau_fom)
                self.logger.info(
                    "Best tau from metric=%7s is: %.2f ms",
                    series.name,
                    np.squeeze(best_tau_fom),
                )
                continue

            self.logger.debug(
                "Finding 'best' tau from FOM=%7s via 3rd deriv.", series.name
            )

            current_window = smoothing_window_size
            if current_window is None:
                self.logger.debug(
                    "No savgol_filter window size provided, choosing sensible value based on FOM series length."
                )
                current_window = len(series.values) // 8
                if current_window <= savgol_polyorder:
                    current_window = savgol_polyorder + 1
                self.logger.debug("Window size set to %d bins", current_window)

            max_window = series.values.size
            if max_window % 2 == 0:
                max_window -= 1

            if max_window <= savgol_polyorder:
                self.logger.warning(
                    "Too few FOM points (%d) for derivative-based peak finding for %s; falling back to heuristic selection.",
                    series.values.size,
                    series.name,
                )
                if series.alt_operation is not None:
                    fn = series.alt_operation
                    best_tau_fom = taus[fn(series.values)]
                    fom_tau_estimates.append(np.squeeze(best_tau_fom))
                else:
                    fom_weights.pop(series.name, None)
                continue

            current_window = min(current_window, max_window)
            if current_window % 2 == 0:
                current_window -= 1
            if current_window <= savgol_polyorder:
                current_window = savgol_polyorder + 2

            if current_window > series.values.size:
                self.logger.error(
                    "Smoothing window size (%d) is greater than the number of FOM values (%d)!",
                    current_window,
                    series.values.size,
                )
                current_window = max_window

            deriv = np.array(
                savgol_filter(
                    series.values,
                    window_length=current_window,
                    polyorder=savgol_polyorder,
                    deriv=savgol_derorder,
                )
            )
            norm_deriv = deriv / deriv.max()
            pidx, _ = find_peaks(
                np.abs(norm_deriv),
                prominence=(0.05, norm_fom_peak_prominance),
                height=(None, norm_fom_peak_height),
            )

            if len(pidx) == 1:
                best_tau_fom = taus[pidx]
                fom_tau_estimates.append(np.squeeze(best_tau_fom))
                self.logger.info(
                    "Best tau from metric=%7s is: %.2f ms",
                    series.name,
                    np.squeeze(best_tau_fom),
                )
            elif len(pidx) > 1:
                self.logger.debug(
                    "Multiple peaks in FOM found, taking mean of first two instances..."
                )
                multi_peak_flag += 1
                best_tau_fom = np.mean(taus[pidx[:1]])
                fom_tau_estimates.append(np.squeeze(best_tau_fom))
                self.logger.info(
                    "Best tau from metric=%7s is: %.2f ms",
                    series.name,
                    np.squeeze(best_tau_fom),
                )
            else:
                self.logger.warning(
                    "Unable to find peaks in the FOM (%s) derivative.",
                    series.name,
                )
                self.logger.warning(
                    "Resorting to heuristic selection (~ underestimates)."
                )
                if series.alt_operation is not None:
                    fn = series.alt_operation
                    best_tau_fom = taus[fn(series.values)]
                    fom_tau_estimates.append(best_tau_fom)
                else:
                    self.logger.debug(
                        "Excluding FOM=%s from further analysis.", series.name
                    )
                    fom_weights.pop(series.name, None)

        if multi_peak_flag > 0:
            self.logger.info(
                "There were %d FOMs with >1 peaks, so the mean of the first two peaks was used in each instance.",
                multi_peak_flag,
            )

        fom_tau_estimates = np.array(fom_tau_estimates)
        weights = np.array(
            [fom_weights[name] for name in fom_names if name in fom_weights]
        )
        fom_wt_mean_tau = np.average(fom_tau_estimates, weights=np.array(weights))
        fom_wt_std_tau = np.sqrt(
            np.average((fom_tau_estimates - fom_wt_mean_tau) ** 2, weights=weights)
        )
        d_tau = taus[1] - taus[0]
        wt_err = np.sqrt(fom_wt_std_tau**2 + d_tau**2)

        self.logger.info(
            "Best overall tau = %.2f +/- %.2f ms  (weighted mean, weighted error)",
            fom_wt_mean_tau,
            wt_err,
        )
        return TauEstimate(best_tau=float(fom_wt_mean_tau), uncertainty=float(wt_err))
