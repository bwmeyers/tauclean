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


@dataclass(frozen=True)
class FomPeakSearchResult:
    """Outcome of the 3rd-derivative peak search for one FOM series."""

    tau_estimate: float | None
    peak_indices: np.ndarray
    norm_derivative: np.ndarray | None
    multi_peak: bool
    used_heuristic: bool


def find_fom_tau_peaks(
    taus: np.ndarray,
    values: np.ndarray,
    *,
    alt_operation=None,
    smoothing_window_size: int | None = None,
    prominence_bounds: tuple[float | None, float | None] = (0.1, None),
    height_bounds: tuple[float | None, float | None] = (None, None),
    polyorder: int = 3,
    deriv_order: int = 3,
    logger: logging.Logger | None = None,
    name: str = "",
) -> FomPeakSearchResult:
    """Locate 3rd-derivative peaks in a FOM series and estimate its 'best' tau.

    This is the single canonical implementation of the smoothing/derivative/
    peak-finding logic used to pick a per-FOM tau estimate; both
    ``TauSearchAnalyzer.estimate_best_tau`` and ``plotting.plot_figures_of_merit``
    call this function so their results stay consistent.
    """
    log = logger or logging.getLogger(__name__)

    current_window = smoothing_window_size
    if current_window is None:
        current_window = len(values) // 8
        if current_window <= polyorder:
            current_window = polyorder + 1

    max_window = values.size
    if max_window % 2 == 0:
        max_window -= 1

    if max_window <= polyorder:
        log.warning(
            "Too few FOM points (%d) for derivative-based peak finding for "
            "%s; falling back to heuristic selection.",
            values.size,
            name,
        )
        tau_estimate = None
        if alt_operation is not None:
            tau_estimate = float(np.squeeze(taus[alt_operation(values)]))
        return FomPeakSearchResult(
            tau_estimate, np.array([], dtype=int), None, False, True
        )

    current_window = min(current_window, max_window)
    if current_window % 2 == 0:
        current_window -= 1
    if current_window <= polyorder:
        current_window = polyorder + 2
    if current_window > values.size:
        log.error(
            "Smoothing window size (%d) is greater than the number of FOM "
            "values (%d)!",
            current_window,
            values.size,
        )
        current_window = max_window

    deriv = np.array(
        savgol_filter(
            values,
            window_length=current_window,
            polyorder=polyorder,
            deriv=deriv_order,
        )
    )
    deriv_abs_max = np.abs(deriv).max()
    norm_deriv = deriv / deriv_abs_max if deriv_abs_max > 0 else np.zeros_like(deriv)

    pidx, _ = find_peaks(
        np.abs(norm_deriv),
        prominence=prominence_bounds,
        height=height_bounds,
    )

    if len(pidx) == 1:
        return FomPeakSearchResult(
            float(np.squeeze(taus[pidx])), pidx, np.abs(norm_deriv), False, False
        )
    elif len(pidx) > 1:
        log.warning("Multiple peaks in the FOM (%s) derivative. Using first prominent peak.", name)
        return FomPeakSearchResult(
            float(np.squeeze(taus[pidx[0]])), pidx, np.abs(norm_deriv), True, False
        )
        # return FomPeakSearchResult(
        #     float(np.mean(taus[pidx[:2]])), pidx, np.abs(norm_deriv), True, False
        # )

    log.warning("Unable to find peaks in the FOM (%s) derivative.", name)
    tau_estimate = None
    used_heuristic = False
    if alt_operation is not None:
        log.warning("Resorting to heuristic selection (~ underestimates).")
        tau_estimate = float(np.squeeze(taus[alt_operation(values)]))
        used_heuristic = True
    return FomPeakSearchResult(tau_estimate, pidx, np.abs(norm_deriv), False, used_heuristic)


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

        if np.all(residuals == 0) or off_rms == 0:
            self.logger.warning(
                "Positivity FOM is undefined (off_rms=%g); setting to NaN.",
                off_rms,
            )
            return np.nan

        return float(
            (m / (len(residuals) * off_rms**2)) * np.sum(mask * residuals**2)
        )

    def skewness(
        self, clean_components: np.ndarray, pulsar_period: float = 100.0
    ) -> float:
        component_times = pulsar_period * np.linspace(
            0, 1, len(clean_components)
        )
        moment_1 = np.average(component_times, weights=clean_components)
        moment_2 = np.average(
            (component_times - moment_1) ** 2, weights=clean_components
        )
        moment_3 = np.average(
            (component_times - moment_1) ** 3, weights=clean_components
        )

        if np.count_nonzero(clean_components) == 1:
            self.logger.warning(
                "Clean components skewness is undefined. Setting to 0."
            )
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

    def build_series(
        self, results
    ) -> tuple[np.ndarray, list[FigureOfMeritSeries]]:
        taus = np.array([result.tau for result in results])
        f_r = np.array(
            [result.figures_of_merit.positivity for result in results]
        )
        gamma = np.array(
            [result.figures_of_merit.skewness for result in results]
        )
        f_c = (f_r + gamma) / 2.0
        r_sigma = np.array(
            [result.figures_of_merit.residual_ratio for result in results]
        )
        r_phi = np.array(
            [
                result.figures_of_merit.consistence_fraction
                for result in results
            ]
        )

        return taus, [
            FigureOfMeritSeries("f_r", f_r, r"$f_r$", True, np.argmin),
            FigureOfMeritSeries("gamma", gamma, r"$\Gamma$", True, np.argmin),
            FigureOfMeritSeries(
                "f_c", f_c, r"$f_c = (f_r + \Gamma)/2$", False, None
            ),
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
            "r_phi": 0.8,
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

        multi_peak_flag = 0
        fom_tau_estimates = []

        for series in fom_series:
            self.logger.debug(
                "Finding 'best' tau from FOM=%7s via 3rd deriv.", series.name
            )

            result = find_fom_tau_peaks(
                taus,
                series.values,
                alt_operation=series.alt_operation,
                smoothing_window_size=smoothing_window_size,
                prominence_bounds=(0.1, norm_fom_peak_prominance),
                height_bounds=(None, norm_fom_peak_height),
                logger=self.logger,
                name=series.name,
            )

            if result.multi_peak:
                multi_peak_flag += 1

            if result.tau_estimate is not None:
                fom_tau_estimates.append(result.tau_estimate)
                self.logger.info(
                    "Best tau from metric=%7s is: %.2f ms (wt=%g)",
                    series.name,
                    result.tau_estimate,
                    fom_weights[series.name],
                )
            else:
                self.logger.debug(
                    "Excluding FOM=%s from further analysis.", series.name
                )
                fom_weights.pop(series.name, None)

        if multi_peak_flag > 0:
            self.logger.info(
                "There were %d FOMs with >1 peaks.",
                multi_peak_flag,
            )

        fom_tau_estimates = np.array(fom_tau_estimates)
        weights = np.array(
            [fom_weights[name] for name in fom_names if name in fom_weights]
        )
        fom_wt_mean_tau = np.average(
            fom_tau_estimates, weights=np.array(weights)
        )
        fom_wt_std_tau = np.sqrt(
            np.average(
                (fom_tau_estimates - fom_wt_mean_tau) ** 2, weights=weights
            )
        )
        d_tau = taus[1] - taus[0]
        wt_err = np.sqrt(fom_wt_std_tau**2 + d_tau**2)

        self.logger.info(
            "Best overall tau = %.2f +/- %.2f ms  (weighted mean, weighted error)",
            fom_wt_mean_tau,
            wt_err,
        )
        return TauEstimate(
            best_tau=float(fom_wt_mean_tau), uncertainty=float(wt_err)
        )
