"""Canonical object-oriented cleaner service."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve

from .components import ComponentHistory, SubtractedComponent
from .fom import FigureOfMeritEvaluator, FigureOfMeritSet
from .kernels import Kernel
from .noise import AutoWindowNoiseEstimator, UserDefinedOnPulseNoiseEstimator
from .profile import ProfileData
from .response import gaussian

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fmt = logging.Formatter(
        "%(asctime)s [pid %(process)d] :: %(name)-22s [%(lineno)d] :: "
        "%(levelname)s - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


@dataclass
class CleanResult:
    """Typed output for a single CLEAN run at one tau value."""

    profile: np.ndarray
    init_off_rms: float
    init_on_rms: float
    nbins: int
    nbins_on: int
    nbins_off: int
    rest_func: np.ndarray
    inst_resp_func: np.ndarray
    tau: float
    pbftype: str
    niter: int
    cc: np.ndarray
    ncc: int
    off_bins: np.ndarray
    on_bins: np.ndarray
    off_rms: float
    off_mean: float
    on_rms: float
    on_mean: float
    total_mean: float
    total_rms: float
    recon: np.ndarray
    threshold: float
    profile_data: ProfileData
    figures_of_merit: FigureOfMeritSet
    component_history: ComponentHistory | None = None

    @property
    def nf(self) -> int:
        return self.figures_of_merit.consistence

    @property
    def fr(self) -> float:
        return self.figures_of_merit.positivity

    @property
    def gamma(self) -> float:
        return self.figures_of_merit.skewness


@dataclass
class Cleaner:
    """Service class for CLEAN deconvolution at one trial tau."""

    period: float
    rest_func: np.ndarray | None = None
    inst_resp_func: np.ndarray | None = None
    gain: float = 0.05
    threshold: float = 3.0
    kernel: Kernel | None = None
    iter_limit: int = 1000
    onpulse_estimator: str = "auto"
    track_components: bool = False
    logger: logging.Logger | None = None

    def run(self, data: np.ndarray | ProfileData, tau: float) -> CleanResult:
        active_logger = self.logger or logger
        fom_evaluator = FigureOfMeritEvaluator(logger=active_logger)

        if isinstance(data, ProfileData):
            profile_data = data
        else:
            profile_data = ProfileData(
                samples=np.asarray(data), period=self.period
            )

        if self.kernel is None:
            raise ValueError("Cleaner requires a kernel object")

        period = profile_data.period
        nbins = profile_data.nbins
        prof_dt = period / nbins
        x = np.linspace(0, 1, nbins) * period

        active_logger.debug("Computing PBF template for tau=%g ms", tau)
        filter_guess = self.kernel(x, tau)
        pbftype = self.kernel.name

        active_logger.debug(
            "Estimating initial profile statistics for tau=%g ms", tau
        )
        if self.onpulse_estimator == "auto":
            estimator = AutoWindowNoiseEstimator()
        else:
            estimator = UserDefinedOnPulseNoiseEstimator.from_string(
                self.onpulse_estimator
            )

        profile_data.initialize_noise(estimator)
        off_pulse_bins = profile_data.off_bins
        on_pulse_bins = profile_data.on_bins
        if off_pulse_bins is None or on_pulse_bins is None:
            raise ValueError("Profile noise regions were not initialized")

        if (
            profile_data.baseline_corrected is None
            or profile_data.initial_noise is None
        ):
            raise ValueError("Profile noise properties are not initialized")

        profile = np.copy(profile_data.baseline_corrected)
        off_pulse = profile_data.off_pulse(profile)
        on_pulse = profile_data.on_pulse(profile)
        init_off_rms = profile_data.initial_noise.off_rms
        init_on_rms = profile_data.initial_noise.on_rms

        clean_components = np.zeros_like(profile)
        delta = np.zeros_like(profile)
        delta[delta.size // 2] = 1.0

        inst_resp_func = self.inst_resp_func
        rest_func = self.rest_func
        if inst_resp_func is None:
            active_logger.warning(
                "No valid instrumental response function defined. "
                "Assuming a delta function."
            )
            inst_resp_func = delta

        active_logger.debug(
            "instr. response function area: %f (should be 1)",
            np.trapz(x=x, y=inst_resp_func),
        )

        if rest_func is None:
            active_logger.warning(
                "No valid restoring function defined. Assuming a Gaussian with "
                "sigma = 2x profile time resolution."
            )
            rest_func = gaussian(x, x[x.size // 2], 2 * prof_dt)

        active_logger.debug(
            "restoring function area: %f (should be 1)",
            np.trapz(x=x, y=rest_func),
        )

        preconv = convolve(inst_resp_func, filter_guess, mode="full")
        preconv = preconv[nbins // 2 : -nbins // 2 + 1]
        preconv = preconv / np.trapz(preconv)

        loop = True
        niter = 0
        component_history = (
            ComponentHistory() if self.track_components else None
        )

        active_logger.debug("Initiating clean loop for tau=%g ms", tau)
        while loop:
            if (self.iter_limit is not None) and (niter >= self.iter_limit):
                active_logger.warning(
                    "Reached iteration limit for tau=%g ms", tau
                )
                break
            niter += 1

            imax = np.argmax(profile)
            dmax = profile[imax]
            cc_amp = dmax * self.gain

            temp_clean_comp = np.zeros_like(clean_components)
            temp_clean_comp[imax] = cc_amp
            clean_components[imax] += cc_amp

            component1 = convolve(temp_clean_comp, preconv, mode="full")
            component1 = component1[nbins // 2 : -nbins // 2 + 1]
            didx = imax - np.argmax(component1)
            component = np.roll(component1, didx)
            component = init_off_rms * (component / component.max())

            if component.size != profile.size:
                active_logger.error(
                    "CLEAN component shape (%d) is not the same as the profile (%d)!",
                    component.size,
                    profile.size,
                )
                raise ValueError

            cleaned = profile - component

            if np.argmax(component) != imax:
                active_logger.error(
                    "Component alignment error - convolved subtraction component "
                    "position (%d) does not match data maximum (%d)!",
                    np.argmax(component),
                    imax,
                )
                active_logger.error(
                    "Error on Iteration number = %d/%d for tau=%g ms",
                    niter,
                    self.iter_limit,
                    tau,
                )
                # Imported lazily to avoid a circular import with plotting (which depends on CleanResult).
                from . import plotting

                plotting.plot_cleaner_debug_component_alignment(
                    initial_data=profile_data.samples,
                    cleaned=cleaned,
                    clean_component=temp_clean_comp,
                    impulse_response=preconv,
                    subtracted_response=component,
                    raw_convolved_response=component1,
                    peak_index=imax,
                    threshold=self.threshold,
                    init_off_rms=init_off_rms,
                    iteration=niter,
                    iter_limit=self.iter_limit,
                )
                raise ValueError

            if component_history is not None:
                current_noise = profile_data.update_noise(cleaned)
                component_history.add(
                    SubtractedComponent(
                        iteration=niter,
                        peak_index=int(imax),
                        peak_value=float(dmax),
                        component_amplitude=float(cc_amp),
                        alignment_offset=int(didx),
                        clean_component=temp_clean_comp.copy(),
                        instrumental_response=inst_resp_func.copy(),
                        kernel_response=filter_guess.copy(),
                        impulse_response=preconv.copy(),
                        raw_convolved_response=component1.copy(),
                        subtracted_response=component.copy(),
                        residual_max=float(np.max(cleaned)),
                        residual_off_rms=float(current_noise.off_rms),
                    )
                )
            else:
                profile_data.update_noise(cleaned)

            on_pulse = profile_data.on_pulse(cleaned)
            off_pulse = profile_data.off_pulse(cleaned)
            loop = _keep_cleaning(
                on_pulse, off_pulse, threshold=self.threshold
            )
            profile = cleaned

        if niter <= 1:
            active_logger.warning(
                "Clean cycle only lasted 1 iteration - something probably went wrong!"
            )
        elif niter >= self.iter_limit:
            active_logger.warning(
                f"Clean cycle terminated prematurely for tau={tau:g} ms"
            )
        else:
            active_logger.debug(
                "Clean cycle terminated successfully for tau=%g ms", tau
            )

        n_unique = np.count_nonzero(clean_components)

        final_noise = profile_data.current_noise
        if final_noise is None:
            final_noise = profile_data.update_noise(profile)

        figures_of_merit = fom_evaluator.evaluate(
            profile=profile,
            clean_components=clean_components,
            pulsar_period=period,
            total_rms=final_noise.total_rms,
            initial_off_rms=init_off_rms,
            nbins=nbins,
            off_rms=final_noise.off_rms,
            off_mean=final_noise.off_mean,
            threshold=self.threshold,
        )
        recon = _reconstruct(clean_components, rest_func)

        return CleanResult(
            profile=profile,
            init_off_rms=init_off_rms,
            init_on_rms=init_on_rms,
            nbins=nbins,
            nbins_on=len(on_pulse_bins),
            nbins_off=len(off_pulse_bins),
            rest_func=rest_func,
            inst_resp_func=inst_resp_func,
            tau=tau,
            pbftype=pbftype,
            niter=niter,
            cc=clean_components,
            ncc=n_unique,
            off_bins=off_pulse_bins,
            on_bins=on_pulse_bins,
            off_rms=final_noise.off_rms,
            off_mean=final_noise.off_mean,
            on_rms=final_noise.on_rms,
            on_mean=final_noise.on_mean,
            total_mean=final_noise.total_mean,
            total_rms=final_noise.total_rms,
            recon=recon,
            threshold=self.threshold,
            profile_data=profile_data,
            figures_of_merit=figures_of_merit,
            component_history=component_history,
        )


def _keep_cleaning(
    on: np.ndarray, off: np.ndarray, threshold: float = 3.0
) -> bool:
    rms = np.std(off)
    mean = np.mean(off)
    datamax = np.max(on)
    limit = mean + threshold * rms
    return datamax > limit


def _reconstruct(ccs: np.ndarray, rest_func: np.ndarray) -> np.ndarray:
    recon = convolve(ccs, rest_func, mode="same")
    peak = np.max(np.abs(recon))
    if peak <= 0 or not np.isfinite(peak):
        return recon
    return recon / peak
