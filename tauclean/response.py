"""Shared signal and instrumental response helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import convolve

logger = logging.getLogger(__name__)

# Order matches the r_dm_width/r_pb_width/r_av_width/r_pd_width arguments of
# get_instrumental_response().
_RESPONSE_ELEMENT_LABELS = (
    "DM smearing",
    "Profile bin width",
    "Backend sampling",
    "Post-detection filtering",
)


@dataclass(frozen=True)
class ResponseComponent:
    """A single named contribution to the instrumental response function."""

    label: str
    width: float
    response: np.ndarray


def _equivalent_width_to_sigma(width: float) -> float:
    """Convert area/peak equivalent width to Gaussian sigma.

    For a unit-area Gaussian, area/peak = sqrt(2*pi)*sigma.
    """
    return width / np.sqrt(2 * np.pi)


def gaussian(x, mu: float = 0, sigma: float = 1) -> np.ndarray:
    """Calculate a Gaussian shape over x."""
    amp = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def dm_delay(dm: float, lo: float, hi: float) -> float:
    """Calculate the dispersion delay between frequencies in GHz."""
    k = 4.148808
    return k * dm * (lo ** (-2) - hi ** (-2))


def get_instrumental_response(
    profile: np.ndarray,
    pulse_period: float,
    r_dm_width: float,
    r_pb_width: float,
    r_av_width: float,
    r_pd_width: float,
    fast: bool = False,
    return_components: bool = False,
) -> (
    tuple[np.ndarray, float]
    | tuple[np.ndarray, float, list[ResponseComponent]]
):
    """Compute the instrumental response function.

    If ``return_components`` is True, a third value is returned: a list of
    :class:`ResponseComponent` objects, one per named contribution with a
    non-zero width (e.g. DM smearing, profile bin width), decimated onto the
    same time grid as the returned response. This is intended for
    visualising how each element shapes the total response - see
    :func:`tauclean.plotting.plot_instrumental_response`.
    """
    upscale_factor = 10

    if fast:
        decimated_resp = np.zeros_like(profile)
        decimated_resp[np.argmax(profile)] = 1
        resp_width = pulse_period / len(profile)
        if return_components:
            return decimated_resp, resp_width, []
        return decimated_resp, resp_width

    raw_elements = np.array(
        [r_dm_width, r_pb_width, r_av_width, r_pd_width], dtype=float
    )
    if not np.all(np.isfinite(raw_elements)):
        raise ValueError("All instrumental response widths must be finite.")

    named_elements = [
        (label, width)
        for label, width in zip(_RESPONSE_ELEMENT_LABELS, raw_elements)
        if width > 0
    ]
    if not named_elements:
        raise ValueError(
            "At least one instrumental response width must be > 0."
        )

    elements = [width for _, width in named_elements]
    logger.debug("Restoring elements = %s ms", elements)
    narrowest_element = min(elements)
    oversamp_nbins = int(upscale_factor * (pulse_period / narrowest_element))
    oversamp_dt = pulse_period / oversamp_nbins
    logger.debug("Upsampled nbins=%s & dt=%sms", oversamp_nbins, oversamp_dt)

    response = np.zeros(oversamp_nbins)
    raw_contributions = []
    for label, element in named_elements:
        contribution = np.zeros(oversamp_nbins)
        width_bins = max(1, int(np.round(element / oversamp_dt)))
        center_bin = oversamp_nbins // 2
        start_bin = center_bin - width_bins // 2
        stop_bin = min(oversamp_nbins, start_bin + width_bins)
        start_bin = max(0, stop_bin - width_bins)
        contribution[start_bin:stop_bin] = 1
        raw_contributions.append((label, element, contribution))
        if response.sum() <= 0:
            response = response + contribution
        else:
            response = convolve(response, contribution, mode="same")

    response = response / response.sum()
    oversampled_x = np.linspace(
        0, pulse_period, oversamp_nbins, endpoint=False
    )
    original_x = np.linspace(0, pulse_period, profile.size, endpoint=False)
    interp_func = PchipInterpolator(oversampled_x, response, extrapolate=False)
    decimated_resp = interp_func(original_x)
    decimated_resp = np.nan_to_num(decimated_resp, nan=0, posinf=0, neginf=0)
    decimated_resp = decimated_resp / np.trapz(
        y=decimated_resp, dx=pulse_period / profile.size
    )
    resp_width = np.trapz(dx=oversamp_dt, y=response) / response.max()

    if return_components:
        dt = pulse_period / profile.size
        components = []
        for label, element, contribution in raw_contributions:
            if contribution.sum() > 0:
                contribution = contribution / contribution.sum()
            comp_interp = PchipInterpolator(
                oversampled_x, contribution, extrapolate=False
            )
            decimated_comp = comp_interp(original_x)
            decimated_comp = np.nan_to_num(
                decimated_comp, nan=0, posinf=0, neginf=0
            )
            area = np.trapz(y=decimated_comp, dx=dt)
            if area > 0:
                decimated_comp = decimated_comp / area
            components.append(
                ResponseComponent(
                    label=label, width=element, response=decimated_comp
                )
            )
        return decimated_resp, resp_width, components

    return decimated_resp, resp_width


def get_restoring_function(
    profile: np.ndarray,
    pulse_period: float,
    inst_resp_width: float,
) -> np.ndarray:
    """Generate the restoring function for pulse shape reconstruction.

    The provided ``inst_resp_width`` is interpreted as an equivalent width
    defined as area/peak of the instrumental response. This is converted to
    Gaussian sigma via sigma = width / sqrt(2*pi).
    """
    if not np.isfinite(inst_resp_width) or inst_resp_width <= 0:
        raise ValueError("Instrumental response width must be finite and > 0.")

    upfact = 10
    nbins = upfact * len(profile)
    x = pulse_period * np.linspace(-0.5, 0.5, nbins, endpoint=False)
    sigma = _equivalent_width_to_sigma(inst_resp_width)
    rest_func = gaussian(x, mu=0, sigma=sigma)
    decimated = rest_func[::upfact]
    dt = pulse_period / len(profile)
    area = np.trapz(y=decimated, dx=dt)
    if area <= 0 or not np.isfinite(area):
        raise ValueError("Restoring function has invalid normalization area.")
    return decimated / area


def reconstruct(
    clean_components: np.ndarray,
    rest_func: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct the intrinsic pulse shape from clean components."""
    if rest_func is None:
        logger.warning(
            "No valid restoring function provided, using a delta function"
        )
        rest_func = np.zeros_like(clean_components)
        rest_func[rest_func.size // 2] = 1

    reconstruction = convolve(clean_components, rest_func, mode="same")
    peak = np.max(np.abs(reconstruction))
    if peak <= 0 or not np.isfinite(peak):
        return reconstruction
    return reconstruction / peak
