"""Shared signal and instrumental response helpers."""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import convolve

logger = logging.getLogger(__name__)


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
) -> tuple[np.ndarray, float]:
    """Compute the instrumental response function."""
    upscale_factor = 10

    if fast:
        decimated_resp = np.zeros_like(profile)
        decimated_resp[np.argmax(profile)] = 1
        resp_width = pulse_period / len(profile)
        return decimated_resp, resp_width

    elements = [
        response_width
        for response_width in [r_dm_width, r_pb_width, r_av_width, r_pd_width]
        if response_width > 0
    ]
    logger.debug("Restoring elements = %s ms", elements)
    narrowest_element = min(elements)
    oversamp_nbins = int(upscale_factor * (pulse_period / narrowest_element))
    oversamp_dt = pulse_period / oversamp_nbins
    logger.debug("Upsampled nbins=%s & dt=%sms", oversamp_nbins, oversamp_dt)

    response = np.zeros(oversamp_nbins)
    for element in elements:
        if np.isfinite(element):
            contribution = np.zeros(oversamp_nbins)
            width_bins = element // oversamp_dt
            slc = slice(
                int(oversamp_nbins / 2 - width_bins / 2),
                int(oversamp_nbins / 2 + width_bins / 2),
            )
            contribution[slc] = 1
            if response.sum() <= 0:
                response = response + contribution
            else:
                response = convolve(response, contribution, mode="same")

    response = response / response.sum()
    oversampled_x = np.linspace(0, pulse_period, oversamp_nbins, endpoint=False)
    original_x = np.linspace(0, pulse_period, profile.size, endpoint=False)
    interp_func = PchipInterpolator(oversampled_x, response, extrapolate=False)
    decimated_resp = interp_func(original_x)
    decimated_resp = np.nan_to_num(decimated_resp, nan=0, posinf=0, neginf=0)
    decimated_resp = decimated_resp / np.trapz(
        y=decimated_resp, dx=pulse_period / profile.size
    )
    resp_width = np.trapz(dx=oversamp_dt, y=response) / response.max()
    return decimated_resp, resp_width


def get_restoring_function(
    profile: np.ndarray,
    pulse_period: float,
    inst_resp_width: float,
) -> np.ndarray:
    """Generate the restoring function for pulse shape reconstruction."""
    upfact = 10
    nbins = upfact * len(profile)
    x = pulse_period * np.linspace(-0.5, 0.5, nbins)
    rest_func = gaussian(x, mu=0, sigma=inst_resp_width)
    return rest_func[::upfact]


def reconstruct(
    clean_components: np.ndarray,
    rest_func: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct the intrinsic pulse shape from clean components."""
    if rest_func is None:
        logger.warning("No valid restoring function provided, using a delta function")
        rest_func = np.zeros_like(clean_components)
        rest_func[rest_func.size // 2] = 1

    reconstruction = convolve(clean_components, rest_func, mode="same")
    return reconstruction / reconstruction.max()
