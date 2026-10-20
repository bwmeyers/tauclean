"""Shared signal and instrumental response helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from .circular import center_to_origin, circ_convolve

logger = logging.getLogger(__name__)

# Order matches the r_dm_width/r_pb_width/r_av_width/r_pd_width arguments of
# get_instrumental_response().
_RESPONSE_ELEMENT_LABELS = (
    "DM smearing",
    "Profile bin width",
    "Backend sampling",
    "Post-detection filtering",
)

# Above this fraction of the pulse period, a boxcar element is wide enough
# that its contribution is likely unphysical (a pulsar profile is periodic,
# so smearing comparable to a full period doesn't have a clean interpretation).
_WIDTH_PERIOD_WARN_FRACTION = 0.5


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

    The profile is cyclic, so boxcars are built with wrapped indices and
    combined by circular convolution. The returned arrays are centred on bin
    ``len(profile) // 2``; use :func:`tauclean.circular.center_to_origin` to
    move the zero lag to index 0 before circularly convolving with a profile.
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
    for label, element in named_elements:
        if element >= pulse_period:
            logger.warning(
                "%s width (%g ms) is >= the pulse period (%g ms); its "
                "contribution will be truncated to span the full period, "
                "which is likely unphysical.",
                label,
                element,
                pulse_period,
            )
        elif element >= _WIDTH_PERIOD_WARN_FRACTION * pulse_period:
            logger.warning(
                "%s width (%g ms) is comparable to the pulse period (%g ms); "
                "the resulting response may be unreliable.",
                label,
                element,
                pulse_period,
            )
    narrowest_element = min(elements)
    oversamp_nbins = int(upscale_factor * (pulse_period / narrowest_element))
    oversamp_dt = pulse_period / oversamp_nbins
    logger.debug("Upsampled nbins=%s & dt=%sms", oversamp_nbins, oversamp_dt)

    raw_contributions = []
    response = None
    for label, element in named_elements:
        width_bins = min(
            oversamp_nbins, max(1, int(np.round(element / oversamp_dt)))
        )
        contribution = np.zeros(oversamp_nbins)
        first = -(width_bins // 2)
        contribution[np.arange(first, first + width_bins) % oversamp_nbins] = 1
        raw_contributions.append((label, element, contribution))
        if response is None:
            response = contribution
        else:
            response = circ_convolve(response, contribution)

    response = response / response.sum()
    oversampled_x = np.linspace(0, pulse_period, oversamp_nbins, endpoint=False)
    original_x = np.linspace(0, pulse_period, profile.size, endpoint=False)

    def _decimate(origin_centred: np.ndarray) -> np.ndarray:
        # Close the cycle so interpolation near the end wraps to bin 0.
        xs = np.append(oversampled_x, pulse_period)
        ys = np.append(origin_centred, origin_centred[0])
        out = PchipInterpolator(xs, ys)(original_x)
        out = np.nan_to_num(out, nan=0, posinf=0, neginf=0)
        return np.roll(out, profile.size // 2)

    decimated_resp = _decimate(response)
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
            decimated_comp = _decimate(contribution)
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
    Gaussian sigma via sigma = width / sqrt(2*pi). The width is floored at
    the profile's native bin width, since a restoring function narrower
    than one bin cannot meaningfully smooth the per-bin CLEAN components
    and produces a jagged reconstruction.
    """
    if not np.isfinite(inst_resp_width) or inst_resp_width <= 0:
        raise ValueError("Instrumental response width must be finite and > 0.")

    bin_width = pulse_period / len(profile)
    if inst_resp_width < bin_width:
        logger.debug(
            "Restoring function width (%g ms) is narrower than the profile "
            "bin width (%g ms); flooring to the bin width.",
            inst_resp_width,
            bin_width,
        )
        inst_resp_width = bin_width

    upfact = 10
    nbins = upfact * len(profile)
    x = pulse_period * np.linspace(-0.5, 0.5, nbins, endpoint=False)
    sigma = _equivalent_width_to_sigma(inst_resp_width)
    # Sum neighbouring periods so the Gaussian tails wrap around the cycle.
    rest_func = sum(
        gaussian(x, mu=k * pulse_period, sigma=sigma) for k in (-1, 0, 1)
    )
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
    """Reconstruct the intrinsic pulse shape from clean components.

    ``rest_func`` must have the same length as ``clean_components`` and be
    centred on bin ``size // 2``. Convolution is circular, so components near
    one edge of the profile contribute to the other.
    """
    if rest_func is None:
        logger.warning(
            "No valid restoring function provided, using a delta function"
        )
        rest_func = np.zeros_like(clean_components)
        rest_func[rest_func.size // 2] = 1

    if rest_func.shape != clean_components.shape:
        raise ValueError(
            "rest_func and clean_components must have the same length, got "
            f"{rest_func.size} and {clean_components.size}"
        )
    reconstruction = circ_convolve(clean_components, center_to_origin(rest_func))
    peak = np.max(np.abs(reconstruction))
    if peak <= 0 or not np.isfinite(peak):
        return reconstruction
    return reconstruction / peak
