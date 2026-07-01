"""
Stokes-aware multiplication of beam and sky in the disc.

For Stokes I, beam-power times sky-temperature is a straightforward
elementwise product followed by a weighted sum -- there is no rotation
of the field components.

For Stokes Q/U, the sky's linear polarisation components live in the
celestial (RA, Dec) tangent basis, while the beam's response is defined
in the antenna (l, m) tangent basis. Going from one to the other
involves a rotation of the polarisation vector by twice the angle
between the two local meridians at each pixel (often called the
'parallactic angle' for the special case of the pointing pixel).

The Q/U path is **not yet implemented** in this first version; the
public entry point raises :class:`NotImplementedError` so that
downstream callers can integrate the interface today and the heavy
lifting can land in a follow-up that depends on TIBEC for the validated
rotation utilities.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

SUPPORTED_STOKES = ("I",)


def integrate_stokes_I(
    beam_disc: np.ndarray,
    sky_disc: np.ndarray,
    *,
    omega_b: np.ndarray,
    d_omega_pix: float,
) -> np.ndarray:
    """Stokes-I weighted-sum integration for a single time sample.

    Parameters
    ----------
    beam_disc : ndarray
        Beam power evaluated at the disc pixels, shape ``(n_freq, n_pix)``.
    sky_disc : ndarray
        Sky temperature at the same disc pixels, shape ``(n_freq, n_pix)``
        or ``(n_pix,)`` if frequency-independent.
    omega_b : ndarray
        Beam solid angle Omega_b(freq), shape ``(n_freq,)``. Must be
        strictly positive; a ``ValueError`` is raised otherwise to avoid
        silent ``inf``/``nan`` propagation into the TOD.
    d_omega_pix : float
        Solid angle of one HEALPix pixel, in steradians (4 pi / Npix).

    Returns
    -------
    sample : ndarray
        Antenna temperature contribution per frequency, shape ``(n_freq,)``.
    """
    if np.any(omega_b <= 0.0):
        raise ValueError(
            "omega_b must be strictly positive; received non-positive entries. "
            "This usually means the beam power cube is identically zero at one "
            "or more frequencies."
        )
    if sky_disc.ndim == 1:
        sky_disc = sky_disc[None, :]  # broadcast across frequencies
    weighted = beam_disc * sky_disc  # (n_freq, n_pix)
    integral = weighted.sum(axis=-1) * d_omega_pix
    return integral / omega_b


def integrate_stokes_full(
    beam_disc: Mapping[str, np.ndarray],
    sky_disc: Mapping[str, np.ndarray],
    *,
    parallactic_angle_rad: np.ndarray,
    omega_b: Mapping[str, np.ndarray],
    d_omega_pix: float,
) -> np.ndarray:
    """Full-Stokes integration (NOT YET IMPLEMENTED).

    The intended interface accepts beam and sky maps keyed by Stokes
    component ('I', 'Q', 'U', 'V') and a per-pixel ``parallactic_angle``
    that rotates sky Q/U into the beam-local Q/U frame. The
    implementation will lean on TIBEC's validated rotation utilities;
    see the Follow-ups section of the package README.
    """
    raise NotImplementedError(
        "Full-Stokes (Q/U/V) integration is on the follow-up list. " "Use stokes='I' for now."
    )
