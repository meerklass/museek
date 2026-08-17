"""
Coordinate projections used by the sky-to-beam integration path.

Three families of transforms are exposed:

1.  ``horizon_to_equatorial`` and ``equatorial_to_horizon``: spherical
    coordinate transforms between (Az, El) at a given Local Sidereal Time
    (LST) and (RA, Dec). Pure numpy formulas, vectorised over all inputs.

2.  ``direction_cosines``: maps a pair of (Az_pointing, El_pointing) and
    (Az_source, El_source) to the direction cosines (l, m) in the
    antenna-local tangent plane. This is the standard SIN projection used
    by the MeerKLASS holographic beam.

3.  ``pixel_directions_to_az_el``: convenience wrapper that takes a
    HEALPix nside plus an array of pixel ids and returns (Az, El) at the
    given LST.

All angles are in degrees on input and output. Internal computations use
radians.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
from numpy.typing import ArrayLike


def horizon_to_equatorial(
    az_deg: ArrayLike,
    el_deg: ArrayLike,
    lst_deg: ArrayLike,
    lat_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert horizontal (Az, El) to equatorial (RA, Dec).

    Parameters
    ----------
    az_deg, el_deg : ndarray or float
        Azimuth and elevation in degrees. Azimuth measured east of north.
    lst_deg : ndarray or float
        Local Sidereal Time in degrees, broadcastable against az/el.
    lat_deg : float
        Observer latitude in degrees (north positive).

    Returns
    -------
    ra_deg, dec_deg : ndarray
        Right ascension and declination in degrees, in [0, 360) and
        [-90, 90] respectively.
    """
    az = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(el_deg, dtype=np.float64))
    lst = np.deg2rad(np.asarray(lst_deg, dtype=np.float64))
    lat = np.deg2rad(lat_deg)

    sin_dec = np.sin(el) * np.sin(lat) + np.cos(el) * np.cos(lat) * np.cos(az)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))

    cos_dec = np.cos(dec)
    # Guard against the pole singularity. Outside the poles the divisor is
    # > 0 by construction.
    safe = cos_dec > 1e-12
    sin_ha = np.where(safe, -np.sin(az) * np.cos(el) / np.where(safe, cos_dec, 1.0), 0.0)
    cos_ha = np.where(
        safe,
        (np.sin(el) - np.sin(dec) * np.sin(lat)) / np.where(safe, cos_dec * np.cos(lat), 1.0),
        1.0,
    )
    ha = np.arctan2(sin_ha, cos_ha)

    ra = (lst - ha) % (2.0 * np.pi)
    return np.rad2deg(ra), np.rad2deg(dec)


def equatorial_to_horizon(
    ra_deg: ArrayLike,
    dec_deg: ArrayLike,
    lst_deg: ArrayLike,
    lat_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert equatorial (RA, Dec) to horizontal (Az, El).

    Inverse of :func:`horizon_to_equatorial`. See that function for the
    parameter conventions.
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    lst = np.deg2rad(np.asarray(lst_deg, dtype=np.float64))
    lat = np.deg2rad(lat_deg)

    ha = lst - ra

    sin_el = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    el = np.arcsin(np.clip(sin_el, -1.0, 1.0))

    cos_el = np.cos(el)
    safe = cos_el > 1e-12
    sin_az = np.where(safe, -np.sin(ha) * np.cos(dec) / np.where(safe, cos_el, 1.0), 0.0)
    cos_az = np.where(
        safe,
        (np.sin(dec) - np.sin(el) * np.sin(lat)) / np.where(safe, cos_el * np.cos(lat), 1.0),
        1.0,
    )
    az = np.arctan2(sin_az, cos_az) % (2.0 * np.pi)
    return np.rad2deg(az), np.rad2deg(el)


def direction_cosines(
    az_pointing_deg: ArrayLike,
    el_pointing_deg: ArrayLike,
    az_source_deg: ArrayLike,
    el_source_deg: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Direction cosines (l, m) of a source in the antenna-local tangent plane.

    Uses the SIN projection centred on the pointing direction. This is
    the convention used by the MeerKLASS holographic beam, where the
    on-disk grid is parameterised by (margin_deg) = (l * 180/pi, m * 180/pi).

    Parameters
    ----------
    az_pointing_deg, el_pointing_deg : ndarray
        Antenna pointing direction in degrees (broadcastable).
    az_source_deg, el_source_deg : ndarray
        Source direction(s) in degrees (broadcastable against pointing).

    Returns
    -------
    l, m : ndarray
        Direction cosines (dimensionless), with the same broadcast shape
        as the inputs.

    Notes
    -----
    Formulas:

        l = cos(el_s) * sin(az_s - az_p)
        m = sin(el_s) * cos(el_p) - cos(el_s) * sin(el_p) * cos(az_s - az_p)

    These match the convention in the existing ``primary_beam.py``.
    """
    az_p = np.deg2rad(np.asarray(az_pointing_deg, dtype=np.float64))
    el_p = np.deg2rad(np.asarray(el_pointing_deg, dtype=np.float64))
    az_s = np.deg2rad(np.asarray(az_source_deg, dtype=np.float64))
    el_s = np.deg2rad(np.asarray(el_source_deg, dtype=np.float64))

    daz = az_s - az_p
    l_cos = np.cos(el_s) * np.sin(daz)  # noqa: E741 (l is the standard radio-astronomy symbol)
    m_cos = np.sin(el_s) * np.cos(el_p) - np.cos(el_s) * np.sin(el_p) * np.cos(daz)
    return l_cos, m_cos


def pixel_directions_to_az_el(
    nside: int,
    pix_ids: np.ndarray,
    lst_deg: float,
    lat_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Look up HEALPix pixel centres and convert to horizontal coordinates.

    Assumes the HEALPix map is in equatorial coordinates with RA = phi and
    Dec = 90 - theta (the convention used by limTOD / standard sky models
    such as GDSM).

    Parameters
    ----------
    nside : int
        HEALPix resolution.
    pix_ids : ndarray of int
        Pixel indices.
    lst_deg : float
        Local Sidereal Time at the time of observation.
    lat_deg : float
        Observer latitude.

    Returns
    -------
    az_deg, el_deg : ndarray
        Azimuth and elevation of each pixel centre, in degrees.
    """
    theta, phi = hp.pix2ang(nside, np.asarray(pix_ids, dtype=np.int64))
    ra_deg = np.rad2deg(phi)
    dec_deg = 90.0 - np.rad2deg(theta)
    return equatorial_to_horizon(ra_deg, dec_deg, lst_deg, lat_deg)


def pointing_radec(
    az_deg: float,
    el_deg: float,
    lst_deg: float,
    lat_deg: float,
) -> tuple[float, float]:
    """Convenience scalar wrapper around :func:`horizon_to_equatorial`.

    Returns the (RA, Dec) of a single pointing direction in degrees.
    """
    ra, dec = horizon_to_equatorial(az_deg, el_deg, lst_deg, lat_deg)
    return float(ra), float(dec)
