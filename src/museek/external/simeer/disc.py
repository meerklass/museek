"""
HEALPix disc selection helpers.

We only ever multiply the beam by sky pixels that fall inside the beam's
angular support (~ +/-6 degrees in (l, m) for MeerKLASS U-band). This
module exposes a thin wrapper around :func:`healpy.query_disc` that
caches results keyed by ``(nside, pointing_pixel)``: in raster scans many
consecutive time samples point at directions that resolve to the same
HEALPix pixel, so the disc selection can be reused without recomputation.

The cache is intentionally simple (LRU on a process-local dict) so that
multiprocessing workers each maintain their own.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import healpy as hp
import numpy as np


@lru_cache(maxsize=4096)
def _query_disc_cached(
    nside: int, center_pix: int, radius_rad: float, inclusive: bool
) -> Tuple[int, ...]:
    """Cache disc queries keyed by the HEALPix pixel of the pointing.

    Returning a tuple keeps the cache values hashable-friendly and lets
    callers convert back to an ndarray cheaply.
    """
    theta, phi = hp.pix2ang(nside, center_pix)
    vec = hp.ang2vec(theta, phi)
    pix = hp.query_disc(nside, vec, radius_rad, inclusive=inclusive)
    return tuple(int(p) for p in pix)


def select_disc(
    nside: int,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    *,
    inclusive: bool = True,
) -> np.ndarray:
    """Return pixel indices within ``radius_deg`` of the given (RA, Dec).

    Results are cached on the HEALPix pixel containing the pointing.
    Sub-pixel jitter therefore does not invalidate the cache.

    Parameters
    ----------
    nside : int
        HEALPix resolution.
    ra_deg, dec_deg : float
        Pointing direction in degrees (equatorial).
    radius_deg : float
        Disc radius in degrees. Should be at least the maximum extent of
        the beam grid (so we never clip beam support); a small safety
        margin of ~25% is wise.
    inclusive : bool, default True
        Whether to include pixels whose centres are slightly outside the
        radius but whose body intersects the disc.

    Returns
    -------
    pix_ids : ndarray of int
        HEALPix pixel indices.
    """
    theta = np.deg2rad(90.0 - dec_deg)
    phi = np.deg2rad(ra_deg % 360.0)
    center_pix = int(hp.ang2pix(nside, theta, phi))
    pix_tuple = _query_disc_cached(nside, center_pix, float(np.deg2rad(radius_deg)), inclusive)
    return np.asarray(pix_tuple, dtype=np.int64)


def clear_disc_cache() -> None:
    """Clear the disc-selection LRU cache.

    Useful in tests, or when switching between sky models with different
    ``nside``.
    """
    _query_disc_cached.cache_clear()
