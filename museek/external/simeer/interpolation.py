"""
Precomputed bilinear interpolation of the (m, l) beam grid.

The MeerKLASS beam is stored on a regular Cartesian grid in direction
cosine space, parameterised by ``margin_deg`` (same grid for both axes).
For each pointing sample we want to interpolate the beam at a moderate
number (~10^3-10^4) of sky-pixel positions ``(l_query, m_query)``, for
every frequency.

Because the (m, l) grid does not depend on frequency, the index/weight
computation can be done **once per pointing**, and reused across every
frequency channel in the beam cube. This module exposes that two-step
API:

1.  :func:`precompute_bilinear_weights` -> :class:`BilinearWeights` holding
    integer corner indices and the four bilinear weights.

2.  :func:`apply_bilinear` applies the precomputed weights to a beam
    cube of shape ``(n_freq, n_m, n_l)`` and returns ``(n_freq, n_query)``.

Out-of-range query points are flagged via a ``valid`` mask on the
weights object; the corresponding columns in the output are set to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BilinearWeights:
    """Precomputed bilinear interpolation weights for a fixed (l, m) grid.

    Attributes
    ----------
    i_l, i_m : ndarray of int
        Lower-corner indices into the l and m grids. Shape ``(n_query,)``.
        Clipped to be in ``[0, n_l - 2]`` and ``[0, n_m - 2]``.
    w00, w01, w10, w11 : ndarray
        Bilinear weights for the four corners (m, l):
        ``(i_m, i_l)``, ``(i_m, i_l+1)``, ``(i_m+1, i_l)``, ``(i_m+1, i_l+1)``.
        Each of shape ``(n_query,)``. Sum to 1 on valid samples, 0 outside.
    valid : ndarray of bool
        ``True`` where the query point lies inside the grid extent.
    """

    i_l: np.ndarray
    i_m: np.ndarray
    w00: np.ndarray
    w01: np.ndarray
    w10: np.ndarray
    w11: np.ndarray
    valid: np.ndarray


def precompute_bilinear_weights(
    l_query: np.ndarray,
    m_query: np.ndarray,
    l_grid: np.ndarray,
    m_grid: np.ndarray,
) -> BilinearWeights:
    """Compute bilinear interpolation indices and weights.

    Parameters
    ----------
    l_query, m_query : ndarray
        Query coordinates in the same units as ``l_grid`` and ``m_grid``
        (typically degrees, i.e. ``direction_cosine * 180/pi``).
        Shape ``(n_query,)``.
    l_grid, m_grid : ndarray
        Monotonically increasing 1D grids of the beam. Need not be equally
        spaced, although the MeerKLASS grid is.

    Returns
    -------
    BilinearWeights
        Precomputed weights, ready to be fed to :func:`apply_bilinear`.

    Notes
    -----
    Query points outside the grid extent are clamped and flagged via the
    ``valid`` mask. The caller can rely on :func:`apply_bilinear` to zero
    them out, so no explicit masking is needed downstream.
    """
    l_query = np.asarray(l_query, dtype=np.float64)
    m_query = np.asarray(m_query, dtype=np.float64)
    l_grid = np.asarray(l_grid, dtype=np.float64)
    m_grid = np.asarray(m_grid, dtype=np.float64)

    if l_query.shape != m_query.shape:
        raise ValueError(
            f"l_query and m_query must have the same shape, "
            f"got {l_query.shape} vs {m_query.shape}"
        )

    valid = (
        (l_query >= l_grid[0])
        & (l_query <= l_grid[-1])
        & (m_query >= m_grid[0])
        & (m_query <= m_grid[-1])
    )

    # searchsorted returns insertion indices in [0, N]; subtract 1 and clip
    # to land on the lower corner of the bracketing cell.
    i_l = np.clip(np.searchsorted(l_grid, l_query, side="right") - 1, 0, len(l_grid) - 2)
    i_m = np.clip(np.searchsorted(m_grid, m_query, side="right") - 1, 0, len(m_grid) - 2)

    l0 = l_grid[i_l]
    l1 = l_grid[i_l + 1]
    m0 = m_grid[i_m]
    m1 = m_grid[i_m + 1]

    tl = (l_query - l0) / (l1 - l0)
    tm = (m_query - m0) / (m1 - m0)

    # Bilinear weights; (m, l) ordering matches the beam array axes
    # (axis -2 is m, axis -1 is l).
    w00 = (1.0 - tm) * (1.0 - tl)
    w01 = (1.0 - tm) * tl
    w10 = tm * (1.0 - tl)
    w11 = tm * tl

    # Zero out weights for invalid points so apply_bilinear returns 0.
    # In-place masking avoids 5 throw-away allocations from np.where.
    invalid = ~valid
    if invalid.any():
        w00[invalid] = 0.0
        w01[invalid] = 0.0
        w10[invalid] = 0.0
        w11[invalid] = 0.0

    return BilinearWeights(
        i_l=i_l.astype(np.int64),
        i_m=i_m.astype(np.int64),
        w00=w00,
        w01=w01,
        w10=w10,
        w11=w11,
        valid=valid,
    )


def apply_bilinear(
    weights: BilinearWeights,
    cube: np.ndarray,
    freq_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Apply precomputed bilinear weights to a beam cube.

    Parameters
    ----------
    weights : BilinearWeights
        Output of :func:`precompute_bilinear_weights`.
    cube : ndarray
        Beam power cube of shape ``(n_freq, n_m, n_l)``. Must be
        compatible with the grid that was used to compute ``weights``.
        Can be a memory-mapped array.
    freq_indices : ndarray of int, optional
        If given, select these frequency channels from ``cube`` before
        applying the interpolation. Otherwise all channels are used.

    Returns
    -------
    out : ndarray
        Interpolated beam values, shape ``(n_freq_out, n_query)``.
        Out-of-range query points are zero.
    """
    if freq_indices is not None:
        cube = cube[np.asarray(freq_indices, dtype=np.int64), :, :]

    if cube.ndim != 3:
        raise ValueError(f"cube must have shape (n_freq, n_m, n_l); got ndim={cube.ndim}")

    i_l = weights.i_l
    i_m = weights.i_m

    # Fancy index to gather the 4 corner slabs over all frequencies in
    # one shot. Each f_ij has shape (n_freq, n_query).
    f00 = cube[:, i_m, i_l]
    f01 = cube[:, i_m, i_l + 1]
    f10 = cube[:, i_m + 1, i_l]
    f11 = cube[:, i_m + 1, i_l + 1]

    # Broadcasting: weights are (n_query,), values are (n_freq, n_query).
    return weights.w00 * f00 + weights.w01 * f01 + weights.w10 * f10 + weights.w11 * f11
