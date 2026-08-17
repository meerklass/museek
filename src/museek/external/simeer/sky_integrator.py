"""
Per-sample sky-TOD integration on a disc of HEALPix sky pixels.

This is the core of Simeer. For one (LST, az_pointing, el_pointing)
sample it:

1.  rotates the pointing into the equatorial frame to identify the
    relevant HEALPix sky pixels (:mod:`simeer.disc`);
2.  rotates those pixels back into the horizontal frame at this LST
    (:mod:`simeer.projection`);
3.  computes their direction cosines (l, m) in the beam frame;
4.  precomputes bilinear interpolation weights against the beam grid
    (:mod:`simeer.interpolation`);
5.  applies the weights to the beam power cube to get B(l, m, freq);
6.  multiplies by the sky temperature, sums over the disc, normalises
    by the beam solid angle Omega_b(freq) (:mod:`simeer.stokes`).

The module exposes both a single-sample function
(:func:`integrate_sample`) and a vectorised driver over a full time
list (:func:`integrate_tod`) that uses :mod:`simeer._parallel`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Callable

import healpy as hp
import numpy as np

from . import _parallel, disc, interpolation, projection, stokes
from .beam import MeerKLASSBeam

# Stokes modes currently supported (Q/U/V tracked in follow-up #1).
SUPPORTED_STOKES = ("I",)


# ---------------------------------------------------------------------- #
# Single-sample integration                                              #
# ---------------------------------------------------------------------- #
def integrate_sample(
    *,
    lst_deg: float,
    az_pointing_deg: float,
    el_pointing_deg: float,
    lat_deg: float,
    beam: MeerKLASSBeam,
    sky_maps: np.ndarray,
    beam_freq_indices: np.ndarray,
    sky_freq_indices: np.ndarray | None = None,
    disc_radius_deg: float,
    polarization: str = "HH",
    stokes_modes: tuple[str, ...] = ("I",),
    horizontal_mask: np.ndarray | None = None,
    _beam_power_cube: np.ndarray | None = None,
    _omega_b_disc: np.ndarray | None = None,
    _margin_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Compute one **Sky TOD** sample (Stokes I) for all requested frequencies.

    The returned value is the beam-weighted sky temperature at this
    single pointing -- no gain, no noise. See :func:`integrate_tod` for
    the multi-pointing driver, and :meth:`simeer.SimeerTODSim.generate_TOD`
    for the Full TOD that injects gain and noise on top.

    Parameters
    ----------
    lst_deg : float
        Local Sidereal Time in degrees.
    az_pointing_deg, el_pointing_deg : float
        Antenna pointing direction in horizontal coordinates (degrees).
    lat_deg : float
        Observer latitude (degrees).
    beam : MeerKLASSBeam
        Pre-loaded beam wrapper.
    sky_maps : ndarray
        Sky brightness temperature, shape ``(n_freq_sky, n_pix_sky)``.
        ``n_pix_sky`` must correspond to a valid HEALPix ``nside``. The
        sky is assumed to be in equatorial coordinates (same frame as
        ``horizontal_mask`` if supplied).
    beam_freq_indices : ndarray of int
        Indices into ``beam.freq_MHz`` for each output channel. Shape
        ``(n_freq_out,)``.
    sky_freq_indices : ndarray of int, optional
        Indices into ``sky_maps``' frequency axis. Default ``None`` means
        ``np.arange(n_freq_out)`` — i.e. the sky cube is assumed to
        already be aligned with the requested output channels, which is
        the typical case after :func:`materialize_sky_cube`. Pass an
        explicit array only when your sky cube has a different frequency
        axis than the beam's.
    disc_radius_deg : float
        Radius of the sky disc to integrate over. Should comfortably
        exceed the beam grid extent (e.g. 8 deg for a +/-6 deg beam).
    polarization : str, default ``'HH'``
        Beam polarisation. Only ``'HH'`` and ``'VV'`` are supported.
    stokes_modes : tuple of str, default ``('I',)``
        Stokes components to compute. Only ``('I',)`` is implemented;
        ``('I', 'Q', 'U')`` is reserved for follow-up #1.
    horizontal_mask : ndarray of bool, optional
        HEALPix mask in **equatorial** coordinates (same frame as
        ``sky_maps``), ``True`` for unmasked pixels. Must have the same
        length as ``sky_maps`` along the pixel axis; an explicit
        ``ValueError`` is raised on mismatch. Despite the name, this is
        applied in the sky frame: the caller is responsible for
        converting any horizontal-frame mask to equatorial. A true
        horizontal-frame mask is tracked in follow-up #4.

    Returns
    -------
    sample : ndarray
        Antenna temperature contribution, shape ``(n_freq_out,)``.

    Notes
    -----
    The two underscore-prefixed parameters (``_beam_power_cube``,
    ``_omega_b_disc``) are an internal optimisation: when called by the
    batched :func:`integrate_tod`, the beam power cube and the
    omega_b-at-output-channels slice are computed once per batch and
    threaded through here to skip redundant attribute lookups. They
    default to ``None`` and are recomputed from ``beam`` on demand,
    which is the right behaviour for one-off calls.
    """
    if stokes_modes != ("I",):
        raise NotImplementedError(
            f"Only Stokes I is supported in v0.1; got stokes_modes={stokes_modes!r}. "
            "Track follow-up #1 in the README for the Q/U/V roadmap."
        )

    nside_sky = hp.npix2nside(sky_maps.shape[-1])
    n_freq_out = len(beam_freq_indices)

    if sky_freq_indices is None:
        sky_freq_indices = np.arange(n_freq_out, dtype=np.int64)

    # Validate horizontal_mask shape eagerly -- silent fallback was a
    # foot-gun (silently disables masking; bug found in review).
    if horizontal_mask is not None:
        expected_npix = hp.nside2npix(nside_sky)
        if horizontal_mask.shape[-1] != expected_npix:
            raise ValueError(
                f"horizontal_mask has {horizontal_mask.shape[-1]} pixels; "
                f"expected {expected_npix} for nside={nside_sky}."
            )

    # 1) pointing -> RA, Dec
    ra_p, dec_p = projection.pointing_radec(az_pointing_deg, el_pointing_deg, lst_deg, lat_deg)

    # 2) disc query in equatorial coords
    pix_ids = disc.select_disc(nside_sky, ra_p, dec_p, disc_radius_deg)
    if pix_ids.size == 0:
        return np.zeros(n_freq_out, dtype=np.float64)

    # 3) disc pixels back to horizontal at this LST
    az_s, el_s = projection.pixel_directions_to_az_el(nside_sky, pix_ids, lst_deg, lat_deg)

    keep = np.ones_like(pix_ids, dtype=bool)
    if horizontal_mask is not None:
        keep &= horizontal_mask[pix_ids]
    # Sky pixels below the horizon don't see the antenna; drop them.
    keep &= el_s > 0.0
    if not np.any(keep):
        return np.zeros(n_freq_out, dtype=np.float64)

    pix_ids = pix_ids[keep]
    az_s = az_s[keep]
    el_s = el_s[keep]

    # 4) direction cosines in beam frame
    l_dc, m_dc = projection.direction_cosines(az_pointing_deg, el_pointing_deg, az_s, el_s)
    l_deg = np.rad2deg(l_dc)
    m_deg = np.rad2deg(m_dc)

    # 5) bilinear weights against the beam (l, m) grid
    margin_deg = _margin_deg if _margin_deg is not None else beam.margin_deg
    weights = interpolation.precompute_bilinear_weights(l_deg, m_deg, margin_deg, margin_deg)

    # 6) apply weights to the beam cube at the requested frequencies
    cube = _beam_power_cube if _beam_power_cube is not None else beam.power_cube(polarization)
    beam_disc = interpolation.apply_bilinear(weights, cube, beam_freq_indices)  # (nf, npix)

    # 7) integrate against the sky
    omega_b = (
        _omega_b_disc
        if _omega_b_disc is not None
        else beam.beam_solid_angle(polarization)[beam_freq_indices]
    )
    d_omega_pix = 4.0 * np.pi / hp.nside2npix(nside_sky)
    sky_disc = np.ascontiguousarray(sky_maps[sky_freq_indices[:, None], pix_ids[None, :]])

    return stokes.integrate_stokes_I(
        beam_disc=beam_disc.astype(np.float64),
        sky_disc=sky_disc,
        omega_b=omega_b,
        d_omega_pix=d_omega_pix,
    )


# ---------------------------------------------------------------------- #
# Batched worker (used by joblib)                                        #
# ---------------------------------------------------------------------- #
def _process_batch(
    indices: np.ndarray,
    lst_arr: np.ndarray,
    az_arr: np.ndarray,
    el_arr: np.ndarray,
    lat_deg: float,
    beam_power_cube: np.ndarray,
    margin_deg: np.ndarray,
    omega_b_disc: np.ndarray,
    sky_maps: np.ndarray,
    sky_freq_idx: np.ndarray,
    beam_freq_idx: np.ndarray,
    disc_radius_deg: float,
    polarization: str,
    horizontal_mask: np.ndarray | None,
) -> np.ndarray:
    """Process a contiguous batch of sample indices serially.

    Used as the worker function for joblib parallelism: dispatching
    *batches* instead of individual samples amortises the per-call IPC
    overhead which, at ~0.3 ms per sample, is otherwise comparable to
    the work itself.

    Note: ``beam_power_cube``, ``margin_deg`` and ``omega_b_disc`` are
    passed as bare ndarrays so that joblib's auto-memmap can kick in
    when ``n_jobs > 1``. joblib's memmap threshold only fires for
    top-level ndarray arguments, not arrays nested inside Python
    objects -- this is why we deliberately do NOT pass the
    :class:`MeerKLASSBeam` wrapper here.
    """
    out = np.empty((len(beam_freq_idx), len(indices)), dtype=np.float64)
    for k, i in enumerate(indices):
        out[:, k] = integrate_sample(
            lst_deg=float(lst_arr[i]),
            az_pointing_deg=float(az_arr[i]),
            el_pointing_deg=float(el_arr[i]),
            lat_deg=lat_deg,
            beam=None,  # type: ignore[arg-type]  # all bare-array overrides are set below
            sky_maps=sky_maps,
            beam_freq_indices=beam_freq_idx,
            sky_freq_indices=sky_freq_idx,
            disc_radius_deg=disc_radius_deg,
            polarization=polarization,
            horizontal_mask=horizontal_mask,
            _beam_power_cube=beam_power_cube,
            _omega_b_disc=omega_b_disc,
            _margin_deg=margin_deg,
        )
    return out


# ---------------------------------------------------------------------- #
# Driver over a full pointing list                                       #
# ---------------------------------------------------------------------- #
def integrate_tod(
    *,
    lst_deg_list: np.ndarray,
    az_deg_list: np.ndarray,
    el_deg_list: np.ndarray,
    lat_deg: float,
    beam: MeerKLASSBeam,
    sky_maps: np.ndarray,
    freq_MHz: Sequence[float],
    disc_radius_deg: float,
    polarization: str = "HH",
    horizontal_mask: np.ndarray | None = None,
    n_jobs: int = 1,
    batch_size: int | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Generate the **Sky TOD** for a full list of pointings.

    This is the noiseless, gain-free beam-weighted sky signal -- it does
    NOT include receiver gain, 1/f gain fluctuations, system-temperature
    offsets, or white noise. For a Full TOD with those instrumental
    effects, use :class:`simeer.SimeerTODSim.generate_TOD` (which calls
    this function internally as its sky-TOD step).

    Implemented as a vectorised driver over :func:`integrate_sample`,
    parallelised across time samples via :mod:`simeer._parallel`.

    Parameters
    ----------
    lst_deg_list, az_deg_list, el_deg_list : ndarray
        Per-sample LST, azimuth, and elevation in degrees. All must have
        the same length ``ntime``.
    sky_maps : ndarray
        ``(n_freq_sky, n_pix_sky)`` in equatorial coordinates.
        ``n_freq_sky`` must equal ``len(freq_MHz)``; the sky cube is
        assumed to be aligned with the output frequency grid.
    freq_MHz : sequence of float
        Output observing frequencies (MHz). Must be representable on the
        beam grid; an error is raised otherwise.
    n_jobs : int, default 1
        Passed to :func:`simeer._parallel.map_samples`. ``1`` runs
        serially; ``-1`` uses every available core via joblib's Loky
        process pool.
    batch_size : int, optional
        Number of samples processed per joblib task. Default ``None``
        auto-selects ``max(64, ntime // (4 * n_workers))`` so that each
        worker sees enough work to dominate the ~0.5-1 ms per-call IPC
        cost. Set to 1 to recover the old "one sample per task"
        behaviour (useful for debugging only).
    progress : bool, default False
        Show a ``tqdm`` progress bar over the dispatched batches.

    Returns
    -------
    tod : ndarray
        Sky-TOD of shape ``(n_freq_out, ntime)`` in the same temperature
        units as ``sky_maps``.
    """
    lst_arr = np.asarray(lst_deg_list, dtype=np.float64)
    az_arr = np.asarray(az_deg_list, dtype=np.float64)
    el_arr = np.asarray(el_deg_list, dtype=np.float64)
    ntime = lst_arr.size
    if not (az_arr.size == ntime and el_arr.size == ntime):
        raise ValueError("lst_deg_list, az_deg_list, el_deg_list must have equal length.")

    freq_arr = np.asarray(freq_MHz, dtype=np.float64)
    beam_freq_idx = beam.freq_indices(freq_arr)
    sky_freq_idx = np.arange(len(freq_arr), dtype=np.int64)
    if sky_maps.shape[0] != len(freq_arr):
        raise ValueError(
            f"sky_maps frequency axis ({sky_maps.shape[0]}) does not match "
            f"len(freq_MHz) ({len(freq_arr)})."
        )

    # Precompute the bare ndarrays the workers actually need. Passing
    # them as top-level args (rather than dereferencing through ``beam``
    # inside each worker call) lets joblib auto-memmap the beam cube,
    # which is the difference between linear scaling and the flat
    # scaling observed in v0.1's benchmark.
    beam_power_cube = beam.power_cube(polarization)
    omega_b_disc = beam.beam_solid_angle(polarization)[beam_freq_idx]

    n_jobs_resolved = _parallel.resolve_n_jobs(n_jobs)
    n_workers = max(1, n_jobs_resolved) if n_jobs_resolved > 0 else (os.cpu_count() or 1)

    if batch_size is None:
        # Aim for ~1 batch per worker. Per-batch overhead (pickling the
        # beam cube to the worker) is ~50 ms regardless of batch size, so
        # one large batch per worker is dramatically faster than many
        # small ones once the cube is non-trivial.
        batch_size = max(1, ntime // n_workers) if n_workers > 1 else ntime
    batch_size = max(1, int(batch_size))

    batches = [
        np.arange(start, min(start + batch_size, ntime), dtype=np.int64)
        for start in range(0, ntime, batch_size)
    ]

    margin_deg = beam.margin_deg
    args = [
        (
            batch,
            lst_arr,
            az_arr,
            el_arr,
            lat_deg,
            beam_power_cube,
            margin_deg,
            omega_b_disc,
            sky_maps,
            sky_freq_idx,
            beam_freq_idx,
            disc_radius_deg,
            polarization,
            horizontal_mask,
        )
        for batch in batches
    ]
    results = _parallel.map_samples(_process_batch, args, n_jobs=n_jobs_resolved, progress=progress)
    return np.concatenate(results, axis=-1)  # (n_freq_out, ntime)


SkyFunc = Callable[..., np.ndarray]


def materialize_sky_cube(
    sky_func: SkyFunc,
    freq_MHz: Sequence[float],
    nside: int,
) -> np.ndarray:
    """Evaluate ``sky_func`` at every requested frequency.

    Convenience helper used by the simulator to build the
    ``(n_freq, n_pix)`` cube once, rather than re-evaluating the sky
    model inside every per-sample call.

    The signature mirrors ``limTOD``'s sky-function convention:
    ``sky_func(freq=..., nside=...) -> ndarray of shape (npix,)``.
    """
    return np.stack(
        [np.asarray(sky_func(freq=float(f), nside=nside)) for f in freq_MHz],
        axis=0,
    )


# Back-compat alias for the original British spelling. Will be removed
# in v0.2; flagged by the architect review.
materialise_sky_cube = materialize_sky_cube
