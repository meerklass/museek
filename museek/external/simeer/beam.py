"""
MeerKLASS holographic primary beam wrapper.

The beam ships as an NPZ archive containing complex Jones matrices on a
regular ``(n_freq, n_m, n_l)`` grid in direction cosine space (the
``margin_deg`` array of the file is the same grid for both ``m`` and
``l`` axes). The dataset is large (tens of GB for the raw cube), so this
module:

*   loads only the requested antenna entry (typically ``array_average``)
    and the requested linear polarisations (default HH and VV);
*   stores the **power** ``|Jones|^2`` in ``float32`` once at
    construction time, so the working cube is ~1/4 the size of the raw
    complex Jones for the selected antenna;
*   exposes the cube as a contiguous ndarray that downstream
    :mod:`simeer.interpolation` can fancy-index.

For unit tests and for the synthetic Gaussian benchmark we also provide
:meth:`MeerKLASSBeam.from_arrays` to construct a beam from in-memory
arrays without an NPZ file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import DTypeLike

_VALID_POLS = ("HH", "VV")
_POL_INDEX = {"HH": 0, "VV": 3}  # indices into the Jones matrix axis


def _validate_uniform_grid(grid: np.ndarray, name: str, rtol: float = 1e-9) -> np.ndarray:
    """Assert that a 1D coordinate grid is uniformly spaced.

    Returns the grid unchanged. Raises ``ValueError`` otherwise.
    """
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError(f"{name} must be a 1D array of length >= 2.")
    spacing = np.diff(grid)
    if not np.allclose(spacing, spacing[0], rtol=rtol):
        raise ValueError(
            f"{name} must be uniformly spaced for beam_solid_angle to be correct "
            f"(observed min/max diff: {spacing.min()} / {spacing.max()})."
        )
    return grid


@dataclass(frozen=True)
class _BeamArrays:
    """Internal container for the per-pol power cubes and shared grid.

    ``power`` is wrapped in :class:`types.MappingProxyType` so external
    code cannot replace or insert entries; the underlying ndarrays remain
    mutable, but callers should treat them as read-only.
    """

    freq_MHz: np.ndarray
    margin_deg: np.ndarray
    power: Mapping[str, np.ndarray]  # {'HH': cube, 'VV': cube}


class MeerKLASSBeam:
    """In-memory wrapper around a MeerKLASS holographic beam file.

    Parameters
    ----------
    beam_file : str or Path
        Path to the NPZ archive. Expected keys: ``beam`` (complex,
        shape ``(4, n_ant, n_freq, n_m, n_l)``), ``pols``, ``antnames``,
        ``freq_MHz``, ``margin_deg``.
    antenna : str, default ``'array_average'``
        Antenna name in ``antnames`` to extract. The default selects the
        array-average beam used by the existing MeerKLASS calibration.
    polarizations : sequence of str, default ``('HH', 'VV')``
        Subset of ``{'HH', 'VV'}`` to materialise. Only the requested
        polarisations occupy memory.
    dtype : numpy dtype, default ``np.float32``
        Working dtype of the power cubes. Float32 is a 2x memory win
        over float64 and is well below the intrinsic precision of the
        holographic measurement.

    Attributes
    ----------
    freq_MHz : ndarray
        Frequency grid of the beam (1D, monotonic, MHz).
    margin_deg : ndarray
        Angular grid used for both the m and l axes (1D, monotonic, deg).
    polarizations : tuple of str
        Polarisations that were materialised.
    """

    def __init__(
        self,
        beam_file: str | Path,
        *,
        antenna: str = "array_average",
        polarizations: Sequence[str] = ("HH", "VV"),
        dtype: DTypeLike = np.float32,
    ):
        beam_file = Path(beam_file)
        if not beam_file.exists():
            raise FileNotFoundError(f"Beam file not found: {beam_file}")

        polarizations = tuple(p.upper() for p in polarizations)
        for pol in polarizations:
            if pol not in _VALID_POLS:
                raise ValueError(
                    f"Polarization '{pol}' not supported. Must be one of {_VALID_POLS}."
                )

        with np.load(beam_file) as data:
            raw_antnames = np.asarray(data["antnames"])
            # NPZ archives store strings as numpy bytes (np.bytes_). Decode
            # to plain str so substring lookup works ergonomically.
            antnames = [
                a.decode() if isinstance(a, (bytes, np.bytes_)) else str(a) for a in raw_antnames
            ]
            try:
                ant_idx = antnames.index(antenna)
            except ValueError as exc:
                raise ValueError(
                    f"Antenna '{antenna}' not found in beam file. "
                    f"Available: {antnames[:5]}{'...' if len(antnames) > 5 else ''}"
                ) from exc

            freq_MHz = np.asarray(data["freq_MHz"], dtype=np.float64)
            margin_deg = np.asarray(data["margin_deg"], dtype=np.float64)

            power: dict[str, np.ndarray] = {}
            beam_raw = data["beam"]
            for pol in polarizations:
                jones = np.asarray(beam_raw[_POL_INDEX[pol], ant_idx, :, :, :])
                power[pol] = np.ascontiguousarray((jones.real**2 + jones.imag**2), dtype=dtype)

        self._arrays = _BeamArrays(
            freq_MHz=freq_MHz,
            margin_deg=_validate_uniform_grid(margin_deg, "margin_deg"),
            power=MappingProxyType(power),
        )
        self._polarizations = polarizations
        self._beam_file = beam_file
        self._beam_solid_angle: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_arrays(
        cls,
        *,
        freq_MHz: np.ndarray,
        margin_deg: np.ndarray,
        power: Mapping[str, np.ndarray],
    ) -> MeerKLASSBeam:
        """Construct a beam from in-memory power cubes.

        Mostly used by tests and the synthetic Gaussian benchmark. Each
        entry in ``power`` must have shape ``(n_freq, n_m, n_l)``
        matching ``freq_MHz`` and ``margin_deg``. ``margin_deg`` must be
        uniformly spaced (a constant cell width is assumed by
        :meth:`beam_solid_angle`).
        """
        polarizations = tuple(p.upper() for p in power.keys())
        for pol in polarizations:
            if pol not in _VALID_POLS:
                raise ValueError(f"Polarization '{pol}' not supported.")
        expected_shape = (len(freq_MHz), len(margin_deg), len(margin_deg))
        normalised: dict[str, np.ndarray] = {}
        for pol in polarizations:
            cube = np.ascontiguousarray(power[pol])
            if cube.shape != expected_shape:
                raise ValueError(
                    f"Power cube for '{pol}' has shape {cube.shape}, " f"expected {expected_shape}."
                )
            normalised[pol] = cube

        obj = cls.__new__(cls)
        obj._arrays = _BeamArrays(
            freq_MHz=np.asarray(freq_MHz, dtype=np.float64),
            margin_deg=_validate_uniform_grid(
                np.asarray(margin_deg, dtype=np.float64), "margin_deg"
            ),
            power=MappingProxyType(normalised),
        )
        obj._polarizations = polarizations
        obj._beam_file = Path("<in-memory>")
        obj._beam_solid_angle = {}
        return obj

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #
    @property
    def freq_MHz(self) -> np.ndarray:
        return self._arrays.freq_MHz

    @property
    def margin_deg(self) -> np.ndarray:
        return self._arrays.margin_deg

    @property
    def polarizations(self) -> tuple[str, ...]:
        return self._polarizations

    @property
    def freq_range_MHz(self) -> tuple[float, float]:
        f = self._arrays.freq_MHz
        return float(f[0]), float(f[-1])

    @property
    def beam_extent_deg(self) -> tuple[float, float]:
        m = self._arrays.margin_deg
        return float(m[0]), float(m[-1])

    # ------------------------------------------------------------------ #
    # Accessors                                                          #
    # ------------------------------------------------------------------ #
    def power_cube(self, polarization: str) -> np.ndarray:
        """Return the ``(n_freq, n_m, n_l)`` power cube for one pol."""
        pol = polarization.upper()
        if pol not in self._arrays.power:
            raise KeyError(
                f"Polarization '{polarization}' was not materialised. "
                f"Available: {self._polarizations}"
            )
        return self._arrays.power[pol]

    def freq_indices(self, freq_MHz: Iterable[float], tol: float = 1e-3) -> np.ndarray:
        """Find indices in the beam frequency grid matching ``freq_MHz``.

        Raises ``ValueError`` if any requested frequency is farther than
        ``tol`` MHz from the nearest beam channel. This is the right
        default for MeerKLASS, where users typically work on the native
        beam frequency grid; pass ``tol=np.inf`` to disable the check.
        """
        freq_MHz = np.asarray(freq_MHz, dtype=np.float64)
        grid = self._arrays.freq_MHz
        idx = np.searchsorted(grid, freq_MHz)
        idx = np.clip(idx, 1, len(grid) - 1)
        left = grid[idx - 1]
        right = grid[idx]
        choose_right = (right - freq_MHz) < (freq_MHz - left)
        chosen = np.where(choose_right, idx, idx - 1)
        residual = np.abs(grid[chosen] - freq_MHz)
        if np.any(residual > tol):
            offending = freq_MHz[residual > tol]
            raise ValueError(
                f"Requested frequencies are not aligned with the beam grid "
                f"(tolerance {tol} MHz). Offending: {offending[:5]}..."
            )
        return chosen.astype(np.int64)

    def beam_solid_angle(self, polarization: str) -> np.ndarray:
        """Beam solid angle Omega_b(freq) in steradians.

        Computed once and cached. Uses the on-grid sum with uniform
        spacing implied by ``margin_deg``.
        """
        pol = polarization.upper()
        cached = self._beam_solid_angle.get(pol)
        if cached is not None:
            return cached

        cube = self.power_cube(pol)
        margin = self._arrays.margin_deg
        d_deg = float(margin[1] - margin[0])
        d_rad = np.deg2rad(d_deg)
        d_omega = d_rad * d_rad  # steradians per pixel
        omega = (cube.sum(axis=(1, 2)) * d_omega).astype(np.float64)
        self._beam_solid_angle[pol] = omega
        return omega

    def evaluate(
        self,
        freq_MHz: float | np.ndarray,
        l_deg: np.ndarray,
        m_deg: np.ndarray,
        polarization: str = "HH",
    ) -> np.ndarray:
        """Convenience interpolator for arbitrary (freq, l, m) queries.

        For the hot TOD-integration path you almost certainly want
        :mod:`simeer.interpolation` directly (it precomputes the (l, m)
        weights once per pointing and applies them to all frequencies in
        a single vectorised pass).
        """
        from . import interpolation

        l_deg = np.atleast_1d(l_deg).astype(np.float64)
        m_deg = np.atleast_1d(m_deg).astype(np.float64)
        freq_arr = np.atleast_1d(freq_MHz).astype(np.float64)

        freq_idx = self.freq_indices(freq_arr, tol=np.inf)
        weights = interpolation.precompute_bilinear_weights(
            l_deg, m_deg, self._arrays.margin_deg, self._arrays.margin_deg
        )
        cube = self.power_cube(polarization)
        return interpolation.apply_bilinear(weights, cube, freq_idx)

    def __repr__(self) -> str:
        f_lo, f_hi = self.freq_range_MHz
        ext_lo, ext_hi = self.beam_extent_deg
        return (
            f"MeerKLASSBeam(file='{self._beam_file.name}', "
            f"freq={f_lo:.1f}-{f_hi:.1f} MHz, "
            f"extent=[{ext_lo:.2f}, {ext_hi:.2f}] deg, "
            f"pols={self._polarizations})"
        )


def synthetic_gaussian_beam(
    *,
    freq_MHz: np.ndarray,
    margin_deg: np.ndarray,
    fwhm_deg: float | np.ndarray = 1.1,
    polarizations: Sequence[str] = ("HH", "VV"),
) -> MeerKLASSBeam:
    """Build an in-memory beam with circular Gaussian *power* response.

    The returned power beam satisfies ``B(0) = 1`` and ``B(r) = 0.5`` at
    radius ``r = fwhm_deg / 2``. Internally the cube stores
    ``exp(-r^2 / sigma_p^2)`` with ``sigma_p = FWHM / (2 sqrt(ln 2))``,
    which is the *power*-Gaussian sigma (not the amplitude-Gaussian
    sigma ``FWHM / (2 sqrt(2 ln 2))``).

    Used by unit tests and by the cross-check against limTOD's HEALPix
    path. ``fwhm_deg`` can be a scalar or per-frequency array.
    """
    freq_MHz = np.asarray(freq_MHz, dtype=np.float64)
    margin_deg = np.asarray(margin_deg, dtype=np.float64)
    fwhm = np.broadcast_to(np.asarray(fwhm_deg, dtype=np.float64), freq_MHz.shape).copy()
    # Power-beam sigma: B(r) = exp(-r^2 / sigma^2) reaches 1/2 at r = sigma*sqrt(ln 2).
    sigma = fwhm / (2.0 * np.sqrt(np.log(2.0)))

    m_grid, l_grid = np.meshgrid(margin_deg, margin_deg, indexing="ij")
    r2 = m_grid**2 + l_grid**2
    power = np.exp(-r2[None, :, :] / sigma[:, None, None] ** 2).astype(np.float32)

    cubes = {pol.upper(): power.copy() for pol in polarizations}
    return MeerKLASSBeam.from_arrays(freq_MHz=freq_MHz, margin_deg=margin_deg, power=cubes)
