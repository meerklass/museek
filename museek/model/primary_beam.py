"""
Primary beam model for MeerKAT single-dish observations.

This module provides primary beam gain patterns as a function of pointing direction,
source position, frequency, and polarization for UHF-band observations.
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator


class PrimaryBeam:
    """
    MeerKAT primary beam model.

    Loads beam patterns from NPZ file and provides interpolated beam gains
    as a function of antenna pointing, source position, frequency, and polarization.

    The beam is defined in a local coordinate system centered on the antenna pointing
    direction, using direction cosines (l, m) to represent angular offsets.

    Attributes
    ----------
    freq_range_MHz : tuple[float, float]
        (min_freq, max_freq) in MHz covered by the beam model
    beam_extent_deg : tuple[float, float]
        Spatial extent (l_extent, m_extent) in degrees
    """

    def __init__(self, beam_file: str | Path):
        """
        Load primary beam model from NPZ file.

        Parameters
        ----------
        beam_file : str or Path
            Path to beam NPZ file (e.g., MeerKAT_U_band_primary_beam_aa_highres.npz)

        Raises
        ------
        FileNotFoundError
            If beam file does not exist
        ValueError
            If beam file has unexpected format

        Notes
        -----
        Expected NPZ file format:
        - 'beam': (4, n_ant, n_freq, n_m, n_l) complex array - Jones matrices
          where beam[pol, ant, freq, row, col] with row→m, col→l
        - 'pols': ['HH', 'HV', 'VH', 'VV'] polarization labels
        - 'antnames': antenna names (we use 'array_average')
        - 'freq_MHz': frequency array in MHz
        - 'margin_deg': spatial coordinate grid (same for both m and l axes)

        The beam values are Jones matrix elements. For intensity (power),
        we compute |beam|² for HH and VV polarizations.

        Beam coordinates:
        - l, m are direction cosines in the local tangent plane
        - Beam center (l=0, m=0) corresponds to antenna pointing direction
        - margin_deg gives the grid in units of (l×180/π, m×180/π)
        - Array indexing: beam[..., row, col] where row uses m, col uses l
        """
        beam_file = Path(beam_file)

        if not beam_file.exists():
            raise FileNotFoundError(f"Beam file not found: {beam_file}")

        try:
            # Load NPZ file
            data = np.load(beam_file)

            # Extract beam data
            # Shape: (4, n_ant, n_freq, n_m, n_l) where n_m=rows, n_l=columns
            # Polarizations: 0=HH, 1=HV, 2=VH, 3=VV
            beam_jones = data['beam']

            # For array average beam, use antenna index 0
            # Convert Jones matrices to power: |beam|²
            # Only compute HH and VV (indices 0 and 3)
            beam_HH_jones = beam_jones[0, 0, :, :, :]  # (n_freq, n_m, n_l)
            beam_VV_jones = beam_jones[3, 0, :, :, :]  # (n_freq, n_m, n_l)

            # Calculate beam power
            beam_HH_power = np.abs(beam_HH_jones) ** 2
            beam_VV_power = np.abs(beam_VV_jones) ** 2

            # Extract coordinate grids
            freq_MHz = data['freq_MHz']  # (n_freq,)
            margin_deg = data['margin_deg']  # (n_spatial,) - coordinate grid for both l and m

            # Calculate beam solid angle (steradians)
            # Pixel size in degrees (uniform square grid)
            d_deg = margin_deg[1] - margin_deg[0]  # degrees
            d_rad = d_deg * np.pi / 180  # radians (explicit conversion)
            d_omega = d_rad**2  # steradians per pixel

            # Beam solid angle: integrate beam power over spatial coordinates
            # Sum over m and l axes (1, 2), leaving frequency axis (0)
            self._beam_solid_angle_HH = d_omega * beam_HH_power.sum(axis=(1, 2))  # (n_freq,)
            self._beam_solid_angle_VV = d_omega * beam_VV_power.sum(axis=(1, 2))  # (n_freq,)

            # Store ranges for validation
            self.freq_range_MHz = (freq_MHz.min(), freq_MHz.max())
            self.beam_extent_deg = (margin_deg.min(), margin_deg.max())

            # Create 3D interpolators: f(freq, m_deg, l_deg) -> beam_gain
            # Axes order matches array dimensions: (n_freq, n_m, n_l)
            # Note: margin_deg is the same for both m and l coordinates (square grid)
            self._interpolator_HH = RegularGridInterpolator(
                (freq_MHz, margin_deg, margin_deg),  # (freq, m_coords, l_coords)
                beam_HH_power,
                method='linear',
                bounds_error=False,
                fill_value=0.0  # Return 0 for points outside beam
            )

            self._interpolator_VV = RegularGridInterpolator(
                (freq_MHz, margin_deg, margin_deg),  # (freq, m_coords, l_coords)
                beam_VV_power,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )

            # Store for reference
            self._beam_file = beam_file
            self._freq_MHz = freq_MHz
            self._margin_deg = margin_deg

        except KeyError as e:
            raise ValueError(
                f"Beam file {beam_file} missing required key: {e}. "
                f"Expected keys: 'beam', 'freq_MHz', 'margin_deg'"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error loading beam file {beam_file}: {e}"
            ) from e

    def calculate_direction_cosines(
        self,
        az_pointing: np.ndarray,
        el_pointing: np.ndarray,
        az_source: np.ndarray,
        el_source: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate direction cosines (l, m) from pointing and source coordinates.

        This transforms source coordinates from the sky frame (az, el) to the
        local tangent plane centered on the antenna pointing direction.

        Parameters
        ----------
        az_pointing : np.ndarray
            Antenna pointing azimuth in degrees, shape (n_time,)
            This is the beam center direction (same for all antennas).
        el_pointing : np.ndarray
            Antenna pointing elevation in degrees, shape (n_time,)
            This is the beam center direction (same for all antennas).
        az_source : np.ndarray
            Source azimuth in degrees, shape (n_time,)
            Source position varies with time as Earth rotates.
        el_source : np.ndarray
            Source elevation in degrees, shape (n_time,)
            Source position varies with time as Earth rotates.

        Returns
        -------
        l : np.ndarray
            Direction cosine l (dimensionless), shape (n_time,)
        m : np.ndarray
            Direction cosine m (dimensionless), shape (n_time,)

        Notes
        -----
        Direction cosines are calculated using the transformations:
            l = cos(el_pointing) * sin(az_source - az_pointing)
            m = cos(el_pointing) * sin(el_source) * cos(az_source - az_pointing)
                - sin(el_pointing) * cos(el_source)

        These give the position of the source in the beam's local coordinate system,
        where (l=0, m=0) corresponds to the beam center (antenna pointing direction).

        Direction cosines are dimensionless quantities, typically in range ±0.1
        for beam extents of ~±6 degrees.

        For calibration: source has fixed (RA, Dec) but varies in (az, el) with time.
        User must transform (RA, Dec) → (az, el) using astropy before calling this.
        """
        # Convert degrees to radians
        az_p_rad = np.radians(az_pointing)
        el_p_rad = np.radians(el_pointing)
        az_s_rad = np.radians(az_source)
        el_s_rad = np.radians(el_source)

        # Calculate direction cosines (dimensionless)
        # l = cos(el_pointing) * sin(az_source - az_pointing)
        l = np.cos(el_s_rad) * np.sin(az_s_rad - az_p_rad)

        # m = cos(el_pointing) * sin(el_source) * cos(az_source - az_pointing)
        #     - sin(el_pointing) * cos(el_source)
        m = (
            np.sin(el_s_rad) * np.cos(el_p_rad) -
            np.cos(el_s_rad) * np.sin(el_p_rad) * np.cos(az_s_rad - az_p_rad)
        )

        return l, m

    def get_beam_gain(
        self,
        az_pointing: np.ndarray,
        el_pointing: np.ndarray,
        az_source: np.ndarray,
        el_source: np.ndarray,
        frequency_MHz: np.ndarray,
        polarization: str
    ) -> np.ndarray:
        """
        Get primary beam gain for given coordinates.

        Assumes array average beam (same for all antennas) and all antennas
        point in the same direction.

        Parameters
        ----------
        az_pointing : np.ndarray
            Antenna pointing azimuth in degrees, shape (n_time,)
            All antennas point in this direction (same for all).
        el_pointing : np.ndarray
            Antenna pointing elevation in degrees, shape (n_time,)
            All antennas point in this direction (same for all).
        az_source : np.ndarray
            Source azimuth in degrees, shape (n_time,)
            Source position varies with time as Earth rotates (fixed RA/Dec).
        el_source : np.ndarray
            Source elevation in degrees, shape (n_time,)
            Source position varies with time as Earth rotates (fixed RA/Dec).
        frequency_MHz : np.ndarray
            Observing frequencies in MHz, shape (n_freq,)
        polarization : str
            Polarization: 'HH', 'hh', 'VV', or 'vv'

        Returns
        -------
        beam_gain : np.ndarray
            Normalized beam power gain (0-1), shape (n_time, n_freq)
            Same beam gain applies to all antennas (array average).

        Raises
        ------
        ValueError
            If polarization is not 'HH' or 'VV'

        Notes
        -----
        Beam gain represents the fraction of the beam's peak response at the
        given position. Values are:
        - 1.0 at beam center (when source is at pointing direction)
        - < 1.0 at offset positions
        - 0.0 outside the beam pattern extent

        For point source calibration:
        - Source has fixed (RA, Dec) coordinates
        - User transforms to time-varying (az, el) using astropy before calling
        - Beam gain varies with time as relative pointing changes
        - Beam gain varies with frequency due to frequency-dependent beam patterns
        - Same beam applies to all receivers (array average beam)

        Examples
        --------
        >>> from astropy.coordinates import SkyCoord, AltAz
        >>> from astropy.time import Time
        >>> import astropy.units as u
        >>>
        >>> beam = PrimaryBeam('MeerKAT_U_band_primary_beam_aa_highres.npz')
        >>>
        >>> # Transform calibrator (RA, Dec) to (az, el)
        >>> cal_radec = SkyCoord(ra=180*u.deg, dec=-30*u.deg)
        >>> times = Time(np.linspace(58000, 58000.1, 100), format='mjd')
        >>> altaz_frame = AltAz(obstime=times, location=site_location)
        >>> cal_altaz = cal_radec.transform_to(altaz_frame)
        >>>
        >>> # Antenna pointing (n_time=100, same for all antennas)
        >>> az_point = np.linspace(170, 190, 100)  # Scanning across calibrator
        >>> el_point = np.ones(100) * 45.0
        >>>
        >>> # Frequencies
        >>> freq = np.linspace(580, 1015, 436)
        >>>
        >>> # Get beam gain
        >>> gain = beam.get_beam_gain(
        ...     az_point, el_point,
        ...     cal_altaz.az.deg, cal_altaz.alt.deg,
        ...     freq, 'HH'
        ... )
        >>> gain.shape
        (100, 436)
        """
        # Validate polarization
        pol_key = polarization.upper()
        if pol_key not in ('HH', 'VV'):
            raise ValueError(
                f"Polarization must be 'HH' or 'VV' (case insensitive), got '{polarization}'"
            )

        # Select interpolator
        interpolator = self._interpolator_HH if pol_key == 'HH' else self._interpolator_VV

        # Calculate direction cosines (dimensionless)
        l, m = self.calculate_direction_cosines(
            az_pointing, el_pointing, az_source, el_source
        )  # shape (n_time,)

        # Convert to beam file coordinate convention: direction_cosine × 180/π
        l_deg = l * 180.0 / np.pi
        m_deg = m * 180.0 / np.pi

        # Prepare points for interpolation
        # Need to broadcast to (n_time, n_freq) evaluation points
        n_time = len(az_pointing)
        n_freq = len(frequency_MHz)

        # Create meshgrid for all (time, freq) combinations
        # Interpolator expects points with shape (n_points, 3) where 3 = (freq, m, l)

        # Broadcast arrays to (n_time, n_freq)
        freq_broadcast = frequency_MHz[np.newaxis, :]  # (1, n_freq)
        l_broadcast = l_deg[:, np.newaxis]  # (n_time, 1)
        m_broadcast = m_deg[:, np.newaxis]  # (n_time, 1)

        # Expand to full grid
        freq_grid = np.broadcast_to(freq_broadcast, (n_time, n_freq))
        l_grid = np.broadcast_to(l_broadcast, (n_time, n_freq))
        m_grid = np.broadcast_to(m_broadcast, (n_time, n_freq))

        # Flatten for interpolation
        # Order must match interpolator axes: (freq, m, l)
        points = np.stack([freq_grid.ravel(), m_grid.ravel(), l_grid.ravel()], axis=1)

        # Interpolate
        beam_gain_flat = interpolator(points)

        # Reshape back to (n_time, n_freq)
        beam_gain = beam_gain_flat.reshape((n_time, n_freq))

        return beam_gain

    @property
    def beam_solid_angle_HH(self) -> np.ndarray:
        """
        Beam solid angle for HH polarization.

        Returns
        -------
        beam_solid_angle : np.ndarray
            Beam solid angle in steradians, shape (n_freq,)

        Notes
        -----
        The beam solid angle Ω is calculated by integrating the normalized
        beam power pattern over the sky:
            Ω = ∫∫ beam_power(l,m) dΩ

        where dΩ = dl×dm is the solid angle element in steradians.

        For calculating effective area: A_eff = λ²/Ω_beam
        For antenna temperature: T_ant = (A_eff / 2k_B) × S × G_beam
        where S is flux density and G_beam is the beam gain.
        """
        return self._beam_solid_angle_HH

    @property
    def beam_solid_angle_VV(self) -> np.ndarray:
        """
        Beam solid angle for VV polarization.

        Returns
        -------
        beam_solid_angle : np.ndarray
            Beam solid angle in steradians, shape (n_freq,)

        Notes
        -----
        See beam_solid_angle_HH documentation for details.
        """
        return self._beam_solid_angle_VV

    def get_beam_solid_angle(self, polarization: str) -> np.ndarray:
        """
        Get beam solid angle for specified polarization.

        Parameters
        ----------
        polarization : str
            Polarization: 'HH', 'hh', 'VV', or 'vv'

        Returns
        -------
        beam_solid_angle : np.ndarray
            Beam solid angle in steradians, shape (n_freq,)

        Raises
        ------
        ValueError
            If polarization is not 'HH' or 'VV'
        """
        pol_key = polarization.upper()
        if pol_key not in ('HH', 'VV'):
            raise ValueError(
                f"Polarization must be 'HH' or 'VV' (case insensitive), got '{polarization}'"
            )

        return self._beam_solid_angle_HH if pol_key == 'HH' else self._beam_solid_angle_VV

    def get_beam_solid_angle_at_freq(
        self,
        frequency_MHz: np.ndarray,
        polarization: str
    ) -> np.ndarray:
        """
        Get beam solid angle interpolated at specific frequencies.

        Parameters
        ----------
        frequency_MHz : np.ndarray
            Frequencies in MHz, any shape
        polarization : str
            Polarization: 'HH', 'hh', 'VV', or 'vv'

        Returns
        -------
        beam_solid_angle : np.ndarray
            Beam solid angle in steradians, same shape as frequency_MHz

        Raises
        ------
        ValueError
            If polarization is not 'HH' or 'VV'

        Notes
        -----
        Uses linear interpolation between frequency points.
        Returns 0 for frequencies outside the beam model range.
        """
        pol_key = polarization.upper()
        if pol_key not in ('HH', 'VV'):
            raise ValueError(
                f"Polarization must be 'HH' or 'VV' (case insensitive), got '{polarization}'"
            )

        # Get solid angle array for this polarization
        solid_angle = self._beam_solid_angle_HH if pol_key == 'HH' else self._beam_solid_angle_VV

        # Interpolate to requested frequencies
        original_shape = frequency_MHz.shape
        freq_flat = frequency_MHz.ravel()

        # Linear interpolation with constant extrapolation at edges
        solid_angle_interp = np.interp(
            freq_flat,
            self._freq_MHz,
            solid_angle,
            left=solid_angle[0],   # Use edge value for low frequencies
            right=solid_angle[-1]  # Use edge value for high frequencies
        )

        return solid_angle_interp.reshape(original_shape)

    def evaluate_beam(
        self,
        frequency_MHz: np.ndarray | float,
        l: np.ndarray | float,
        m: np.ndarray | float,
        polarization: str
    ) -> np.ndarray | float:
        """
        Evaluate beam gain directly at direction cosine coordinates.

        This is a lower-level method useful for testing and visualization.
        For point source calibration, use get_beam_gain() instead.

        Parameters
        ----------
        frequency_MHz : np.ndarray or float
            Frequency in MHz, any shape
        l : np.ndarray or float
            Direction cosine l (dimensionless, azimuth-like), same shape as frequency
        m : np.ndarray or float
            Direction cosine m (dimensionless, elevation-like), same shape as frequency
        polarization : str
            Polarization: 'HH', 'hh', 'VV', or 'vv'

        Returns
        -------
        beam_gain : np.ndarray or float
            Beam gain, same shape as inputs

        Raises
        ------
        ValueError
            If polarization is not 'HH' or 'VV'

        Notes
        -----
        Direction cosines l, m are dimensionless quantities, typically in range
        ±0.1 for beam extents of ~±6 degrees (since sin(6°) ≈ 0.105).

        For small angles: l ≈ sin(Δaz), m ≈ sin(Δel) where Δaz, Δel are
        angular offsets from beam center.

        Examples
        --------
        >>> beam = PrimaryBeam('MeerKAT_U_band_primary_beam_aa_highres.npz')
        >>> # Beam gain at center
        >>> gain_center = beam.evaluate_beam(850, 0.0, 0.0, 'HH')
        >>> # Beam gain along l axis at fixed frequency (scan ±3° in azimuth)
        >>> offsets_deg = np.linspace(-3, 3, 100)
        >>> l = np.sin(np.radians(offsets_deg))  # Convert to direction cosines
        >>> m = np.zeros_like(l)
        >>> freq = np.full_like(l, 850)
        >>> gain = beam.evaluate_beam(freq, l, m, 'HH')
        """
        # Validate polarization
        pol_key = polarization.upper()
        if pol_key not in ('HH', 'VV'):
            raise ValueError(
                f"Polarization must be 'HH' or 'VV' (case insensitive), got '{polarization}'"
            )

        # Select interpolator
        interpolator = self._interpolator_HH if pol_key == 'HH' else self._interpolator_VV

        # Convert to arrays
        freq_array = np.atleast_1d(frequency_MHz)
        l_array = np.atleast_1d(l)
        m_array = np.atleast_1d(m)

        # Check shapes match
        if not (freq_array.shape == l_array.shape == m_array.shape):
            raise ValueError(
                f"All inputs must have the same shape. "
                f"Got freq={freq_array.shape}, l={l_array.shape}, m={m_array.shape}"
            )

        # Convert direction cosines to beam file convention: × 180/π
        l_deg = l_array * 180.0 / np.pi
        m_deg = m_array * 180.0 / np.pi

        # Stack points for interpolation: (n_points, 3) with columns (freq, m, l)
        points = np.stack([freq_array.ravel(), m_deg.ravel(), l_deg.ravel()], axis=1)

        # Interpolate
        beam_gain_flat = interpolator(points)

        # Reshape to original shape
        beam_gain = beam_gain_flat.reshape(freq_array.shape)

        # Return scalar if inputs were scalars
        if np.isscalar(frequency_MHz) and np.isscalar(l) and np.isscalar(m):
            return float(beam_gain.item())

        return beam_gain

    def __repr__(self):
        """String representation."""
        return (
            f"PrimaryBeam(file='{self._beam_file.name}', "
            f"freq_range={self.freq_range_MHz[0]:.1f}-{self.freq_range_MHz[1]:.1f} MHz, "
            f"extent=±{abs(self.beam_extent_deg[0]):.1f}°)"
        )
