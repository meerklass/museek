"""
Spillover temperature model for MeerKAT single-dish observations.

This module provides spillover (ground pickup) temperature models as a function
of elevation and frequency for both L-band and UHF-band observations.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from pathlib import Path


class SpilloverTemperature:
    """
    Load and interpolate spillover temperature models.

    The spillover temperature represents ground and structure pickup that varies
    with telescope elevation and observing frequency.
    """

    def __init__(self, filename: str | Path):
        """
        Initialize spillover temperature model from data file.

        Parameters
        ----------
        filename : str or Path
            Path to spillover model data file. File format:
            - Row 1: Comment header
            - Row 2: ZA=0, then frequencies in MHz (alternating H/V pairs)
            - Subsequent rows: ZA [deg], then T_H, T_V values for each frequency
            Required for calibration - will raise error if loading fails.

        Attributes
        ----------
        spill : dict
            Dictionary with keys 'HH' and 'VV' containing interpolation functions.
            Each function takes (elevation_array, frequency_MHz_array) and returns
            temperature in Kelvin.
        freq_range_MHz : tuple
            (min_freq, max_freq) in MHz from the data file
        elevation_range : tuple
            (min_el, max_el) in degrees from the data file
        """
        self.spill = {}
        self.filename = filename

        try:
            # Load spillover data file
            data = np.loadtxt(filename)

            # Extract elevation (90 - zenith angle from column 0, rows 2+)
            zenith_angle = data[1:, 0]
            elevation = 90.0 - zenith_angle

            # RectBivariateSpline requires strictly increasing x values
            # Since elevation decreases as ZA increases, we need to reverse the arrays
            elevation = elevation[::-1]

            # Extract frequencies from row 1 (skip first column, take every 2nd starting at 1)
            # Format: ZA, freq1, freq1, freq2, freq2, ... (H and V share same freq value)
            frequencies = data[0, 1::2]  # MHz

            # Extract spillover temperatures (reverse rows to match reversed elevation)
            # Columns alternate: H @ freq1, V @ freq1, H @ freq2, V @ freq2, ...
            T_HH_data = data[1:, 1::2][::-1]  # All odd columns (H polarization), reversed
            T_VV_data = data[1:, 2::2][::-1]  # All even columns (V polarization), reversed

            # Store frequency and elevation ranges for validation
            self.freq_range_MHz = (frequencies.min(), frequencies.max())
            self.elevation_range = (elevation.min(), elevation.max())

            # Create 2D spline interpolators (elevation, frequency) -> T_spill
            # Use cubic splines (kx=3, ky=3) for smooth interpolation
            self.spill['HH'] = RectBivariateSpline(
                elevation, frequencies, T_HH_data,
                kx=3, ky=3, s=0  # s=0 for interpolation (not smoothing)
            )
            self.spill['VV'] = RectBivariateSpline(
                elevation, frequencies, T_VV_data,
                kx=3, ky=3, s=0
            )

        except (IOError, FileNotFoundError) as e:
            raise FileNotFoundError(
                f'Failed to load spillover model from {filename}: {e}. '
                f'Spillover temperature model is required for calibration.'
            ) from e
        except (ValueError, IndexError) as e:
            raise ValueError(
                f'Invalid spillover model file format in {filename}: {e}. '
                f'Expected format: Row 1=header, Row 2=frequencies, subsequent rows=ZA and temperatures.'
            ) from e

    def get_temperature(self, elevation: np.ndarray, frequency_MHz: np.ndarray,
                       polarization: str) -> np.ndarray:
        """
        Get spillover temperature for given elevation(s) and frequency(ies).

        Parameters
        ----------
        elevation : np.ndarray
            Elevation angle(s) in degrees. Shape: (n_times,) or scalar
        frequency_MHz : np.ndarray
            Frequency(ies) in MHz. Shape: (n_freqs,) or scalar
        polarization : str
            Polarization: 'HH', 'hh', 'VV', or 'vv'

        Returns
        -------
        T_spill : np.ndarray
            Spillover temperature in Kelvin.
            If elevation is (n_times,) and frequency is (n_freqs,):
                Returns shape (n_times, n_freqs)
            Broadcasting rules apply for scalar inputs.

        Examples
        --------
        >>> spill = SpilloverTemperature('spillover_model.dat')
        >>> el = np.array([45., 60., 75.])  # 3 elevations
        >>> freq = np.array([700., 800., 900.])  # 3 frequencies in MHz
        >>> T = spill.get_temperature(el, freq, 'HH')  # Shape: (3, 3)
        """
        # Normalize polarization input to uppercase
        pol_key = polarization.upper()
        if pol_key not in ('HH', 'VV'):
            raise ValueError(f"Polarization must be 'HH' or 'VV' (case insensitive), got '{polarization}'")

        # Ensure inputs are arrays
        el_array = np.atleast_1d(elevation)
        freq_array = np.atleast_1d(frequency_MHz)

        # Evaluate interpolator
        # RectBivariateSpline returns shape (len(el), len(freq))
        T_spill = self.spill[pol_key](el_array, freq_array)

        # If inputs were scalars, return scalar
        if np.isscalar(elevation) and np.isscalar(frequency_MHz):
            return float(T_spill[0, 0])

        return T_spill
