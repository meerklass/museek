"""
Noise diode temperature model for MeerKAT single-dish observations.

This module provides noise diode calibration temperature as a function of frequency
for individual receivers identified by their serial numbers.
"""

import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path


class NoiseDiodeTemperature:
    """
    Noise diode temperature interpolator.

    Loads noise diode model file and provides temperature as a function of frequency.
    """

    def __init__(self, receiver, data_dir: str | Path):
        """
        Load noise diode temperature model.

        Parameters
        ----------
        receiver : Receiver
            Receiver object with receiver_id populated (e.g., 'u.4001')
        data_dir : str or Path
            Directory containing noise diode model files

        Attributes
        ----------
        freq_range_MHz : tuple
            (min_freq, max_freq) in MHz from the data file

        Raises
        ------
        ValueError
            If receiver.receiver_id is not set or has invalid format
        FileNotFoundError
            If noise diode model file does not exist

        Examples
        --------
        >>> from museek.receiver import Receiver
        >>> receiver = track_data.receivers[0]  # Receiver with receiver_id='u.4001'
        >>> nd = NoiseDiodeTemperature(receiver, 'data/noise_diodes')
        >>> T = nd(np.array([700., 800., 900.]))  # MHz -> K
        """
        if receiver.receiver_id is None:
            raise ValueError(
                f"Receiver {receiver.name} does not have receiver_id populated. "
                f"Ensure TimeOrderedData was created from katdal data with receiver metadata."
            )

        # Parse receiver ID: 'u.4001' -> band='u', serial='4001'
        try:
            band, serial = receiver.receiver_id.split('.')
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Invalid receiver_id format: '{receiver.receiver_id}' for {receiver.name}. "
                f"Expected format: 'band.serial_number' (e.g., 'u.4001')"
            ) from e

        # Get polarization from receiver: 'h' or 'v' (lowercase)
        pol = receiver.polarisation.lower()

        # Construct filename: rx.u.4001.h.csv
        filename = f'rx.{band}.{serial}.{pol}.csv'
        filepath = Path(data_dir) / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Noise diode model not found: {filepath}\n"
                f"receiver={receiver.name}, receiver_id={receiver.receiver_id}"
            )

        try:
            # Load file: frequency (Hz), temperature (K)
            data = np.loadtxt(filepath, delimiter=',', comments='#')

            # Extract frequency (Hz -> MHz) and temperature (K)
            freq_MHz = data[:, 0] / 1e6
            T_nd = data[:, 1]

            self.freq_range_MHz = (freq_MHz.min(), freq_MHz.max())

            # Create linear interpolator with constant extrapolation at edges
            self._interpolator = interp1d(
                freq_MHz, T_nd,
                kind='linear',
                bounds_error=False,  # Allow out-of-range queries
                fill_value=(T_nd[0], T_nd[-1])  # Use edge values for extrapolation
            )

        except (ValueError, IndexError) as e:
            raise ValueError(
                f'Invalid noise diode model file format: {filepath}\n'
                f'Expected CSV: frequency (Hz), temperature (K)'
            ) from e

    def __call__(self, frequency_MHz: np.ndarray | float) -> np.ndarray | float:
        """
        Get noise diode temperature for given frequency(ies).

        Parameters
        ----------
        frequency_MHz : np.ndarray or float
            Frequency(ies) in MHz

        Returns
        -------
        T_nd : np.ndarray or float
            Noise diode temperature in Kelvin

        Notes
        -----
        Frequencies outside the measured range are extrapolated using constant
        edge values (nearest neighbor extrapolation). This provides conservative
        estimates for calibration at band edges. A warning is issued when
        extrapolation occurs.
        """
        freq_array = np.atleast_1d(frequency_MHz)
        if freq_array.min() < self.freq_range_MHz[0] or freq_array.max() > self.freq_range_MHz[1]:
            import warnings
            warnings.warn(
                f"Frequency partially out of range: {freq_array.min():.1f}-{freq_array.max():.1f} MHz. "
                f"Valid: {self.freq_range_MHz[0]:.1f}-{self.freq_range_MHz[1]:.1f} MHz. "
                f"Using constant extrapolation at edges.",
                UserWarning
            )
        return self._interpolator(frequency_MHz)
