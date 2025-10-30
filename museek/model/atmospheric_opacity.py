"""
Atmospheric opacity and emission model for MeerKAT single-dish observations.

This module implements the ITU-R P.676-9 standard for calculating atmospheric
opacity due to oxygen and water vapor absorption. It provides atmospheric
transmission and emission temperature as functions of time, frequency, and elevation.
"""

import numpy as np


def calc_zenith_opacity(
    temperature_C: np.ndarray,
    relative_humidity: np.ndarray,
    pressure_hPa: np.ndarray,
    height_km: float,
    frequency_GHz: np.ndarray
) -> np.ndarray:
    """
    Calculate atmospheric zenith opacity using ITU-R P.676-9 standard.

    This function implements the ITU-R P.676-9 model for atmospheric attenuation
    due to dry air (oxygen) and water vapor. Valid for frequencies up to 55 GHz
    and elevations above 10 degrees.

    Ported from KATcali (katcali/models.py:201-253).

    Parameters
    ----------
    temperature_C : np.ndarray
        Surface temperature in degrees Celsius, shape (n_time,)
    relative_humidity : np.ndarray
        Relative humidity as fraction (0 < RH < 1), shape (n_time,)
    pressure_hPa : np.ndarray
        Dry air pressure in hPa (equivalent to mbar), shape (n_time,)
    height_km : float
        Site elevation above sea level in kilometers (scalar)
    frequency_GHz : np.ndarray
        Frequency in GHz, shape (n_freq,)

    Returns
    -------
    opacity : np.ndarray
        Zenith atmospheric opacity in Nepers, shape (n_time, n_freq)

    Raises
    ------
    ValueError
        If any frequency exceeds 55 GHz (ITU-R P.676-9 validity limit)

    Notes
    -----
    The atmospheric opacity is used to calculate:
    - Transmission factor: exp(-opacity / sin(elevation))
    - Emission: T_atm * (1 - transmission_factor)

    For single-dish radio astronomy, atmospheric opacity at UHF/L-band
    frequencies is typically very small (< 0.01 Nepers), resulting in
    transmission factors close to 1 and small emission contributions.

    References
    ----------
    ITU-R Recommendation P.676-9: "Attenuation by atmospheric gases"
    """
    # Validate frequency range
    if np.any(frequency_GHz > 55.0):
        raise ValueError(
            f"Frequency exceeds ITU-R P.676-9 validity limit (55 GHz). "
            f"Max frequency: {frequency_GHz.max():.2f} GHz"
        )

    # Ensure proper shapes for broadcasting: (n_time, 1) and (1, n_freq)
    T = temperature_C[:, np.newaxis]  # (n_time, 1)
    RH = relative_humidity[:, np.newaxis]  # (n_time, 1)
    P = pressure_hPa[:, np.newaxis]  # (n_time, 1)
    h = height_km  # scalar
    f = frequency_GHz[np.newaxis, :]  # (1, n_freq)

    # Saturation vapor pressure [hPa] from A. L. Buck research manual 1996
    es = 6.1121 * np.exp((18.678 - T / 234.5) * T / (257.14 + T))

    # Water vapor density [g/m^3] from A. L. Buck research manual 1996
    # Note: ITU-R omitted the RH factor - this is a known error
    rho = RH * es * 216.7 / (T + 273.15)

    # Total pressure from ITU-R P.676-9 eq 3
    p_tot = P + es

    # Adjust water vapor density to sea level as per eq 32
    rho = rho * np.exp(h / 2)

    # ITU-R P.676-9 eq 22 - pressure and temperature ratios
    r_t = 288.0 / (273.0 + T)
    r_p = p_tot / 1013.0

    # Helper function for phi calculations
    def phi(a, b, c, d):
        return r_p**a * r_t**b * np.exp(c * (1 - r_p) + d * (1 - r_t))

    E_1 = phi(0.0717, -1.8132, 0.0156, -1.6515)
    E_2 = phi(0.5146, -4.6368, -0.1921, -5.7416)
    E_3 = phi(0.3414, -6.5851, 0.2130, -8.5854)

    # Dry air specific attenuation [dB/km] - valid only for f <= 54 GHz
    yo = (
        7.2 * r_t**2.8 / (f**2 + 0.34 * r_p**2 * r_t**1.6)
        + 0.62 * E_3 / ((54 - f) ** (1.16 * E_1) + 0.83 * E_2)
    ) * f**2 * r_p**2 * 1e-3

    # ITU-R P.676-9 eq 23 - water vapor parameters
    n_1 = 0.955 * r_p * r_t**0.68 + 0.006 * rho
    n_2 = 0.735 * r_p * r_t**0.5 + 0.0353 * r_t**4 * rho

    # Line shape factor
    def g(f, f_i):
        return 1 + (f - f_i) ** 2 / (f + f_i) ** 2

    # Water vapor specific attenuation [dB/km]
    yw = (
        3.98 * n_1 * np.exp(2.23 * (1 - r_t)) / ((f - 22.235) ** 2 + 9.42 * n_1**2) * g(f, 22)
        + 11.96 * n_1 * np.exp(0.7 * (1 - r_t)) / ((f - 183.31) ** 2 + 11.14 * n_1**2)
        + 0.081 * n_1 * np.exp(6.44 * (1 - r_t)) / ((f - 321.226) ** 2 + 6.29 * n_1**2)
        + 3.66 * n_1 * np.exp(1.6 * (1 - r_t)) / ((f - 325.153) ** 2 + 9.22 * n_1**2)
        + 25.37 * n_1 * np.exp(1.09 * (1 - r_t)) / (f - 380) ** 2
        + 17.4 * n_1 * np.exp(1.46 * (1 - r_t)) / (f - 448) ** 2
        + 844.6 * n_1 * np.exp(0.17 * (1 - r_t)) / (f - 557) ** 2 * g(f, 557)
        + 290 * n_1 * np.exp(0.41 * (1 - r_t)) / (f - 752) ** 2 * g(f, 752)
        + 8.3328e4 * n_2 * np.exp(0.99 * (1 - r_t)) / (f - 1780) ** 2 * g(f, 1780)
    ) * f**2 * r_t**2.5 * rho * 1e-4

    # ITU-R P.676-9 eq 25 - equivalent height for dry air
    t_1 = 4.64 / (1 + 0.066 * r_p**-2.3) * np.exp(-((f - 59.7) / (2.87 + 12.4 * np.exp(-7.9 * r_p))) ** 2)
    t_2 = 0.14 * np.exp(2.12 * r_p) / ((f - 118.75) ** 2 + 0.031 * np.exp(2.2 * r_p))
    t_3 = (
        0.0114 / (1 + 0.14 * r_p**-2.6)
        * f * (-0.0247 + 0.0001 * f + 1.61e-6 * f**2)
        / (1 - 0.0169 * f + 4.1e-5 * f**2 + 3.2e-7 * f**3)
    )
    ho = 6.1 / (1 + 0.17 * r_p**-1.1) * (1 + t_1 + t_2 + t_3)

    # ITU-R P.676-9 eq 26 - equivalent height for water vapor
    sigma_w = 1.013 / (1 + np.exp(-8.6 * (r_p - 0.57)))
    hw = 1.66 * (
        1
        + 1.39 * sigma_w / ((f - 22.235) ** 2 + 2.56 * sigma_w)
        + 3.37 * sigma_w / ((f - 183.31) ** 2 + 4.69 * sigma_w)
        + 1.58 * sigma_w / ((f - 325.1) ** 2 + 2.89 * sigma_w)
    )

    # Total attenuation [dB] from equations 27, 30, and 31
    # Attenuation relative to a point outside the atmosphere
    A = yo * ho * np.exp(-h / ho) + yw * hw * np.exp(-h / hw)

    # Convert from dB to Nepers
    opacity = A * np.log(10) / 10.0

    return opacity


class AtmosphericModel:
    """
    Atmospheric emission and transmission model for MeerKAT observations.

    This class computes atmospheric opacity, transmission, and emission
    temperature using the ITU-R P.676-9 standard. It takes TimeOrderedData
    as input and provides vectorized calculations across all times, frequencies,
    and antennas.

    Atmospheric emission temperature does not depend on polarization, so results
    are provided per antenna (not per receiver). Each receiver should use its
    corresponding antenna index to extract the appropriate values.

    Attributes
    ----------
    zenith_opacity : np.ndarray
        Zenith atmospheric opacity (Nepers), shape (n_time, n_freq)
    transmission_factor : np.ndarray
        Atmospheric transmission exp(-tau/sin(el)), shape (n_time, n_freq, n_antennas)
    emission_temperature : np.ndarray
        Atmospheric emission temperature (K), shape (n_time, n_freq, n_antennas)
    """

    def __init__(self, track_data):
        """
        Initialize atmospheric model from TimeOrderedData.

        Parameters
        ----------
        track_data : TimeOrderedData
            MuSEEK time ordered data containing weather sensor data
            (temperature, humidity, pressure), site elevation, pointing
            elevation angles, and frequency information.

        Raises
        ------
        ValueError
            If required attributes are missing from track_data
        AttributeError
            If track_data does not have required sensor data
        """
        # Validate required attributes
        required_attrs = ['temperature', 'humidity', 'pressure', 'site_elevation_km',
                          'elevation', 'frequencies']
        for attr in required_attrs:
            if not hasattr(track_data, attr) or getattr(track_data, attr) is None:
                raise ValueError(
                    f"TimeOrderedData missing required attribute: {attr}. "
                    f"Ensure data is loaded from katdal with sensor data."
                )

        # Extract and convert units
        # Temperature: DataElement (n_time, 1, 1) -> array (n_time,) in Celsius
        temperature_C = track_data.temperature.squeeze

        # Humidity: DataElement (n_time, 1, 1) stored as % -> array (n_time,) as fraction
        humidity_fraction = track_data.humidity.squeeze / 100.0

        # Pressure: DataElement (n_time, 1, 1) -> array (n_time,) in hPa
        pressure_hPa = track_data.pressure.squeeze

        # Site elevation: scalar in km
        site_elevation_km = track_data.site_elevation_km

        # Frequency: DataElement (1, n_freq, 1) in Hz -> array (n_freq,) in GHz
        frequency_GHz = track_data.frequencies.squeeze / 1e9

        # Elevation: DataElement (n_time, 1, n_antennas) -> array (n_time, n_antennas) in degrees
        elevation_deg = track_data.elevation.squeeze

        # Calculate zenith opacity using ITU-R P.676-9
        self._zenith_opacity = calc_zenith_opacity(
            temperature_C=temperature_C,
            relative_humidity=humidity_fraction,
            pressure_hPa=pressure_hPa,
            height_km=site_elevation_km,
            frequency_GHz=frequency_GHz
        )  # shape (n_time, n_freq)

        # Calculate atmospheric effective temperature
        # Empirical formula from KATcali (source unclear)
        T_atm = 1.12 * (273.15 + temperature_C) - 50.0  # shape (n_time,)
        self._T_atm = T_atm

        # Calculate transmission factor: exp(-tau / sin(elevation))
        # Broadcasting: tau (n_time, n_freq, 1) and el (n_time, 1, n_antennas)
        elevation_rad = np.radians(elevation_deg)  # (n_time, n_antennas)
        tau = self._zenith_opacity[:, :, np.newaxis]  # (n_time, n_freq, 1)
        sin_el = np.sin(elevation_rad[:, np.newaxis, :])  # (n_time, 1, n_antennas)

        self._transmission = np.exp(-tau / sin_el)  # (n_time, n_freq, n_antennas)

        # Calculate emission temperature: T_atm * (1 - transmission)
        # Broadcasting: T_atm (n_time, 1, 1) and transmission (n_time, n_freq, n_antennas)
        self._emission = T_atm[:, np.newaxis, np.newaxis] * (1 - self._transmission)

    @property
    def zenith_opacity(self) -> np.ndarray:
        """
        Zenith atmospheric opacity in Nepers.

        This is the opacity looking straight up (zenith). To get opacity
        at other elevations, use: tau(el) = tau_zenith / sin(el)

        Returns
        -------
        np.ndarray
            Zenith opacity (Nepers), shape (n_time, n_freq)
        """
        return self._zenith_opacity

    @property
    def transmission_factor(self) -> np.ndarray:
        """
        Atmospheric transmission factor: exp(-tau / sin(elevation)).

        This is the fraction of signal that passes through the atmosphere
        without being absorbed. Values close to 1 indicate transparent
        atmosphere, while values near 0 indicate strong absorption.

        Returns
        -------
        np.ndarray
            Transmission factor (dimensionless, 0-1), shape (n_time, n_freq, n_antennas)
        """
        return self._transmission

    @property
    def emission_temperature(self) -> np.ndarray:
        """
        Atmospheric emission temperature in Kelvin.

        This is the brightness temperature contribution from atmospheric
        emission along the line of sight. It represents the temperature
        added to the sky signal by the atmosphere.

        Returns
        -------
        np.ndarray
            Emission temperature (K), shape (n_time, n_freq, n_antennas)

        Notes
        -----
        For calibration purposes, each receiver extracts its antenna index:
            antenna_idx = receiver.antenna_index(track_data.receivers)
            T_atm_recv = emission_temperature[:, :, antenna_idx]
        """
        return self._emission
