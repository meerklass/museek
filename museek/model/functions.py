import numpy as np

from definitions import KILO, GIGA


def atmospheric_opacity(temperature: np.ndarray[float] | float,
                        relative_humidity: np.ndarray[float] | float,
                        pressure: np.ndarray[float] | float,
                        height: np.ndarray[float] | float,
                        frequency: np.ndarray[float] | float) -> np.ndarray[float] | float:
    """
    Calculates zenith opacity according to ITU-R P.676-9. For elevations > 10 deg. 10 years outdated.
    Use as "Tsky*(1-exp(-opacity/sin(elevation)))" for elevation dependence.
    :param temperature: temperature in deg C
    :param relative_humidity: relative humidity, 0 < RH < 1
    :param pressure: dry air pressure in hPa (equiv. mbar)
    :param height: height above sea level in m
    :param frequency: frequency in Hz (must be < 55 GHz)
    :return: approximate atmospheric opacity at zenith [Nepers]
    """
    frequency = frequency / GIGA  # Hz to GHz
    if (max_f := np.max(frequency)) > 55:
        raise ValueError(f'Frequency must not be higher than 55 GHz for this atmosphere model, got {max_f}')
    height = height / KILO  # meters to kilometers
    temperature = np.asarray(temperature, dtype=np.float128)
    # [hPa] from A. L. Buck research manual 1996
    partial_pressure_water_vapour = 6.1121 * np.exp(
        (18.678 - temperature / 234.5) * temperature / (257.14 + temperature)
    )
    # [g/m^3] from A. L. Buck research manual 1996 (ITU-R ommited the factor "relative_humidity" - a mistake)
    water_vapor_density_local = relative_humidity * partial_pressure_water_vapour * 216.7 / (temperature + 273.15)

    # The following is taken directly from ITU-R P.676-9
    pressure_total = pressure + partial_pressure_water_vapour  # below eq 3

    water_vapor_density_sea_level = water_vapor_density_local * np.exp(height / 2)  # eq 32

    # eq 22
    r_temperature = 288. / (273. + temperature)
    r_pressure = pressure_total / 1013.

    def phi(a, b, c, d):
        return r_pressure ** a * r_temperature ** b * np.exp(c * (1 - r_pressure) + d * (1 - r_temperature))

    xi_1 = phi(0.0717, -1.8132, 0.0156, -1.6515)
    xi_2 = phi(0.5146, -4.6368, -0.1921, -5.7416)
    xi_3 = phi(0.3414, -6.5851, 0.2130, -8.5854)
    # Following is valid only for f <= 54 GHz
    attenuation_dry_air = frequency ** 2 * r_pressure ** 2 * 1e-3 * (
            7.2 * r_temperature ** 2.8 / (frequency ** 2 + 0.34 * r_pressure ** 2 * r_temperature ** 1.6)
            + 0.62 * xi_3 / ((54 - frequency) ** (1.16 * xi_1) + 0.83 * xi_2)
    )
    # eq 23
    eta_1 = 0.955 * r_pressure * r_temperature ** 0.68 + 0.006 * water_vapor_density_sea_level
    eta_2 = 0.735 * r_pressure * r_temperature ** 0.5 + 0.0353 * r_temperature ** 4 * water_vapor_density_sea_level

    def g(a, b):
        return 1 + (a - b) ** 2 / (a + b) ** 2

    attenuation_water_vapour = \
        frequency ** 2 * r_temperature ** 2.5 * water_vapor_density_sea_level * 1e-4 * (
                3.98 * eta_1 * np.exp(2.23 * (1 - r_temperature)) / ((frequency - 22.235) ** 2 + 9.42 * eta_1 ** 2)
                * g(frequency, 22)
                + 11.96 * eta_1 * np.exp(0.7 * (1 - r_temperature)) / ((frequency - 183.31) ** 2 + 11.14 * eta_1 ** 2)
                + 0.081 * eta_1 * np.exp(6.44 * (1 - r_temperature)) / ((frequency - 321.226) ** 2 + 6.29 * eta_1 ** 2)
                + 3.66 * eta_1 * np.exp(1.6 * (1 - r_temperature)) / ((frequency - 325.153) ** 2 + 9.22 * eta_1 ** 2)
                + 25.37 * eta_1 * np.exp(1.09 * (1 - r_temperature)) / (frequency - 380) ** 2
                + 17.4 * eta_1 * np.exp(1.46 * (1 - r_temperature)) / (frequency - 448) ** 2
                + 844.6 * eta_1 * np.exp(0.17 * (1 - r_temperature)) / (frequency - 557) ** 2 * g(frequency, 557)
                + 290 * eta_1 * np.exp(0.41 * (1 - r_temperature)) / (frequency - 752) ** 2 * g(frequency, 752)
                + 8.3328e4 * eta_2 * np.exp(0.99 * (1 - r_temperature)) / (frequency - 1780) ** 2 * g(frequency, 1780)
        )

    # eq 25
    t_1 = 4.64 / (1 + 0.066 * r_pressure ** -2.3) * np.exp(
        -((frequency - 59.7) / (2.87 + 12.4 * np.exp(-7.9 * r_pressure))) ** 2)
    t_2_denominator = ((frequency - 118.75) ** 2 + 0.031 * np.exp(2.2 * r_pressure))
    t_2 = 0.14 * np.exp(2.12 * r_pressure) / t_2_denominator
    t_3 = 0.0114 \
          / (1 + 0.14 * r_pressure ** -2.6) \
          * frequency \
          * (-0.0247 + 0.0001 * frequency + 1.61e-6 * frequency ** 2) \
          / (1 - 0.0169 * frequency + 4.1e-5 * frequency ** 2 + 3.2e-7 * frequency ** 3)
    equivalent_height_dry_air = 6.1 / (1 + 0.17 * r_pressure ** -1.1) * (1 + t_1 + t_2 + t_3)

    # eq 26
    sigma_water_vapour = 1.013 / (1 + np.exp(-8.6 * (r_pressure - 0.57)))
    equivalent_height_water_vapour = 1.66 * (
            1
            + 1.39 * sigma_water_vapour / ((frequency - 22.235) ** 2 + 2.56 * sigma_water_vapour)
            + 3.37 * sigma_water_vapour / ((frequency - 183.31) ** 2 + 4.69 * sigma_water_vapour)
            + 1.58 * sigma_water_vapour / ((frequency - 325.1) ** 2 + 2.89 * sigma_water_vapour)
    )

    # Attenuation from dry & wet atmosphere relative to a point outside of the atmosphere
    dry_air_part = attenuation_dry_air * equivalent_height_dry_air * np.exp(-height / equivalent_height_dry_air)
    water_vapour_part = attenuation_water_vapour * equivalent_height_water_vapour * np.exp(
        -height / equivalent_height_water_vapour
    )
    total_attenuation = dry_air_part + water_vapour_part  # [dB] from equations 27, 30 & 31

    return total_attenuation * np.log(10) / 10.0  # Convert dB to Nepers
