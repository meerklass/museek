import numpy as np
from katpoint import Antenna

from museek.time_ordered_data import TimeOrderedData


class ConstantElevationScans:
    """
    Class for antenna sanity checks in a scanning strategy with constant elevation pointings but motion in azimuth.
    """

    @staticmethod
    def get_antennas_with_non_constant_elevation(
        data: TimeOrderedData, threshold: float
    ) -> list[Antenna]:
        """
        Return a `list` of `Antenna`s which do not have constant elevation individually`.
        :param data: time ordered data
        :param threshold: `float` threshold on the elevation standard deviation
        :return: `list` of `Antenna`s with non-constant elevation
        """
        result: list[Antenna] = []
        for i_antenna, antenna in enumerate(data.antennas):
            antenna_elevation = data.elevation.get(recv=i_antenna).squeeze
            standard_deviation = np.std(antenna_elevation)
            if standard_deviation > threshold:
                result.append(antenna)
        return result
