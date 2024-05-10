from museek.data_element import DataElement
import numpy as np


class Swings:
    """ Util class relating to scanning swings. """

    @staticmethod
    def swing_turnaround_dumps(azimuth: DataElement) -> list[int]:
        """ Time dumps for the `azimuth` turnaround moments of the scan. """
        sign = np.sign(np.diff(azimuth.squeeze))
        sign_change = ((np.roll(sign, 1) - sign) != 0).astype(bool)
        return np.where(sign_change)[0]
