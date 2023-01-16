import numpy as np
from scipy.interpolate import griddata

from museek.data_element import DataElement
from museek.flag_element import FlagElement
from museek.time_ordered_data import TimeOrderedData


class TimeOrderedDataMapper:
    """ Class to map time ordered data of one antenna to a celestial coordinate system ra-dec. """

    def __init__(self,
                 right_ascension: DataElement,
                 declination: DataElement,
                 to_map: DataElement,
                 flags: FlagElement | None = None):
        """
        Initialise
        :param right_ascension: celestial coordinate right ascension, any unit
        :param declination: celestial coordinate declination, any unit
        :param to_map: quantity to map
        :param flags: optional flags to mask `to_map`
        """
        self._right_ascension = right_ascension
        self._declination = declination
        self._to_map = to_map
        if flags is None:
            self._flags = []
        else:
            self._flags = flags

    @classmethod
    def from_time_ordered_data(cls, data: TimeOrderedData, recv: int = 0) -> 'TimeOrderedDataMapper':
        """
        Constructor using `TimeOrderedData` directly.
        :param data: time ordered data to map
        :param recv: receiver index to use, defaults to 0
        """
        return cls(right_ascension=data.right_ascension.get(recv=recv),
                   declination=data.declination.get(recv=recv),
                   to_map=data.visibility.get(recv=recv),
                   flags=data.flags.get(recv=recv))

    def grid(self, grid_size: tuple[int, int] = (60, 60), flag_threshold: int = 1, method: str = 'linear') \
            -> tuple[list[np.ndarray], list[np.ndarray | None]]:
        """
        Grid the data in bins in right ascension and declination
        :param grid_size: `tuple` of `integers` to specify the resolution in right ascension and declination
        :param flag_threshold: forwarded to `FlagElement`, combined flag entries less than this value are interpreted
                               as unmasked
        :param method: interpolation method for `griddata`
        :return: `tuple` of gridded map and the optional mask
        """
        right_ascension_i = np.linspace(self._right_ascension.min(), self._right_ascension.max(), grid_size[0])
        declination_i = np.linspace(self._declination.min(), self._declination.max(), grid_size[1])

        maps = [griddata((self._right_ascension.squeeze, self._declination.squeeze),
                         channel.squeeze,
                         (right_ascension_i[np.newaxis, :], declination_i[:, np.newaxis]),
                         method=method)
                for channel in DataElement.channel_iterator(data_element=self._to_map)]
        if self._flags:
            masks = [griddata(
                (self._right_ascension.squeeze, self._declination.squeeze),
                flag.squeeze,
                (right_ascension_i[None, :], declination_i[:, None]),
                method='linear'
            )
                for flag in DataElement.channel_iterator(data_element=self._flags.combine(threshold=flag_threshold))]
        else:
            masks = [None for _ in maps]
        return maps, masks
