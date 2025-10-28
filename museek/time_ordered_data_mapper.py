import numpy as np
from scipy.interpolate import griddata

from museek.data_element import DataElement
from museek.flag_list import FlagList
from museek.time_ordered_data import TimeOrderedData


class TimeOrderedDataMapper:
    """ Class to map time ordered data of one antenna to a celestial coordinate system ra-dec. """

    def __init__(self,
                 right_ascension: DataElement,
                 declination: DataElement,
                 to_map: DataElement,
                 flag_threshold: int = 1,
                 flags: FlagList | None = None):
        """
        Initialise
        :param right_ascension: celestial coordinate right ascension, any unit
        :param declination: celestial coordinate declination, any unit
        :param to_map: quantity to map
        :param flag_threshold: flags are only used if they overlap more than this value
        :param flags: optional flags to mask `to_map`
        """
        self._right_ascension = right_ascension
        self._declination = declination
        self._to_map = to_map
        if flags is not None:
            self._flags = flags.combine(threshold=flag_threshold)
            self._channel_iterator = DataElement.flagged_channel_iterator(data_element=self._to_map,
                                                                          flag_element=self._flags)
        else:
            self._flags = None
            self._channel_iterator = DataElement.channel_iterator(data_element=self._to_map)

    @classmethod
    def from_time_ordered_data(cls,
                               data: TimeOrderedData,
                               recv: int = 0,
                               flag_threshold: int = 1) -> 'TimeOrderedDataMapper':
        """
        Constructor using `TimeOrderedData` directly.
        :param data: time ordered data to map
        :param recv: receiver index to use, defaults to 0
        :param flag_threshold: flags are only used if they overlap more than this value
        """
        return cls(right_ascension=data.right_ascension.get(recv=recv),
                   declination=data.declination.get(recv=recv),
                   to_map=data.visibility.get(recv=recv),
                   flags=data.flags.get(recv=recv),
                   flag_threshold=flag_threshold)

    def grid(self,
             grid_size: tuple[int, int] = (60, 60),
             method: str = 'linear'
             ) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
        """
        Grid the data in bins in right ascension and declination and return a tuple of map and mask.
        If a all pixels of a map are flagged `None` is returned instead of a map array
        If no flags are given, `None` is returned instead of a mask
        :param grid_size: `tuple` of `integers` to specify the resolution in right ascension and declination
        :param method: interpolation method for `griddata`
        :return: `tuple` of gridded map and mask, both optional
        """
        right_ascension_i = np.linspace(self._right_ascension.squeeze.min(),
                                        self._right_ascension.squeeze.max(),
                                        grid_size[0])
        declination_i = np.linspace(self._declination.squeeze.min(), self._declination.squeeze.max(), grid_size[1])

        maps = [
            griddata((self._right_ascension.get(time=unmasked).squeeze, self._declination.get(time=unmasked).squeeze),
                     channel.get(time=unmasked).squeeze,
                     (right_ascension_i[np.newaxis, :], declination_i[:, np.newaxis]),
                     method=method)
            if unmasked.size > 0 else None for channel, unmasked in self._channel_iterator
        ]
        if self._flags:
            masks = [griddata(
                (self._right_ascension.squeeze, self._declination.squeeze),
                flag.squeeze,
                (right_ascension_i[np.newaxis, :], declination_i[:, np.newaxis]),
                method='nearest'
            )
                for flag, _ in DataElement.channel_iterator(data_element=self._flags)]
        else:
            masks = [None for _ in maps]
        return maps, masks
