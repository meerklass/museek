import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord

from museek.data_element import DataElement
from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_element import FlagElement
from museek.receiver import Receiver


class FlagFactory:
    """ Class to instantiate `FlagElement`s, basically `DataElement`s with boolean entries. """

    def __init__(self):
        """ Initialise and set the `FlagElementFactory`. """
        self._data_element_factory = FlagElementFactory()

    def empty_flag(self, shape: tuple[int, int, int]) -> FlagElement:
        """ Returns an empty `FlagElement` of shape `shape`. """
        return self._data_element_factory.create(array=np.zeros(shape, dtype=bool))

    def from_list_of_receiver_flags(self, list_: list[FlagElement]) -> FlagElement:
        """ Combines all flags in `list_` to one single `FlagElement` and returns the result. """
        shape = (list_[0].shape[0], list_[0].shape[1], len(list_))
        result = self.empty_flag(shape=shape)
        for i_receiver, flag in enumerate(list_):
            result.insert_receiver_flag(flag=flag, i_receiver=i_receiver)
        return result

    @staticmethod
    def point_sources_coordinate_list(point_source_file_path: str) -> list[SkyCoord]:
        """
        Return a `list` of `SkyCoord` coordinates of point source data
        loaded from a file located at `point_source_file_path`.
        """
        point_sources = np.loadtxt(point_source_file_path)
        result = [SkyCoord(*(point_source * units.deg), frame='icrs') for point_source in point_sources]
        return result

    def get_point_source_mask(self,
                              shape: tuple[int, int, int],
                              receivers: list[Receiver],
                              right_ascension: DataElement,
                              declination: DataElement,
                              angle_threshold: float,
                              point_source_file_path: str) \
            -> FlagElement:
        """
        Return a `FlagElement` that is `True` wherever a dump is close enough to a point source.
        :param shape: the returned `FlagElement` will have this `shape`
        :param receivers: list of `Receiver`s to get the point source masks for
        :param right_ascension: celestial coordinate right ascension
        :param declination: celestial coordinate declination
        :param angle_threshold: all points up to this angular separation (degrees) are masked
        :param point_source_file_path: directory of the point source data
        :return: a `FlagElement` which is `True` for all masked pixels
        """
        point_source_mask = np.zeros(shape, dtype=bool)
        mask_points = FlagFactory.point_sources_coordinate_list(point_source_file_path=point_source_file_path)

        for receiver in receivers:
            i_receiver = receiver.antenna_index(receivers=receivers)
            point_source_mask_dump_list = self._coordinates_mask_dumps(
                right_ascension=right_ascension.get(recv=i_receiver),
                declination=declination.get(recv=i_receiver),
                mask_points=mask_points,
                angle_threshold=angle_threshold
            )
            point_source_mask[point_source_mask_dump_list, :, i_receiver] = True
        return self._data_element_factory.create(array=point_source_mask)

    @staticmethod
    def _coordinates_mask_dumps(right_ascension: DataElement,
                                declination: DataElement,
                                mask_points: list[SkyCoord],
                                angle_threshold: float) \
            -> list[int]:
        """
        Return a list of dump indices that are less than `angle_threshold` away from a point in `mask_points`.
        :param right_ascension: celestial coordinate right ascension
        :param declination: celestial coordinate declination
        :param mask_points: `list` of `SkyCoord` coordinates of the point sources
        :param angle_threshold: all points up to this angular separation (degrees) are masked
        :return: `list` of masked dump indices
        """
        result = []
        data_points = SkyCoord(right_ascension.squeeze * units.deg, declination.squeeze * units.deg, frame='icrs')

        for mask_coord in mask_points:
            separation = (mask_coord.separation(data_points) / units.deg)
            result.extend(np.where(separation < angle_threshold)[0])
        result = list(set(result))
        result.sort()
        return result
