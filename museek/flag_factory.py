import os

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord

from museek.data_element import DataElement


class FlagFactory:
    """ Class to instantiate flags, i.e. `DataElement`s with boolean entries. """

    @staticmethod
    def point_sources_coordinate_list(point_sources_directory: str = None) -> list[SkyCoord]:
        """
        Return a `list` of `SkyCoord` coordinates of point source data
        loaded from a file located at `point_sources_directory`.
        """
        if point_sources_directory is None:  # TODO: remove the default path
            point_sources_directory = os.path.join(os.path.dirname(__file__), '../data/radio_point_sources.txt')
        point_sources = np.loadtxt(point_sources_directory)
        result = [SkyCoord(*(point_source * units.deg), frame='icrs') for point_source in point_sources]
        return result

    def get_point_source_mask(self,
                              shape: tuple[int, int, int],
                              right_ascension: DataElement,
                              declination: DataElement,
                              angle_threshold: float = 0.5,
                              point_sources_directory: str = None) \
            -> DataElement:
        """
        Return a `DataElement` that is `True` wherever a dump is close enough to a point source.
        :param shape: the returned `DataElement` will have this `shape`
        :param right_ascension: celestial coordinate right ascension
        :param declination: celestial coordinate declination
        :param angle_threshold: all points up to this angular separation (degrees) are masked
        :param point_sources_directory: directory of the point source data
        :return: a `DataElement` which is `True` for all masked pixels
        """
        mask_points = FlagFactory.point_sources_coordinate_list(point_sources_directory=point_sources_directory)
        point_source_mask_dump_list = self._coordinates_mask_dumps(right_ascension=right_ascension,
                                                                   declination=declination,
                                                                   mask_points=mask_points,
                                                                   angle_threshold=angle_threshold)
        point_source_mask = np.zeros(shape, dtype=bool)
        point_source_mask[point_source_mask_dump_list] = True
        return DataElement(array=point_source_mask)

    @staticmethod
    def _coordinates_mask_dumps(right_ascension: DataElement,
                                declination: DataElement,
                                mask_points: list[SkyCoord],
                                angle_threshold: float = 0.5) \
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
