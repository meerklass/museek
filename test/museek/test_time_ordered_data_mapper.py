import unittest
from unittest.mock import MagicMock

import numpy as np

from museek.data_element import DataElement
from museek.time_ordered_data_mapper import TimeOrderedDataMapper


class TestTimeOrderedDataMapper(unittest.TestCase):
    def test_from_time_ordered_data(self):
        mock_data = MagicMock()
        self.assertIsInstance(TimeOrderedDataMapper.from_time_ordered_data(data=mock_data, recv=1),
                              TimeOrderedDataMapper)
        mock_data.right_ascension.get.assert_called_once_with(recv=1)
        mock_data.declination.get.assert_called_once_with(recv=1)
        mock_data.visibility.get.assert_called_once_with(recv=1)
        mock_data.flags.get.assert_called_once_with(recv=1)

    def test_grid(self):
        n_dump = 10

        right_ascension_array = np.linspace(0, 90, n_dump)[:, np.newaxis, np.newaxis]
        right_ascension = DataElement(array=right_ascension_array)

        declination_array = np.append(np.linspace(180, 360, n_dump // 2)[::-1],
                                      np.linspace(180, 360, n_dump // 2))[::, np.newaxis, np.newaxis]
        declination = DataElement(array=declination_array)

        to_map_array = np.linspace(0, 1, n_dump)[:, np.newaxis, np.newaxis]
        to_map = DataElement(array=to_map_array)

        time_ordered_data_mapper = TimeOrderedDataMapper(
            right_ascension=right_ascension,
            declination=declination,
            to_map=to_map
        )
        maps, _ = time_ordered_data_mapper.grid(grid_size=(n_dump // 2, n_dump // 2), method='linear')

        self.assertEqual(0, np.nanmin(maps[0]))
        self.assertEqual(1, np.nanmax(maps[0]))
        self.assertTupleEqual((n_dump // 2, n_dump // 2), maps[0].shape)
