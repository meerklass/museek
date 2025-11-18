import unittest
from unittest.mock import patch, Mock, MagicMock

import numpy as np

from museek.flag_factory import FlagFactory


class TestFlagFactory(unittest.TestCase):

    @patch("museek.flag_factory.np")
    @patch("museek.flag_factory.FlagElementFactory")
    def test_empty_flag(self, mock_flag_element_factory, mock_np):
        flag_factory = FlagFactory()
        mock_shape = Mock()
        empty_flag = flag_factory.empty_flag(shape=mock_shape)
        mock_np.zeros.assert_called_once_with(mock_shape, dtype=bool)
        mock_flag_element_factory.return_value.create.assert_called_once_with(
            array=mock_np.zeros.return_value
        )
        self.assertEqual(
            empty_flag, mock_flag_element_factory.return_value.create.return_value
        )

    @patch.object(FlagFactory, "empty_flag")
    def test_from_list_of_receiver_flags(self, mock_empty_flag):
        flag_factory = FlagFactory()
        mock_flag_list = [MagicMock()]
        flag = flag_factory.from_list_of_receiver_flags(list_=mock_flag_list)
        mock_empty_flag.return_value.insert_receiver_flag.assert_called_once_with(
            flag=mock_flag_list[0], i_receiver=0
        )
        self.assertEqual(flag, mock_empty_flag.return_value)

    @patch("museek.flag_factory.SkyCoord")
    @patch("museek.flag_factory.units")
    @patch.object(np, "loadtxt")
    def test_point_sources_coordinate_list(
        self, mock_loadtxt, mock_units, mock_sky_coord
    ):
        mock_point_source_file_path = "mock"
        mock_units.deg = 1
        mock_loadtxt.return_value = [[Mock(), Mock()]]
        point_sources_coordinate_list = FlagFactory.point_sources_coordinate_list(
            point_source_file_path=mock_point_source_file_path
        )
        mock_loadtxt.assert_called_once_with(mock_point_source_file_path)
        self.assertEqual(mock_sky_coord.return_value, point_sources_coordinate_list[0])
        mock_sky_coord.assert_called_once_with(
            *mock_loadtxt.return_value[0], frame="icrs"
        )

    @patch.object(FlagFactory, "_coordinates_mask_dumps")
    @patch.object(FlagFactory, "point_sources_coordinate_list")
    def test_get_point_source_mask(
        self, mock_point_sources_coordinate_list, mock_coordinates_mask_dumps
    ):
        shape = (3, 3, 3)
        mock_coordinates_mask_dumps.side_effect = [0, 1, 2]
        mock_point_source_file_path = Mock()
        mock_receiver = MagicMock(antenna_index=MagicMock(side_effect=[0, 1, 2]))
        point_source_mask = FlagFactory().get_point_source_mask(
            shape=shape,
            receivers=[mock_receiver, mock_receiver, mock_receiver],
            right_ascension=Mock(),
            declination=Mock(),
            angle_threshold=Mock(),
            point_source_file_path=mock_point_source_file_path,
        )
        np.testing.assert_array_equal(
            np.asarray(
                [[True, False, False], [True, False, False], [True, False, False]]
            ),
            point_source_mask[0],
        )
        np.testing.assert_array_equal(
            np.asarray(
                [[False, True, False], [False, True, False], [False, True, False]]
            ),
            point_source_mask[1],
        )
        np.testing.assert_array_equal(
            np.asarray(
                [[False, False, True], [False, False, True], [False, False, True]]
            ),
            point_source_mask[2],
        )
        mock_point_sources_coordinate_list.assert_called_once_with(
            point_source_file_path=mock_point_source_file_path
        )
        self.assertEqual(3, mock_coordinates_mask_dumps.call_count)

    @patch.object(np, "where")
    @patch("museek.flag_factory.units")
    @patch("museek.flag_factory.SkyCoord")
    def test_coordinates_mask_dumps(self, mock_sky_coord, mock_units, mock_where):
        mock_units.deg = 1
        mock_where.return_value = [[0]]
        mock_coordinate = MagicMock()
        mock_coordinate.separation.return_value.__truediv__.return_value = 2
        mock_mask_points = [mock_coordinate]
        mock_sky_coord.return_value = [MagicMock()]
        coordinates_mask_dumps = FlagFactory._coordinates_mask_dumps(
            right_ascension=MagicMock(),
            declination=MagicMock(),
            mask_points=mock_mask_points,
            angle_threshold=3,
        )
        self.assertListEqual([0], coordinates_mask_dumps)


if __name__ == "__main__":
    unittest.main()
