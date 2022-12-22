import os
import unittest
from unittest.mock import patch, Mock, MagicMock

import numpy as np

from museek.flag_factory import FlagFactory


class TestFlagFactory(unittest.TestCase):

    @patch('museek.flag_factory.SkyCoord')
    @patch('museek.flag_factory.units')
    @patch.object(np, 'loadtxt')
    def test_point_sources_coordinate_list(self, mock_loadtxt, mock_units, mock_sky_coord):
        mock_point_sources_directory = 'mock'
        mock_units.deg = 1
        mock_loadtxt.return_value = [[Mock(), Mock()]]
        point_sources_coordinate_list = FlagFactory.point_sources_coordinate_list(
            point_sources_directory=mock_point_sources_directory
        )
        mock_loadtxt.assert_called_once_with(mock_point_sources_directory)
        self.assertEqual(mock_sky_coord.return_value, point_sources_coordinate_list[0])
        mock_sky_coord.assert_called_once_with(*mock_loadtxt.return_value[0], frame='icrs')

    @patch.object(os, 'path')
    @patch('museek.flag_factory.SkyCoord')
    @patch('museek.flag_factory.units')
    @patch.object(np, 'loadtxt')
    def test_point_sources_coordinate_list_when_point_sources_directory_none(self,
                                                                             mock_loadtxt,
                                                                             mock_units,
                                                                             mock_sky_coord,
                                                                             mock_path):
        mock_point_sources_directory = None
        mock_units.deg = 1
        mock_loadtxt.return_value = [[Mock(), Mock()]]
        point_sources_coordinate_list = FlagFactory.point_sources_coordinate_list(
            point_sources_directory=mock_point_sources_directory
        )
        mock_loadtxt.assert_called_once_with(mock_path.join.return_value)
        self.assertEqual(mock_sky_coord.return_value, point_sources_coordinate_list[0])
        mock_sky_coord.assert_called_once_with(*mock_loadtxt.return_value[0], frame='icrs')

    @patch.object(FlagFactory, '_coordinates_mask_dumps')
    @patch.object(FlagFactory, 'point_sources_coordinate_list')
    def test_get_point_source_mask(self, mock_point_sources_coordinate_list, mock_coordinates_mask_dumps):
        shape = (3, 3, 3)
        mock_coordinates_mask_dumps.return_value = [1]
        mock_point_sources_directory = Mock()
        point_source_mask = FlagFactory().get_point_source_mask(shape=shape,
                                                                right_ascension=Mock(),
                                                                declination=Mock(),
                                                                angle_threshold=Mock(),
                                                                point_sources_directory=mock_point_sources_directory)
        np.testing.assert_array_equal(np.zeros((3, 3), dtype=bool), point_source_mask[0])
        np.testing.assert_array_equal(np.ones((3, 3), dtype=bool), point_source_mask[1])
        np.testing.assert_array_equal(np.zeros((3, 3), dtype=bool), point_source_mask[2])
        mock_point_sources_coordinate_list.assert_called_once_with(
            point_sources_directory=mock_point_sources_directory
        )

    @patch.object(np, 'where')
    @patch('museek.flag_factory.units')
    @patch('museek.flag_factory.SkyCoord')
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
            angle_threshold=3
        )
        self.assertListEqual([0], coordinates_mask_dumps)
