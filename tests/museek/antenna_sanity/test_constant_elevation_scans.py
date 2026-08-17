import unittest
from unittest.mock import MagicMock, Mock, PropertyMock

from museek.antenna_sanity.constant_elevation_scans import ConstantElevationScans


class TestConstantElevationScans(unittest.TestCase):
    def test_get_antennas_with_non_constant_elevation(self):
        mock_data = MagicMock()
        mock_antenna_1 = Mock()
        mock_antenna_2 = Mock()
        mock_antenna_3 = Mock()
        mock_data.antennas = [mock_antenna_1, mock_antenna_2, mock_antenna_3]

        type(mock_data.elevation.get.return_value).squeeze = PropertyMock(
            side_effect=[[1, 2, 3], [1, 1.1, 0.9], [0, 0, 0.1]]
        )

        antennas = ConstantElevationScans.get_antennas_with_non_constant_elevation(
            data=mock_data, threshold=0.1
        )
        self.assertListEqual([mock_antenna_1], antennas)
