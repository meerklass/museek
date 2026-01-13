import unittest
from unittest.mock import patch, Mock, MagicMock

from museek.factory.data_element_factory import (
    DataElementFactory,
    ScanElementFactory,
    FlagElementFactory,
)


class TestDataElementFactory(unittest.TestCase):
    @patch("museek.factory.data_element_factory.DataElement")
    def test_create(self, mock_data_element):
        mock_array = Mock()
        factory = DataElementFactory().create(array=mock_array)
        self.assertEqual(factory, mock_data_element.return_value)


class TestFlagElementFactory(unittest.TestCase):
    @patch("museek.factory.data_element_factory.FlagElement")
    def test_create(self, mock_flag_element):
        mock_array = Mock()
        factory = FlagElementFactory().create(array=mock_array)
        self.assertEqual(factory, mock_flag_element.return_value)


class TestScanDataElementFactory(unittest.TestCase):
    @patch("museek.factory.data_element_factory.DataElement")
    def test_create(self, mock_data_element):
        mock_scan_dumps = Mock()
        mock_array = MagicMock()
        mock_array.shape = (3, 3, 3)
        mock_array.__getitem__.return_value = mock_array
        factory = ScanElementFactory(
            scan_dumps=mock_scan_dumps, component=DataElementFactory()
        ).create(array=mock_array)
        self.assertEqual(factory, mock_data_element.return_value)
        mock_array.__getitem__.assert_called_once_with(mock_scan_dumps)


if __name__ == "__main__":
    unittest.main()
