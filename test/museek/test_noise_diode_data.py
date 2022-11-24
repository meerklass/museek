import unittest
from unittest.mock import MagicMock, patch

from museek.noise_diode_data import NoiseDiodeData


class TestNoiseDiodeData(unittest.TestCase):
    @patch('museek.noise_diode_data.NoiseDiode')
    @patch('museek.time_ordered_data.TimeOrderedDataElement')
    @patch('museek.noise_diode_data.TimeOrderedData._dumps_of_scan_state')
    @patch('museek.noise_diode_data.TimeOrderedData._correlator_products_indices')
    @patch('museek.noise_diode_data.TimeOrderedData.load_data')
    def test_init(self,
                  mock_load_data,
                  mock_correlator_products_indices,
                  mock_dumps_of_scan_state,
                  mock_time_ordered_data_element,
                  mock_noise_diode):
        mock_dumps_of_scan_state.return_value = [0]
        mock_noise_diode.return_value.get_noise_diode_off_scan_dumps.return_value = [0]
        noise_diode_data = NoiseDiodeData(block_name='block_name',
                                          receivers=MagicMock(),
                                          token=None,
                                          data_folder='data_folder')
        mock_noise_diode.assert_called_once_with(data=noise_diode_data)
        mock_noise_diode.return_value.get_noise_diode_off_scan_dumps.assert_called_once()
        self.assertEqual(0, noise_diode_data.scan_dumps[0])
        mock_load_data.assert_called_once()
        mock_correlator_products_indices.assert_called_once()
        self.assertEqual(10, mock_time_ordered_data_element.call_count)
