import unittest
from unittest.mock import MagicMock, patch

from museek.enum.scan_state_enum import ScanStateEnum
from museek.noise_diode_data import NoiseDiodeData


class TestNoiseDiodeData(unittest.TestCase):
    @patch('museek.noise_diode_data.NoiseDiode')
    @patch('museek.noise_diode_data.TimeOrderedData._dumps_of_scan_state')
    @patch('museek.noise_diode_data.TimeOrderedData._correlator_products_indices')
    @patch('museek.noise_diode_data.TimeOrderedData._get_data')
    @patch('museek.noise_diode_data.TimeOrderedData.set_data_elements')
    def test_init(self,
                  mock_set_data_elements,
                  mock_get_data,
                  mock_correlator_products_indices,
                  mock_dumps_of_scan_state,
                  mock_noise_diode):
        mock_dumps_of_scan_state.return_value = [0]
        NoiseDiodeData(block_name='block_name',
                       receivers=MagicMock(),
                       token=None,
                       scan_state=ScanStateEnum.SCAN,
                       data_folder='data_folder')
        mock_noise_diode.assert_called_once_with(dump_period=mock_get_data.return_value.dump_period,
                                                 observation_log=mock_get_data.return_value.obs_script_log)
        mock_get_data.assert_called_once()
        mock_correlator_products_indices.assert_called_once()
        mock_set_data_elements.assert_called_once()

    @patch.object(NoiseDiodeData, '_dumps_of_scan_state')
    @patch('museek.noise_diode_data.NoiseDiode')
    @patch('museek.noise_diode_data.TimeOrderedData.set_data_elements')
    @patch('museek.noise_diode_data.TimeOrderedData._correlator_products_indices')
    @patch('museek.noise_diode_data.TimeOrderedData._get_data')
    def test_dumps(self,
                   mock_get_data,
                   mock_correlator_products_indices,
                   mock_set_data_elements,
                   mock_noise_diode,
                   mock_dumps_of_scan_state):
        noise_diode_data = NoiseDiodeData(block_name='block_name',
                                          receivers=MagicMock(),
                                          token=None,
                                          scan_state=ScanStateEnum.SCAN,
                                          data_folder='data_folder')
        noise_diode_data.scan_state = ScanStateEnum.SCAN
        mock_dumps_of_scan_state.return_value = [0, 1, 2, 3, 4]
        mock_noise_diode_off_scan_dumps = [1, 3, 5]
        mock_noise_diode.return_value.get_noise_diode_off_scan_dumps.return_value = mock_noise_diode_off_scan_dumps
        self.assertListEqual([1, 3], noise_diode_data._dumps())
        mock_set_data_elements.assert_called_once()
        mock_correlator_products_indices.assert_called_once()
        mock_get_data.assert_called_once()

    @patch.object(NoiseDiodeData, '_dumps_of_scan_state')
    @patch('museek.noise_diode_data.NoiseDiode')
    @patch('museek.noise_diode_data.TimeOrderedData.set_data_elements')
    @patch('museek.noise_diode_data.TimeOrderedData._correlator_products_indices')
    @patch('museek.noise_diode_data.TimeOrderedData._get_data')
    def test_dumps_when_not_scan(self,
                                 mock_get_data,
                                 mock_correlator_products_indices,
                                 mock_set_data_elements,
                                 mock_noise_diode,
                                 mock_dumps_of_scan_state):
        noise_diode_data = NoiseDiodeData(block_name='block_name',
                                          receivers=MagicMock(),
                                          token=None,
                                          data_folder='data_folder')
        self.assertEqual(mock_dumps_of_scan_state.return_value, noise_diode_data._dumps())
        mock_set_data_elements.assert_called_once()
        mock_correlator_products_indices.assert_called_once()
        mock_get_data.assert_called_once()
        mock_noise_diode.assert_called_once()
