import unittest
from unittest.mock import patch, Mock, MagicMock, call

import numpy as np

from museek.receiver import Receiver, Polarisation
from museek.time_ordered_data import TimeOrderedData, ScanStateEnum, ScanTuple


class TestTimeOrderedData(unittest.TestCase):

    @patch.object(TimeOrderedData, '_get_data_element_factory')
    @patch.object(TimeOrderedData, '_correlator_products_indices')
    @patch.object(TimeOrderedData, '_get_data')
    def setUp(self,
              mock_get_data,
              mock_correlator_products_indices,
              mock_get_data_element_factory):
        self.mock_katdal_data = MagicMock()
        self.mock_correlator_products_indices = mock_correlator_products_indices
        self.mock_get_data_element_factory = mock_get_data_element_factory
        mock_get_data.return_value = self.mock_katdal_data
        mock_block_name = Mock()
        mock_receiver_list = [Mock(), Mock()]
        mock_data_folder = Mock()

        self.time_ordered_data = TimeOrderedData(block_name=mock_block_name,
                                                 receivers=mock_receiver_list,
                                                 token=None,
                                                 data_folder=mock_data_folder)

    def test_init(self):
        self.mock_katdal_data.select.assert_called_once_with(
            corrprods=self.mock_correlator_products_indices.return_value
        )

    def test_str(self):
        expect = str(self.mock_katdal_data)
        self.assertEqual(str(self.time_ordered_data), expect)

    def test_set_data_elements(self):
        mock_scan_state = Mock()
        mock_data = MagicMock()
        self.time_ordered_data.set_data_elements(scan_state=mock_scan_state, data=mock_data)
        self.assertEqual(self.time_ordered_data.scan_state, mock_scan_state)
        self.assertEqual(mock_scan_state.factory(), self.time_ordered_data._element_factory)
        expect = mock_scan_state.factory().create.return_value
        self.assertEqual(expect, self.time_ordered_data.timestamps)
        self.assertEqual(expect, self.time_ordered_data.timestamp_dates)
        self.assertEqual(expect, self.time_ordered_data.frequencies)
        self.assertEqual(expect, self.time_ordered_data.azimuth)
        self.assertEqual(expect, self.time_ordered_data.elevation)
        self.assertEqual(expect, self.time_ordered_data.declination)
        self.assertEqual(expect, self.time_ordered_data.right_ascension)
        self.assertEqual(expect, self.time_ordered_data.temperature)
        self.assertEqual(expect, self.time_ordered_data.humidity)
        self.assertEqual(expect, self.time_ordered_data.pressure)

    @patch.object(TimeOrderedData, '_select')
    @patch.object(TimeOrderedData, '_get_data')
    def test_set_data_elements_when_data_is_none(self, mock_get_data, mock_select):
        mock_scan_state = Mock()
        self.time_ordered_data.set_data_elements(scan_state=mock_scan_state, data=None)
        self.assertEqual(self.time_ordered_data.scan_state, mock_scan_state)
        mock_get_data.assert_called_once()
        mock_select.assert_called_once_with(data=mock_get_data.return_value)

    @patch('museek.time_ordered_data.FlagElement')
    @patch.object(TimeOrderedData, '_visibility_flags_weights')
    def test_load_visibility_flag_weights(self, mock_visibility_flags_weights, mock_flag_element):
        mock_visibility_flags_weights.return_value = (Mock(), Mock(), Mock())
        self.time_ordered_data.load_visibility_flags_weights()
        mock_flag_element.assert_called_once_with(
            flags=[self.mock_get_data_element_factory.return_value.create.return_value]
        )
        self.assertEqual(self.time_ordered_data.visibility,
                         self.mock_get_data_element_factory.return_value.create.return_value)
        self.assertEqual(self.time_ordered_data.flags, mock_flag_element.return_value)
        self.assertEqual(self.time_ordered_data.weights,
                         self.mock_get_data_element_factory.return_value.create.return_value)

    @patch('museek.time_ordered_data.FlagElement')
    @patch.object(TimeOrderedData, '_visibility_flags_weights')
    def test_delete_visibility_flags_weights(self, mock_visibility_flags_weights, mock_flag_element):
        mock_visibility_flags_weights.return_value = (Mock(), Mock(), Mock())
        self.time_ordered_data.load_visibility_flags_weights()
        mock_flag_element.assert_called_once_with(
            flags=[self.mock_get_data_element_factory.return_value.create.return_value]
        )
        self.assertEqual(self.time_ordered_data.visibility,
                         self.mock_get_data_element_factory.return_value.create.return_value)
        self.assertEqual(self.time_ordered_data.flags, mock_flag_element.return_value)
        self.assertEqual(self.time_ordered_data.weights,
                         self.mock_get_data_element_factory.return_value.create.return_value)
        self.time_ordered_data.delete_visibility_flags_weights()
        self.assertIsNone(self.time_ordered_data.visibility)
        self.assertIsNone(self.time_ordered_data.flags)
        self.assertIsNone(self.time_ordered_data.weights)

    def test_antenna(self):
        mock_receiver = MagicMock()
        mock_antenna_name_list = MagicMock()
        self.time_ordered_data._antenna_name_list = mock_antenna_name_list
        antenna = self.time_ordered_data.antenna(receiver=mock_receiver)
        mock_antenna_name_list.index.assert_called_once_with(mock_receiver.antenna_name)
        self.assertEqual(self.time_ordered_data.antennas.__getitem__.return_value, antenna)

    def test_antenna_when_explicit(self):
        self.time_ordered_data._antenna_name_list = ['m000', 'm001']
        self.time_ordered_data.antennas = ['antenna0', 'antenna1']
        antenna = self.time_ordered_data.antenna(receiver=Receiver(antenna_number=1, polarisation=Polarisation.v))
        self.assertEqual('antenna1', antenna)

    @patch.object(TimeOrderedData, 'antenna')
    def test_antenna_index_of_receiver(self, mock_antenna):
        self.time_ordered_data.antennas = [MagicMock(), mock_antenna.return_value, MagicMock()]
        antenna_index = self.time_ordered_data.antenna_index_of_receiver(receiver=MagicMock())
        self.assertEqual(1, antenna_index)

    @patch.object(TimeOrderedData, 'antenna')
    def test_antenna_index_of_receiver_when_not_there(self, mock_antenna):
        self.time_ordered_data.antennas = [MagicMock(), MagicMock()]
        antenna_index = self.time_ordered_data.antenna_index_of_receiver(receiver=MagicMock())
        mock_antenna.assert_called_once()
        self.assertIsNone(antenna_index)

    @patch.object(TimeOrderedData, 'set_data_elements')
    @patch.object(TimeOrderedData, '_select')
    @patch('museek.time_ordered_data.katdal.open')
    def test_load_data_when_data_folder(self, mock_open, mock_select, mock_set_data_elements):
        block_name = 'block'
        token = None
        data_folder = 'folder'
        mock_receiver_list = [Mock(), Mock()]

        TimeOrderedData(block_name=block_name,
                        receivers=mock_receiver_list,
                        token=token,
                        data_folder=data_folder)
        mock_open.assert_called_once_with(f'{data_folder}/{block_name}/{block_name}/{block_name}_sdp_l0.full.rdb')
        mock_select.assert_called_once()
        mock_set_data_elements.assert_called_once()

    @patch.object(TimeOrderedData, 'set_data_elements')
    @patch.object(TimeOrderedData, '_select')
    @patch('museek.time_ordered_data.katdal.open')
    def test_load_data_when_token(self, mock_open, mock_select, mock_set_data_elements):
        block_name = 'block'
        token = 'token'
        mock_receiver_list = [Mock(), Mock()]

        TimeOrderedData(block_name=block_name,
                        receivers=mock_receiver_list,
                        token=token,
                        data_folder=None)
        mock_open.assert_called_once_with(
            f'https://archive-gw-1.kat.ac.za/{block_name}/{block_name}_sdp_l0.full.rdb?{token}'
        )
        mock_select.assert_called_once()
        mock_set_data_elements.assert_called_once()

    def test_dumps_of_scan_state(self):
        mock_scan_state = Mock(state='2')
        mock_scan_tuple_list = [Mock(state='mock'),
                                Mock(state=mock_scan_state, dumps=[Mock()]),
                                Mock(state='mock')]
        self.time_ordered_data._scan_tuple_list = mock_scan_tuple_list
        self.time_ordered_data.scan_state = mock_scan_state
        dumps = self.time_ordered_data._dumps_of_scan_state()
        expect = mock_scan_tuple_list[1].dumps
        self.assertListEqual(expect, dumps)

    def test_dumps_of_scan_state_when_scan_state_is_none(self):
        self.assertIsNone(self.time_ordered_data._dumps_of_scan_state())

    @patch.object(TimeOrderedData, '_dumps_of_scan_state')
    def test_dumps(self, mock_dumps_of_scan_state):
        self.time_ordered_data._dumps()
        mock_dumps_of_scan_state.assert_called_once()

    @patch('museek.time_ordered_data.DataElementFactory')
    def test_get_data_element_factory(self, mock_factory):
        mock_scan_state = Mock()
        self.time_ordered_data.scan_state = mock_scan_state
        self.assertEqual(mock_scan_state.factory(), self.time_ordered_data._get_data_element_factory())

    @patch('museek.time_ordered_data.DataElementFactory')
    def test_get_data_element_factory_when_scan_state_is_none(self, mock_factory):
        self.assertEqual(mock_factory(), self.time_ordered_data._get_data_element_factory())

    @patch.object(TimeOrderedData, '_select')
    @patch('museek.time_ordered_data.np')
    @patch('museek.time_ordered_data.katdal')
    @patch.object(TimeOrderedData, '_load_autocorrelation_visibility')
    def test_visibility_flags_weights_when_force_load_from_correlator_data(
            self,
            mock_load_autocorrelation_visibility,
            mock_katdal,
            mock_np,
            mock_select
    ):
        self.time_ordered_data._force_load_from_correlator_data = True
        mock_load_autocorrelation_visibility.return_value = (Mock(), Mock(), Mock())
        visibility, flags, weights = self.time_ordered_data._visibility_flags_weights()
        mock_katdal.open.assert_called_once()
        mock_np.savez_compressed.assert_called_once()
        mock_select.assert_called_once()
        self.assertEqual(mock_load_autocorrelation_visibility.return_value[0].real, visibility)
        self.assertEqual(mock_load_autocorrelation_visibility.return_value[1], flags)
        self.assertEqual(mock_load_autocorrelation_visibility.return_value[2], weights)

    @patch('museek.time_ordered_data.DaskLazyIndexer')
    def test_load_autocorrelation_visibility(self, mock_dask_lazy_indexer):
        self.time_ordered_data.shape = (1, 1, 1)
        visibility, flags, weights = self.time_ordered_data._load_autocorrelation_visibility(
            data=self.mock_katdal_data
        )
        mock_dask_lazy_indexer.get.assert_called_once()
        np.testing.assert_array_equal(np.asarray([[[0. + 0.j]]]), visibility)
        np.testing.assert_array_equal(np.asarray([[[0]]]), flags)
        np.testing.assert_array_equal(np.asarray([[[0]]]), weights)

    def test_correlator_products_indices(self):
        all_correlator_products = np.asarray([('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')])
        self.time_ordered_data.correlator_products = np.asarray([('c', 'c'), ('a', 'a')])
        expect = [2, 0]
        indices = self.time_ordered_data._correlator_products_indices(all_correlator_products=all_correlator_products)
        self.assertListEqual(expect, indices)

    def test_correlator_products_indices_when_all_missing_expect_raise(self):
        all_correlator_products = np.asarray([('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')])
        self.time_ordered_data.correlator_products = np.asarray([('e', 'e'), ('f', 'f')])
        self.assertRaises(ValueError,
                          self.time_ordered_data._correlator_products_indices,
                          all_correlator_products=all_correlator_products)

    def test_correlator_products_indices_when_one_missing_expect_raise(self):
        all_correlator_products = np.asarray([('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')])
        self.time_ordered_data.correlator_products = np.asarray([('a', 'a'), ('f', 'f')])
        self.assertRaises(ValueError,
                          self.time_ordered_data._correlator_products_indices,
                          all_correlator_products=all_correlator_products)

    def test_get_correlator_products(self):
        self.time_ordered_data.receivers = [Receiver(antenna_number=0, polarisation=Polarisation.v),
                                            Receiver(antenna_number=0, polarisation=Polarisation.h),
                                            Receiver(antenna_number=200, polarisation=Polarisation.h)]
        expect_list = [['m000v', 'm000v'], ['m000h', 'm000h'], ['m200h', 'm200h']]
        for expect, correlator_product in zip(expect_list, self.time_ordered_data._get_correlator_products()):
            self.assertListEqual(expect, correlator_product)

    @patch.object(TimeOrderedData, '_correlator_products_indices')
    def test_select(self, mock_correlator_products_indices):
        self.time_ordered_data._select(data=self.mock_katdal_data)
        self.mock_katdal_data.select.assert_has_calls(
            calls=[call(corrprods=self.mock_correlator_products_indices.return_value),
                   call(corrprods=mock_correlator_products_indices.return_value)])
        mock_correlator_products_indices.assert_called_once_with(
            all_correlator_products=self.mock_katdal_data.corr_products
        )

    def test_get_receivers_if_receivers_given(self):
        mock_receivers = [MagicMock()]
        self.assertEqual(mock_receivers, self.time_ordered_data._get_receivers(receivers=mock_receivers,
                                                                               data=MagicMock()))

    @patch.object(Receiver, 'from_string')
    def test_get_receivers_if_receivers_none(self, mock_from_string):
        mock_data = MagicMock()
        mock_data.corr_products = np.array([['1', '2'], ['2', '1'], ['1', '3']])
        expect = [mock_from_string()] * 3
        self.assertListEqual(expect, self.time_ordered_data._get_receivers(receivers=None,
                                                                           data=mock_data))
        mock_from_string.assert_has_calls(calls=[call(receiver_string='1'),
                                                 call(receiver_string='2'),
                                                 call(receiver_string='3')])

    def test_get_scan_tuple_list(self):
        mock_target = Mock()
        mock_scans = MagicMock(return_value=[(0, 'scan', mock_target),
                                             (1, 'track', mock_target)])
        mock_data = MagicMock(scans=mock_scans)
        scan_tuple_list = self.time_ordered_data._get_scan_tuple_list(data=mock_data)
        expect_list = [ScanTuple(dumps=mock_data.dumps, state=ScanStateEnum.SCAN, index=0, target=mock_target),
                       ScanTuple(dumps=mock_data.dumps, state=ScanStateEnum.TRACK, index=1, target=mock_target)]
        for expect, scan_tuple in zip(expect_list, scan_tuple_list):
            self.assertTupleEqual(expect, scan_tuple)
