import unittest
from unittest.mock import patch, Mock, MagicMock, call

import numpy as np

from museek.receiver import Receiver, Polarisation
from museek.time_ordered_data import TimeOrderedData, ScanStateEnum, ScanTuple


class TestTimeOrderedData(unittest.TestCase):

    @patch('museek.time_ordered_data.TimeOrderedDataElement')
    @patch.object(TimeOrderedData, '_set_scan_state_dumps')
    @patch.object(TimeOrderedData, '_correlator_products_indices')
    @patch.object(TimeOrderedData, 'load_data')
    def setUp(self,
              mock_load_data,
              mock_correlator_products_indices,
              mock_set_scan_state_dumps,
              mock_time_ordered_data_element):
        self.mock_katdal_data = MagicMock()
        self.mock_correlator_products_indices = mock_correlator_products_indices
        mock_load_data.return_value = self.mock_katdal_data
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
        self.assertIsNone(self.time_ordered_data.scan_dumps)
        self.assertIsNone(self.time_ordered_data.track_dumps)
        self.assertIsNone(self.time_ordered_data.slew_dumps)
        self.assertIsNone(self.time_ordered_data.stop_dumps)

    def test_str(self):
        expect = str(self.mock_katdal_data)
        self.assertEqual(str(self.time_ordered_data), expect)

    @patch('museek.time_ordered_data.katdal.open')
    def test_load_data_when_data_folder(self, mock_open):
        block_name = 'block'
        self.time_ordered_data.load_data(block_name=block_name, token=None, data_folder='')
        mock_open.assert_called_once_with(f'{block_name}/{block_name}/{block_name}_sdp_l0.full.rdb')

    @patch('museek.time_ordered_data.katdal.open')
    def test_load_data_when_token(self, mock_open):
        block_name = 'block'
        token = 'token'
        self.time_ordered_data.load_data(block_name=block_name, token=token, data_folder=None)
        mock_open.assert_called_once_with(
            f'https://archive-gw-1.kat.ac.za/{block_name}/{block_name}_sdp_l0.full.rdb?{token}'
        )

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

    @patch.object(TimeOrderedData, '_element')
    @patch.object(TimeOrderedData, '_visibility_flags_weights')
    def test_load_visibility_flag_weights(self, mock_visibility_flags_weights, mock_element):
        mock_visibility_flags_weights.return_value = (Mock(), Mock(), Mock())
        self.time_ordered_data.load_visibility_flags_weights()
        self.assertEqual(self.time_ordered_data.visibility, mock_element.return_value)
        self.assertEqual(self.time_ordered_data.flags, [mock_element.return_value])
        self.assertEqual(self.time_ordered_data.weights, mock_element.return_value)

    @patch.object(TimeOrderedData, '_element')
    @patch.object(TimeOrderedData, '_visibility_flags_weights')
    def test_delete_visibility_flags_weights(self, mock_visibility_flags_weights, mock_element):
        mock_visibility_flags_weights.return_value = (Mock(), Mock(), Mock())
        self.time_ordered_data.load_visibility_flags_weights()
        self.assertEqual(self.time_ordered_data.visibility, mock_element.return_value)
        self.assertEqual(self.time_ordered_data.flags, [mock_element.return_value])
        self.assertEqual(self.time_ordered_data.weights, mock_element.return_value)
        self.time_ordered_data.delete_visibility_flags_weights()
        self.assertIsNone(self.time_ordered_data.visibility)
        self.assertIsNone(self.time_ordered_data.flags)
        self.assertIsNone(self.time_ordered_data.weights)

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

    @patch.object(TimeOrderedData, '__setattr__')
    @patch.object(TimeOrderedData, '_dumps_of_scan_state')
    def test_set_scan_state_dumps(self, mock_dumps_of_scan_state, mock_setattr):
        self.time_ordered_data._set_scan_state_dumps()
        mock_dumps_of_scan_state.assert_has_calls(calls=[call(scan_state=ScanStateEnum.SCAN),
                                                         call(scan_state=ScanStateEnum.TRACK),
                                                         call(scan_state=ScanStateEnum.SLEW),
                                                         call(scan_state=ScanStateEnum.STOP)])
        mock_setattr.assert_has_calls(
            calls=[call('scan_dumps', mock_dumps_of_scan_state.return_value),
                   call('track_dumps', mock_dumps_of_scan_state.return_value),
                   call('slew_dumps', mock_dumps_of_scan_state.return_value),
                   call('stop_dumps', mock_dumps_of_scan_state.return_value)]
        )

    @patch.object(TimeOrderedData, '_dumps_of_scan_state')
    def test_set_scan_state_dumps_when_force_false_expect_dumps_unchanged(self, mock_dumps_of_scan_state):
        self.time_ordered_data.scan_dumps = [1]
        self.time_ordered_data._set_scan_state_dumps(force=False)
        mock_dumps_of_scan_state.assert_has_calls(calls=[call(scan_state=ScanStateEnum.SCAN),
                                                         call(scan_state=ScanStateEnum.TRACK),
                                                         call(scan_state=ScanStateEnum.SLEW),
                                                         call(scan_state=ScanStateEnum.STOP)])
        self.assertEqual(mock_dumps_of_scan_state.return_value, self.time_ordered_data.track_dumps)
        self.assertEqual([1], self.time_ordered_data.scan_dumps)

    @patch.object(TimeOrderedData, '_dumps_of_scan_state')
    def test_set_scan_state_dumps_when_force_true_expect_dumps_changed(self, mock_dumps_of_scan_state):
        self.time_ordered_data.scan_dumps = [1]
        self.time_ordered_data._set_scan_state_dumps(force=True)
        mock_dumps_of_scan_state.assert_has_calls(calls=[call(scan_state=ScanStateEnum.SCAN),
                                                         call(scan_state=ScanStateEnum.TRACK),
                                                         call(scan_state=ScanStateEnum.SLEW),
                                                         call(scan_state=ScanStateEnum.STOP)])
        self.assertEqual(mock_dumps_of_scan_state.return_value, self.time_ordered_data.track_dumps)
        self.assertEqual(mock_dumps_of_scan_state.return_value, self.time_ordered_data.scan_dumps)

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

    def test_dumps_of_scan_state(self):
        self.time_ordered_data._scan_tuple_list = [
            ScanTuple(dumps=[0], state=ScanStateEnum.SCAN, index=0, target=Mock()),
            ScanTuple(dumps=[1], state=ScanStateEnum.TRACK, index=1, target=Mock())]
        self.assertEqual([0], self.time_ordered_data._dumps_of_scan_state(scan_state=ScanStateEnum.SCAN))
        self.assertEqual([1], self.time_ordered_data._dumps_of_scan_state(scan_state=ScanStateEnum.TRACK))

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

    @patch('museek.time_ordered_data.TimeOrderedDataElement')
    def test_element(self, mock_time_ordered_data_element):
        mock_array = Mock()
        element = self.time_ordered_data._element(array=mock_array)
        mock_time_ordered_data_element.assert_called_once_with(array=mock_array, parent=self.time_ordered_data)
        self.assertEqual(mock_time_ordered_data_element(), element)

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
