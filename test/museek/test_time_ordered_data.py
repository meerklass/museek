import itertools
import unittest
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import numpy as np

from museek.flag_list import FlagList
from museek.receiver import Polarisation, Receiver
from museek.time_ordered_data import ScanStateEnum, ScanTuple, TimeOrderedData


class TestTimeOrderedData(unittest.TestCase):
    @patch.object(TimeOrderedData, "_get_flag_element_factory")
    @patch.object(TimeOrderedData, "_get_data_element_factory")
    @patch.object(TimeOrderedData, "_correlator_products_indices")
    @patch.object(TimeOrderedData, "_get_data")
    def setUp(
        self,
        mock_get_data,
        mock_correlator_products_indices,
        mock_get_data_element_factory,
        mock_get_flag_element_factory,
    ):
        mock_receiver_name_list = ["m000h", "m000v", "m001h", "m001v"]
        self.mock_receiver_list = [
            Receiver.from_string(name) for name in mock_receiver_name_list
        ]
        mock_corr_products = np.asarray(
            list(itertools.product(mock_receiver_name_list, mock_receiver_name_list))
        )
        self.mock_katdal_data = MagicMock()
        type(self.mock_katdal_data).corr_products = PropertyMock(
            return_value=mock_corr_products
        )

        self.mock_correlator_products_indices = mock_correlator_products_indices
        self.mock_get_data_element_factory = mock_get_data_element_factory
        self.mock_get_flag_element_factory = mock_get_flag_element_factory
        mock_get_data.return_value = self.mock_katdal_data
        mock_block_name = Mock()
        mock_data_folder = Mock()

        self.time_ordered_data = TimeOrderedData(
            block_name=mock_block_name,
            receivers=self.mock_receiver_list,
            token=None,
            data_folder=mock_data_folder,
        )

    def test_init(self):
        self.mock_katdal_data.select.assert_called_once_with(
            corrprods=self.mock_correlator_products_indices.return_value
        )

    def test_str(self):
        expect = str(self.mock_katdal_data)
        self.assertEqual(str(self.time_ordered_data), expect)

    @patch.object(TimeOrderedData, "_set_data_elements_from_self")
    def test_set_data_elements(self, mock_set_data_elements_from_self):
        self.time_ordered_data.timestamps = 1
        mock_scan_state = Mock()
        mock_data = Mock()
        self.time_ordered_data.set_data_elements(
            scan_state=mock_scan_state, data=mock_data
        )
        mock_set_data_elements_from_self.assert_called_once_with(
            scan_state=mock_scan_state
        )

    @patch.object(TimeOrderedData, "_set_data_elements_from_katdal")
    def test_set_data_elements_when_timestamps_is_none(
        self, mock_set_data_elements_from_katdal
    ):
        self.time_ordered_data.timestamps = None
        mock_scan_state = Mock()
        mock_data = Mock()
        self.time_ordered_data.set_data_elements(
            scan_state=mock_scan_state, data=mock_data
        )
        mock_set_data_elements_from_katdal.assert_called_once_with(
            scan_state=mock_scan_state, data=mock_data
        )

    @patch.object(FlagList, "from_array")
    @patch.object(TimeOrderedData, "_visibility_flags_weights")
    def test_load_visibility_flag_weights(
        self, mock_visibility_flags_weights, mock_from_array
    ):
        mock_visibility, mock_flags, mock_weights = Mock(), Mock(), Mock()
        mock_visibility_flags_weights.return_value = (
            mock_visibility,
            mock_flags,
            mock_weights,
        )
        self.time_ordered_data.load_visibility_flags_weights(polars="auto")
        mock_from_array.assert_called_once_with(
            array=mock_flags,
            element_factory=self.mock_get_flag_element_factory.return_value,
        )
        self.assertEqual(
            self.time_ordered_data.visibility,
            self.mock_get_data_element_factory.return_value.create.return_value,
        )
        self.assertEqual(self.time_ordered_data.flags, mock_from_array.return_value)
        self.assertEqual(
            self.time_ordered_data.weights,
            self.mock_get_data_element_factory.return_value.create.return_value,
        )

    @patch("museek.time_ordered_data.FlagList")
    def test_load_visibility_flag_weights_when_already_loaded(self, mock_flag_list):
        self.time_ordered_data.visibility = 1
        self.time_ordered_data.flags = 1
        self.time_ordered_data.weights = 1
        self.time_ordered_data.load_visibility_flags_weights(polars="auto")
        mock_flag_list.assert_not_called()
        self.assertEqual(self.time_ordered_data.visibility, 1)
        self.assertEqual(self.time_ordered_data.flags, 1)
        self.assertEqual(self.time_ordered_data.weights, 1)

    @patch.object(FlagList, "from_array")
    @patch.object(TimeOrderedData, "_visibility_flags_weights")
    def test_delete_visibility_flags_weights(
        self, mock_visibility_flags_weights, mock_from_array
    ):
        mock_visibility, mock_flags, mock_weights = Mock(), Mock(), Mock()
        mock_visibility_flags_weights.return_value = (
            mock_visibility,
            mock_flags,
            mock_weights,
        )
        self.time_ordered_data.load_visibility_flags_weights(polars="auto")
        mock_from_array.assert_called_once_with(
            array=mock_flags,
            element_factory=self.mock_get_flag_element_factory.return_value,
        )
        self.assertEqual(
            self.time_ordered_data.visibility,
            self.mock_get_data_element_factory.return_value.create.return_value,
        )
        self.assertEqual(self.time_ordered_data.flags, mock_from_array.return_value)
        self.assertEqual(
            self.time_ordered_data.weights,
            self.mock_get_data_element_factory.return_value.create.return_value,
        )
        self.time_ordered_data.delete_visibility_flags_weights(polars="auto")
        self.assertIsNone(self.time_ordered_data.visibility)
        self.assertIsNone(self.time_ordered_data.flags)
        self.assertIsNone(self.time_ordered_data.weights)

    def test_antenna(self):
        mock_receiver = MagicMock()
        mock_antenna_name_list = MagicMock()
        self.time_ordered_data._antenna_name_list = mock_antenna_name_list
        antenna = self.time_ordered_data.antenna(receiver=mock_receiver)
        mock_antenna_name_list.index.assert_called_once_with(mock_receiver.antenna_name)
        self.assertEqual(
            self.time_ordered_data.antennas.__getitem__.return_value, antenna
        )

    def test_antenna_when_explicit(self):
        self.time_ordered_data._antenna_name_list = ["m000", "m001"]
        self.time_ordered_data.antennas = ["antenna0", "antenna1"]
        antenna = self.time_ordered_data.antenna(
            receiver=Receiver(antenna_number=1, polarisation=Polarisation.v)
        )
        self.assertEqual("antenna1", antenna)

    @patch.object(TimeOrderedData, "antenna")
    def test_antenna_index_of_receiver(self, mock_antenna):
        self.time_ordered_data.antennas = [
            MagicMock(),
            mock_antenna.return_value,
            MagicMock(),
        ]
        antenna_index = self.time_ordered_data.antenna_index_of_receiver(
            receiver=MagicMock()
        )
        self.assertEqual(1, antenna_index)

    @patch.object(TimeOrderedData, "antenna")
    def test_antenna_index_of_receiver_when_not_there(self, mock_antenna):
        self.time_ordered_data.antennas = [MagicMock(), MagicMock()]
        antenna_index = self.time_ordered_data.antenna_index_of_receiver(
            receiver=MagicMock()
        )
        mock_antenna.assert_called_once()
        self.assertIsNone(antenna_index)

    def test_receiver_indices_of_antenna_when_occurs_twice(self):
        mock_receiver_1 = MagicMock(antenna_name="angela")
        mock_receiver_2 = MagicMock(antenna_name="michael")
        mock_receiver_3 = MagicMock(antenna_name="angela")
        mock_antenna = MagicMock()
        mock_antenna.name = "angela"
        self.time_ordered_data.receivers = [
            mock_receiver_1,
            mock_receiver_2,
            mock_receiver_3,
        ]
        self.assertListEqual(
            [0, 2],
            self.time_ordered_data.receiver_indices_of_antenna(antenna=mock_antenna),
        )

    def test_receiver_indices_of_antenna_when_occurs_once(self):
        mock_receiver_1 = MagicMock(antenna_name="angela")
        mock_receiver_2 = MagicMock(antenna_name="michael")
        mock_receiver_3 = MagicMock(antenna_name="angela")
        mock_antenna = MagicMock()
        self.time_ordered_data.receivers = [
            mock_receiver_1,
            mock_receiver_2,
            mock_receiver_3,
        ]
        mock_antenna.name = "michael"
        self.assertListEqual(
            [1],
            self.time_ordered_data.receiver_indices_of_antenna(antenna=mock_antenna),
        )

    def test_receiver_indices_of_antenna_when_occurs_never(self):
        mock_receiver_1 = MagicMock(antenna_name="angela")
        mock_receiver_2 = MagicMock(antenna_name="michael")
        mock_receiver_3 = MagicMock(antenna_name="angela")
        mock_antenna = MagicMock()
        self.time_ordered_data.receivers = [
            mock_receiver_1,
            mock_receiver_2,
            mock_receiver_3,
        ]
        mock_antenna.name = "fred"
        self.assertListEqual(
            [], self.time_ordered_data.receiver_indices_of_antenna(antenna=mock_antenna)
        )

    def test_set_gain_solution(self):
        mock_gain_solution_array = MagicMock()
        mock_gain_solution_mask_array = MagicMock()
        mock_flags = MagicMock()
        self.time_ordered_data.flags = mock_flags
        self.time_ordered_data.set_gain_solution(
            gain_solution_array=mock_gain_solution_array,
            gain_solution_mask_array=mock_gain_solution_mask_array,
        )
        self.assertEqual(
            [
                call(array=mock_gain_solution_array),
                call(array=mock_gain_solution_mask_array),
            ],
            self.mock_get_data_element_factory.return_value.create.call_args_list[-2:],
        )
        self.assertIsNotNone(self.time_ordered_data.gain_solution)
        mock_flags.add_flag.assert_called_once()

    def test_corrected_visibility_when_no_gain_solution_expect_none(self):
        self.assertIsNone(self.time_ordered_data.corrected_visibility())

    def test_corrected_visibility(self):
        self.time_ordered_data.visibility = 2
        self.time_ordered_data.gain_solution = 3
        self.assertEqual(2 / 3, self.time_ordered_data.corrected_visibility())

    def test_set_data_elements_from_katdal(self):
        mock_scan_state = Mock()
        mock_data = MagicMock()
        self.time_ordered_data._set_data_elements_from_katdal(
            scan_state=mock_scan_state, data=mock_data
        )
        self.assertEqual(self.time_ordered_data.scan_state, mock_scan_state)
        self.assertEqual(
            mock_scan_state.factory(), self.time_ordered_data._element_factory
        )
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

    @patch.object(TimeOrderedData, "_auto_select")
    @patch.object(TimeOrderedData, "_get_data")
    def test_set_data_elements_from_katdal_when_data_is_none(
        self, mock_get_data, mock_auto_select
    ):
        mock_scan_state = Mock()
        self.time_ordered_data._set_data_elements_from_katdal(
            scan_state=mock_scan_state, data=None
        )
        self.assertEqual(self.time_ordered_data.scan_state, mock_scan_state)
        mock_get_data.assert_called_once()
        mock_auto_select.assert_called_once_with(data=mock_get_data.return_value)

    @patch.object(FlagList, "from_array")
    def test_set_data_elements_from_self(self, mock_from_array):
        mock_scan_state = Mock()
        self.time_ordered_data.visibility = Mock(array=1)
        self.time_ordered_data.flags = Mock(array=1)
        self.time_ordered_data.weights = Mock(array=1)
        self.time_ordered_data._set_data_elements_from_self(scan_state=mock_scan_state)
        self.assertEqual(self.time_ordered_data.scan_state, mock_scan_state)
        self.assertEqual(
            mock_scan_state.factory(), self.time_ordered_data._element_factory
        )
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
        self.assertEqual(expect, self.time_ordered_data.visibility)
        self.assertEqual(expect, self.time_ordered_data.weights)
        self.assertEqual(mock_from_array.return_value, self.time_ordered_data.flags)
        mock_from_array.assert_called_once()

    @patch.object(TimeOrderedData, "set_data_elements")
    @patch.object(TimeOrderedData, "_auto_select")
    @patch("museek.time_ordered_data.katdal.open")
    def test_get_data_when_data_folder(
        self, mock_open, mock_auto_select, mock_set_data_elements
    ):
        block_name = "block"
        token = None
        data_folder = "folder"
        mock_receiver_list = [Mock(), Mock()]

        TimeOrderedData(
            block_name=block_name,
            receivers=mock_receiver_list,
            token=token,
            data_folder=data_folder,
        )
        mock_open.assert_called_once_with(
            f"{data_folder}/{block_name}/{block_name}/{block_name}_sdp_l0.full.rdb"
        )
        mock_auto_select.assert_called_once()
        mock_set_data_elements.assert_called_once()

    @patch.object(TimeOrderedData, "set_data_elements")
    @patch.object(TimeOrderedData, "_auto_select")
    @patch("museek.time_ordered_data.katdal.open")
    def test_get_data_when_token(
        self, mock_open, mock_auto_select, mock_set_data_elements
    ):
        block_name = "block"
        token = "token"
        mock_receiver_list = [Mock(), Mock()]

        TimeOrderedData(
            block_name=block_name,
            receivers=mock_receiver_list,
            token=token,
            data_folder=None,
        )
        mock_open.assert_called_once_with(
            f"https://archive-gw-1.kat.ac.za/{block_name}/{block_name}_sdp_l0.full.rdb?token={token}"
        )
        mock_auto_select.assert_called_once()
        mock_set_data_elements.assert_called_once()

    def test_dumps_of_scan_state(self):
        mock_scan_state = Mock(state="2")
        mock_scan_tuple_list = [
            Mock(state="mock"),
            Mock(state=mock_scan_state, dumps=[Mock()]),
            Mock(state="mock"),
        ]
        self.time_ordered_data._scan_tuple_list = mock_scan_tuple_list
        self.time_ordered_data.scan_state = mock_scan_state
        dumps = self.time_ordered_data._dumps_of_scan_state()
        expect = mock_scan_tuple_list[1].dumps
        self.assertListEqual(expect, dumps)

    def test_dumps_of_scan_state_when_scan_state_is_none(self):
        self.assertIsNone(self.time_ordered_data._dumps_of_scan_state())

    @patch.object(TimeOrderedData, "_dumps_of_scan_state")
    def test_dumps(self, mock_dumps_of_scan_state):
        self.time_ordered_data._dumps()
        mock_dumps_of_scan_state.assert_called_once()

    def test_get_data_element_factory(self):
        mock_scan_state = Mock()
        self.time_ordered_data.scan_state = mock_scan_state
        self.assertEqual(
            mock_scan_state.factory(),
            self.time_ordered_data._get_data_element_factory(),
        )

    @patch("museek.time_ordered_data.DataElementFactory")
    def test_get_data_element_factory_when_scan_state_is_none(self, mock_factory):
        self.assertEqual(
            mock_factory(), self.time_ordered_data._get_data_element_factory()
        )

    @patch.object(TimeOrderedData, "_split_cross_and_auto_products")
    @patch.object(TimeOrderedData, "_auto_select")
    @patch("museek.time_ordered_data.katdal")
    @patch.object(TimeOrderedData, "_load_visibility")
    def test_visibility_flags_weights_when_force_load_from_correlator_data(
        self, mock_load_visibility, mock_katdal, mock_auto_select, mock_split
    ):
        self.time_ordered_data._force_load_from_correlator_data = True
        # Mock the return values with proper shapes
        mock_vis = np.array([[[1.0 + 2.0j, 3.0 + 4.0j]]])
        mock_flags = np.array([[[[True, False]]]])
        mock_weights = np.array([[[0.5, 0.6]]])
        mock_load_visibility.return_value = (mock_vis, mock_flags, mock_weights)
        # Mock split to return data and indices [0, 1] for auto products
        mock_split.return_value = (mock_vis, [0, 1])

        visibility, flags, weights = self.time_ordered_data._visibility_flags_weights(
            polars="auto"
        )
        mock_katdal.open.assert_called_once()
        mock_auto_select.assert_called_once()
        mock_load_visibility.assert_called_once()

    @patch("museek.time_ordered_data.DaskLazyIndexer")
    def test_load_visibility(self, mock_dask_lazy_indexer):
        self.time_ordered_data.shape = (1, 1, 1)
        visibility, flags, weights = self.time_ordered_data._load_visibility(
            data=self.mock_katdal_data
        )
        mock_dask_lazy_indexer.get.assert_called_once()
        np.testing.assert_array_equal(np.asarray([[[0.0 + 0.0j]]]), visibility)
        np.testing.assert_array_equal(np.asarray([[[[0]]]]), flags)
        np.testing.assert_array_equal(np.asarray([[[0]]]), weights)

    @patch.object(np, "asarray")
    @patch.object(np, "savez_compressed")
    def test_visibility_flag_weights_to_cache_file(
        self, mock_savez_compressed, mock_asarray
    ):
        mock_visibility = Mock()
        mock_flags = Mock()
        mock_weights = Mock()
        mock_correlator_products = Mock()
        mock_cache_file = "test_cache.npz"
        self.time_ordered_data._visibility_flag_weights_to_cache_file(
            cache_file=mock_cache_file,
            visibility=mock_visibility,
            flags=mock_flags,
            weights=mock_weights,
            correlator_products=mock_correlator_products,
        )
        mock_savez_compressed.assert_called_once_with(
            mock_cache_file,
            visibility=mock_visibility,
            flags=mock_flags,
            weights=mock_weights,
            correlator_products=mock_asarray.return_value,
        )
        mock_asarray.assert_called_once_with(mock_correlator_products)

    @patch.object(np, "asarray")
    @patch.object(np, "savez_compressed")
    def test_visibility_flag_weights_to_cache_file_when_scan_state_not_none(
        self, mock_savez_compressed, mock_asarray
    ):
        mock_visibility = Mock()
        mock_flags = Mock()
        mock_weights = Mock()
        mock_correlator_products = Mock()
        mock_cache_file = "test_cache.npz"
        self.time_ordered_data.scan_state = Mock()
        self.assertRaises(
            ValueError,
            self.time_ordered_data._visibility_flag_weights_to_cache_file,
            cache_file=mock_cache_file,
            visibility=mock_visibility,
            flags=mock_flags,
            weights=mock_weights,
            correlator_products=mock_correlator_products,
        )

        mock_savez_compressed.assert_not_called()
        mock_asarray.assert_not_called()

    def test_correlator_products_indices(self):
        all_correlator_products = np.asarray(
            [("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")]
        )
        self.time_ordered_data.auto_correlator_products = np.asarray(
            [("c", "c"), ("a", "a")]
        )
        expect = [2, 0]
        indices = self.time_ordered_data._correlator_products_indices(
            all_correlator_products=all_correlator_products
        )
        self.assertListEqual(expect, indices)

    def test_correlator_products_indices_when_all_missing_expect_raise(self):
        all_correlator_products = np.asarray(
            [("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")]
        )
        self.time_ordered_data.auto_correlator_products = np.asarray(
            [("e", "e"), ("f", "f")]
        )
        self.assertRaises(
            ValueError,
            self.time_ordered_data._correlator_products_indices,
            all_correlator_products=all_correlator_products,
        )

    def test_correlator_products_indices_when_one_missing_expect_raise(self):
        all_correlator_products = np.asarray(
            [("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")]
        )
        self.time_ordered_data.auto_correlator_products = np.asarray(
            [("a", "a"), ("f", "f")]
        )
        self.assertRaises(
            ValueError,
            self.time_ordered_data._correlator_products_indices,
            all_correlator_products=all_correlator_products,
        )

    def test_get_auto_correlator_products(self):
        self.time_ordered_data.receivers = [
            Receiver(antenna_number=0, polarisation=Polarisation.v),
            Receiver(antenna_number=0, polarisation=Polarisation.h),
            Receiver(antenna_number=200, polarisation=Polarisation.h),
        ]
        expect_list = [["m000v", "m000v"], ["m000h", "m000h"], ["m200h", "m200h"]]
        for expect, correlator_product in zip(
            expect_list, self.time_ordered_data._get_auto_correlator_products()
        ):
            self.assertListEqual(expect, correlator_product)

    @patch.object(TimeOrderedData, "_correlator_products_indices")
    def test_auto_select(self, mock_correlator_products_indices):
        # Reset the mock since setUp already called _auto_select
        self.mock_katdal_data.select.reset_mock()
        mock_correlator_products_indices.reset_mock()

        self.time_ordered_data._auto_select(data=self.mock_katdal_data)
        self.mock_katdal_data.select.assert_called_once_with(
            corrprods=mock_correlator_products_indices.return_value
        )
        mock_correlator_products_indices.assert_called_once_with(
            all_correlator_products=self.mock_katdal_data.corr_products
        )

    def test_get_receivers_if_receivers_given(self):
        mock_receivers = [Receiver.from_string("m000h")]
        self.assertEqual(
            mock_receivers,
            self.time_ordered_data._get_receivers(
                requested_receivers=mock_receivers, data=self.mock_katdal_data
            ),
        )

    def test_get_receivers_if_receivers_given_but_not_available(self):
        mock_receiver = Mock()
        mock_receiver.name = "nonexistent_receiver"
        mock_receivers = [mock_receiver]
        self.assertEqual(
            [],
            self.time_ordered_data._get_receivers(
                requested_receivers=mock_receivers, data=self.mock_katdal_data
            ),
        )

    def test_get_receivers_if_receivers_none(self):
        self.assertListEqual(
            self.mock_receiver_list,
            self.time_ordered_data._get_receivers(
                requested_receivers=None, data=self.mock_katdal_data
            ),
        )

    def test_get_scan_tuple_list(self):
        mock_target = Mock()
        mock_scans = MagicMock(
            return_value=[(0, "scan", mock_target), (1, "track", mock_target)]
        )
        mock_data = MagicMock(scans=mock_scans)
        scan_tuple_list = self.time_ordered_data._get_scan_tuple_list(data=mock_data)
        expect_list = [
            ScanTuple(
                dumps=mock_data.dumps,
                state=ScanStateEnum.SCAN,
                index=0,
                target=mock_target,
            ),
            ScanTuple(
                dumps=mock_data.dumps,
                state=ScanStateEnum.TRACK,
                index=1,
                target=mock_target,
            ),
        ]
        for expect, scan_tuple in zip(expect_list, scan_tuple_list):
            self.assertTupleEqual(expect, scan_tuple)

    @patch.object(TimeOrderedData, "_shift_right_ascension")
    def test_coherent_right_ascension_when_not_coherent(
        self, mock_shift_right_ascension
    ):
        mock_right_ascension = np.asarray(
            [
                324.854,
                324.855,
                325.984,
                324.855,
                324.854,
                323.726,
                3.852,
                1.494,
                3.527,
                3.977,
                2.714,
                5.920,
                3.986,
                5.248,
                6.423,
                4.390,
                7.674,
                6.471,
                6.958,
                8.870,
                6.407,
                9.416,
                8.945,
                8.654,
                11.295,
                8.909,
                11.151,
                11.406,
                10.341,
                13.531,
                11.412,
                12.868,
                13.864,
                12.020,
                15.288,
                13.892,
                14.580,
                16.304,
                13.824,
                17.032,
                16.375,
                16.281,
                18.741,
                16.332,
                18.767,
                18.842,
                17.971,
                21.106,
                18.832,
                20.491,
                21.298,
                19.647,
                22.899,
                21.328,
                22.200,
                23.751,
                21.339,
                24.646,
                23.808,
                23.902,
                26.186,
                23.756,
                26.381,
                26.274,
                25.593,
                28.620,
                26.261,
                28.105,
                324.854,
                324.854,
                325.984,
                324.854,
                324.854,
                323.725,
            ]
        )
        coherent = self.time_ordered_data._coherent_right_ascension(
            right_ascension=mock_right_ascension[:, np.newaxis]
        )
        self.assertEqual(mock_shift_right_ascension.return_value, coherent)
        mock_shift_right_ascension.assert_called_once()

    def test_coherent_right_ascension_when_coherent(self):
        mock_right_ascension = np.asarray(
            [
                324.854,
                324.855,
                325.984,
                324.855,
                324.854,
                323.726,
                363.852,
                361.494,
                363.527,
                363.977,
                362.714,
                365.920,
                363.986,
                365.248,
                366.423,
                364.390,
                367.674,
                366.471,
                366.958,
                368.870,
                366.407,
                369.416,
                368.945,
                368.654,
                371.295,
                368.909,
                371.151,
                371.406,
                370.341,
                373.531,
                371.412,
                372.868,
                373.864,
                372.020,
                375.288,
                373.892,
                374.580,
                376.304,
                373.824,
                377.032,
                376.375,
                376.281,
                378.741,
                376.332,
                378.767,
                378.842,
                377.971,
                381.106,
                378.832,
                380.491,
                381.298,
                379.647,
                382.899,
                381.328,
                382.200,
                383.751,
                381.339,
                384.646,
                383.808,
                383.902,
                386.186,
                383.756,
                386.381,
                386.274,
                385.593,
                388.620,
                386.261,
                388.105,
                324.854,
                324.854,
                325.984,
                324.854,
                324.854,
                323.725,
            ]
        )
        coherent = self.time_ordered_data._coherent_right_ascension(
            right_ascension=mock_right_ascension[:, np.newaxis]
        )
        np.testing.assert_array_equal(mock_right_ascension[:, np.newaxis], coherent)

    def test_shift_right_ascension(self):
        mock_right_ascension = np.array([[1, 2, 363, 364, 365], [6, 7, 368, 369, 370]])
        expect = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        shifted = self.time_ordered_data._shift_right_ascension(
            right_ascension=mock_right_ascension
        )
        np.testing.assert_array_equal(expect, shifted)
