import unittest
from unittest.mock import MagicMock, patch, Mock

import numpy as np

from museek.data_element import DataElement
from museek.flag_element import FlagElement
from museek.flag_list import FlagList


class TestDataElement(unittest.TestCase):
    def setUp(self):
        self.shape = (3, 3, 3)
        self.mock_parent = MagicMock()
        self.array = np.resize(np.arange(27), (3, 3, 3))
        self.element = DataElement(array=self.array)

    def test_init_expect_raise(self):
        array = np.zeros((2, 2))
        self.assertRaises(ValueError, DataElement, array=array)

    def test_mul_when_other_is_time_ordered_data_element(self):
        element_2 = DataElement(array=np.ones(self.shape) * 2)
        multiplied = self.element * element_2
        expect = np.resize(np.arange(27), self.shape) * 2
        np.testing.assert_array_equal(expect, multiplied.array)

    def test_mul_when_other_is_numpy_array(self):
        multiplied = self.element * np.ones(self.shape) * 3
        expect = np.resize(np.arange(27), self.shape) * 3
        np.testing.assert_array_equal(expect, multiplied.array)

    def test_mul_when_other_is_float(self):
        multiplied = self.element * 2.
        expect = np.resize(np.arange(27), self.shape) * 2.
        np.testing.assert_array_equal(expect, multiplied.array)

    def test_truediv_when_other_is_time_ordered_data_element(self):
        element_2 = DataElement(array=np.ones(self.shape) * 2)
        divided = self.element / element_2
        expect = np.resize(np.arange(27), self.shape) / 2
        np.testing.assert_array_equal(expect, divided.array)

    def test_truediv_when_other_is_numpy_array(self):
        divided = self.element / (np.ones(self.shape) * 3)
        expect = np.resize(np.arange(27), self.shape) / 3
        np.testing.assert_array_equal(expect, divided.array)

    def test_truediv_when_other_is_float(self):
        divided = self.element / 2.
        expect = np.resize(np.arange(27), self.shape) / 2.
        np.testing.assert_array_equal(expect, divided.array)

    def test_sub_when_other_is_time_ordered_data_element(self):
        element_2 = DataElement(array=np.ones(self.shape))
        subtracted = self.element - element_2
        expect = np.resize(np.arange(-1, 26), self.shape)
        np.testing.assert_array_equal(expect, subtracted.array)

    def test_sub_when_other_is_numpy_array(self):
        subtracted = self.element - (np.ones(self.shape) * 3)
        expect = np.resize(np.arange(-3, 24), self.shape)
        np.testing.assert_array_equal(expect, subtracted.array)

    def test_sub_when_other_is_float(self):
        subtracted = self.element - 2.
        expect = np.resize(np.arange(-2, 25), self.shape)
        np.testing.assert_array_equal(expect, subtracted.array)

    def test_add_when_other_is_time_ordered_data_element(self):
        element_2 = DataElement(array=np.ones(self.shape))
        subtracted = self.element + element_2
        expect = np.resize(np.arange(1, 28), self.shape)
        np.testing.assert_array_equal(expect, subtracted.array)

    def test_add_when_other_is_numpy_array(self):
        subtracted = self.element + (np.ones(self.shape) * 3)
        expect = np.resize(np.arange(3, 30), self.shape)
        np.testing.assert_array_equal(expect, subtracted.array)

    def test_add_when_other_is_float(self):
        subtracted = self.element + 2.
        expect = np.resize(np.arange(2, 29), self.shape)
        np.testing.assert_array_equal(expect, subtracted.array)

    @patch('museek.abstract_data_element.np')
    def test_getitem(self, mock_np):
        self.assertEqual(mock_np.squeeze.return_value, self.element[0, 1, 2])
        mock_np.squeeze.assert_called_once_with(5)

    def test_str(self):
        self.assertEqual(str(self.element.get_array()), str(self.element))

    def test_eq_when_true(self):
        self.assertEqual(self.element, self.element.get())

    def test_eq_when_false(self):
        array = np.resize(np.arange(27, 54), (3, 3, 3))
        element = DataElement(array=array)
        self.assertNotEqual(element, self.element)

    def test_eq_when_shapes_different_expect_false(self):
        array_1 = np.ones((3, 3, 1))
        array_2 = np.ones((3, 3, 2))
        self.assertNotEqual(DataElement(array=array_1), DataElement(array=array_2))

    @patch.object(np, 'squeeze')
    @patch.object(DataElement, 'get_array')
    def test_squeeze(self, mock_get_array, mock_np_squeeze):
        self.element.squeeze
        mock_get_array.assert_called_once()
        mock_np_squeeze.assert_called_once_with(mock_get_array.return_value)

    def test_shape(self):
        self.assertTupleEqual((3, 3, 3), self.element.shape)

    def test_mean_when_explicit(self):
        mean = self.element.mean(axis=0)
        self.assertEqual(3, len(mean.array.shape))
        expect = np.asarray([[9., 10., 11.],
                             [12., 13., 14.],
                             [15., 16., 17.]])
        np.testing.assert_array_equal(expect, mean.squeeze)
        
    def test_median_when_explicit(self):
        median = self.element.median(axis=0)
        self.assertEqual(3, len(median.array.shape))
        expect = np.asarray([[9., 10., 11.],
                             [12., 13., 14.],
                             [15., 16., 17.]])
        np.testing.assert_array_equal(expect, median.squeeze)    

    @patch('museek.data_element.np')
    def test_mean_when_mocked(self, mock_np):
        mock_np.mean.return_value.shape = (1, 1, 1)
        mock_axis = MagicMock()
        mean = self.element.mean(axis=mock_axis)
        self.assertEqual(mean.array, mock_np.mean.return_value)
        mock_np.mean.assert_called_once_with(self.element.array,
                                             axis=mock_axis,
                                             keepdims=True)
        
    @patch('museek.data_element.np')
    def test_median_when_mocked(self, mock_np):
        mock_np.median.return_value.shape = (1, 1, 1)
        mock_axis = MagicMock()
        median = self.element.median(axis=mock_axis)
        self.assertEqual(median.array, mock_np.median.return_value)
        mock_np.median.assert_called_once_with(self.element.array,
                                             axis=mock_axis,
                                             keepdims=True)    

    def test_mean_when_flags_is_not_none(self):
        flag_array = np.zeros((3, 3, 3), dtype=bool)
        flag_array[0, 0, 0] = True
        flag_array[1, 1, 1] = True
        flag_array[2, 2, 2] = True
        flags = FlagList(flags=[FlagElement(array=flag_array)])
        mean = self.element.mean(axis=0, flags=flags)
        self.assertEqual(3, len(mean.array.shape))
        expect = np.asarray([[13.5, 10., 11.],
                             [12., 13., 14.],
                             [15., 16., 12.5]])
        np.testing.assert_array_equal(expect, mean.squeeze)
    
    def test_median_when_flags_is_not_none(self):
        flag_array = np.zeros((3, 3, 3), dtype=bool)
        flag_array[0, 0, 0] = True
        flag_array[1, 1, 1] = True
        flag_array[2, 2, 2] = True
        flags = FlagList(flags=[FlagElement(array=flag_array)])
        median = self.element.median(axis=0, flags=flags)
        self.assertEqual(3, len(median.array.shape))
        expect = np.asarray([[13.5, 10., 11.],
                             [12., 13., 14.],
                             [15., 16., 12.5]])
        np.testing.assert_array_equal(expect, median.squeeze)

    def test_standard_deviation_when_explicit(self):
        std = self.element.standard_deviation(axis=0)
        self.assertEqual(3, len(std.array.shape))
        expect = np.ones((3, 3)) * np.sqrt(54)
        np.testing.assert_array_equal(expect, std.squeeze)

    @patch('museek.data_element.np')
    def test_standard_deviation_when_mocked(self, mock_np):
        mock_np.std.return_value.shape = (1, 1, 1)
        mock_axis = MagicMock()
        std = self.element.standard_deviation(axis=mock_axis)
        self.assertEqual(std.array, mock_np.std.return_value)
        mock_np.std.assert_called_once_with(self.element.array,
                                            axis=mock_axis,
                                            keepdims=True)

    def test_standard_deviation_when_flags_is_not_none(self):
        flag_array = np.zeros((3, 3, 3), dtype=bool)
        flag_array[0, 0, 0] = True
        flag_array[1, 1, 1] = True
        flag_array[2, 2, 2] = True
        flags = FlagList(flags=[FlagElement(array=flag_array)])
        std = self.element.standard_deviation(axis=0, flags=flags)
        self.assertEqual(3, len(std.array.shape))
        expect = np.asarray([[20.25, 54., 54.],
                             [54., 81., 54.],
                             [54., 54., 20.25]]) ** (1 / 2)
        np.testing.assert_array_equal(expect, std.squeeze)

    @patch('museek.data_element.DataElement')
    @patch('museek.data_element.np')
    def test_sum(self, mock_np, mock_data_element):
        mock_axis = MagicMock()
        mean = self.element.sum(axis=mock_axis)
        mock_np.sum.assert_called_once_with(self.element.array, axis=mock_axis, keepdims=True)
        mock_data_element.assert_called_once_with(array=mock_np.sum.return_value)
        self.assertEqual(mean, mock_data_element.return_value)

    def test_get(self):
        list_ = [1] * 9 + [2] * 9 + [3] * 9
        array = np.reshape(list_, (3, 3, 3))
        element = DataElement(array=array)
        np.testing.assert_array_equal(np.ones((3, 3)), element.get(time=0).squeeze)
        np.testing.assert_array_equal(np.ones((3, 3)) * 2, element.get(time=1).squeeze)
        expect = np.asarray([[1, 1, 1],
                             [2, 2, 2],
                             [3, 3, 3]])
        np.testing.assert_array_equal(expect, element.get(freq=1).squeeze)
        np.testing.assert_array_equal(expect, element.get(recv=1).squeeze)

    def test_get_when_only_time_given(self):
        shape_ = (10, 11, 3)
        element = DataElement(array=np.zeros(shape_))
        np.testing.assert_array_equal(np.zeros((11, 3)), element.get(time=5).squeeze)

    def test_get_when_three_integers_given(self):
        shape_ = (10, 11, 3)
        element = DataElement(array=np.zeros(shape_))
        self.assertEqual(0, element.get(time=5, freq=10, recv=2).squeeze)

    def test_get_when_lists_given(self):
        shape_ = (10, 11, 3)
        element = DataElement(array=np.zeros(shape_))
        np.testing.assert_array_equal(np.zeros((3, 2, 2)),
                                      element.get(time=[5, 6, 7], freq=[9, 10], recv=[0, 1]).squeeze)

    def test_get_when_slices_given(self):
        shape_ = (10, 11, 3)
        element = DataElement(array=np.zeros(shape_))
        np.testing.assert_array_equal(np.zeros((10, 2, 2)),
                                      element.get(time=slice(None), freq=slice(2, 4), recv=slice(0, 2)).squeeze)

    def test_get_when_minus_1_given(self):
        shape_ = (10, 11, 3)
        element = DataElement(array=np.zeros(shape_))
        np.testing.assert_array_equal(0, element.get(time=-1, freq=-1, recv=-1).squeeze)

    @patch.object(DataElement, 'get')
    def test_get_array(self, mock_get):
        mock_kwargs = MagicMock()
        self.assertEqual(mock_get.return_value.array, self.element.get_array(**mock_kwargs))
        mock_get.assert_called_once_with(**mock_kwargs)

    def test_get_array_when_shape_1_1_1(self):
        shape_ = (1, 1, 1)
        element = DataElement(array=np.zeros(shape_))
        result = element.get_array()
        self.assertEqual(0., result)
        self.assertIsInstance(result, np.ndarray)

    def test_get_array_when_shape_1_2_1(self):
        shape_ = (1, 2, 1)
        element = DataElement(array=np.zeros(shape_))
        result = element.get_array()
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(np.zeros(shape_), result)

    def test_min_when_axis_0(self):
        expect = np.array([[0, 1, 2],
                           [3, 4, 5],
                           [6, 7, 8]])
        np.testing.assert_array_equal(expect, self.element.min(axis=0).squeeze)

    def test_min_when_axis_1(self):
        expect = np.array([[0, 1, 2],
                           [9, 10, 11],
                           [18, 19, 20]])
        np.testing.assert_array_equal(expect, self.element.min(axis=1).squeeze)

    def test_min_when_axis_2(self):
        expect = np.array([[0, 3, 6],
                           [9, 12, 15],
                           [18, 21, 24]])
        np.testing.assert_array_equal(expect, self.element.min(axis=2).squeeze)

    def test_max_when_axis_0(self):
        expect = np.array([[18, 19, 20],
                           [21, 22, 23],
                           [24, 25, 26]])
        np.testing.assert_array_equal(expect, self.element.max(axis=0).squeeze)

    def test_max_when_axis_1(self):
        expect = np.array([[6, 7, 8],
                           [15, 16, 17],
                           [24, 25, 26]])
        np.testing.assert_array_equal(expect, self.element.max(axis=1).squeeze)

    def test_max_when_axis_2(self):
        expect = np.array([[2, 5, 8],
                           [11, 14, 17],
                           [20, 23, 26]])
        np.testing.assert_array_equal(expect, self.element.max(axis=2).squeeze)

    def test_channel_iterator(self):
        for i, (channel, arange) in enumerate(DataElement.channel_iterator(data_element=self.element)):
            self.assertEqual(self.element.get(freq=i), channel)
            np.testing.assert_array_equal(np.arange(3), arange)

    def test_channel_iterator_when_one_channel(self):
        array = np.resize(np.arange(9), (3, 1, 3))
        element = DataElement(array=array)
        for i, (channel, arange) in enumerate(DataElement.channel_iterator(data_element=element)):
            self.assertLess(i, 1)
            self.assertEqual(element.get(freq=i), channel)
            np.testing.assert_array_equal(np.arange(3), arange)

    def test_flagged_channel_iterator(self):
        flag_array = np.zeros_like(self.array, dtype=bool)
        flag_array[1, :, :] = True
        flag_element = FlagElement(array=flag_array)

        for i, (channel, arange) in enumerate(DataElement.flagged_channel_iterator(
                data_element=self.element.get(recv=0),
                flag_element=flag_element.get(recv=0)
        )):
            self.assertEqual(self.element.get(freq=i, recv=0), channel)
            np.testing.assert_array_equal(np.asarray([0, 2]), arange)

    @patch('museek.data_element.np')
    def test__mean(self, mock_np):
        mock_np.mean.return_value.shape = (1, 1, 1)
        mock_axis = Mock()
        self.assertIsInstance(self.element._mean(axis=mock_axis), DataElement)
        mock_np.mean.assert_called_once_with(self.element.array, axis=mock_axis, keepdims=True)
        
    @patch('museek.data_element.np')
    def test__median(self, mock_np):
        mock_np.median.return_value.shape = (1, 1, 1)
        mock_axis = Mock()
        self.assertIsInstance(self.element._median(axis=mock_axis), DataElement)
        mock_np.median.assert_called_once_with(self.element.array, axis=mock_axis, keepdims=True)

    @patch('museek.data_element.np')
    def test_kurtosis(self, mock_np):
        mock_np.kurtosis.return_value.shape = (1, 1, 1)
        mock_axis = Mock()
        self.assertIsInstance(self.element._kurtosis(axis=1), DataElement)

    @patch('museek.data_element.np')
    def test_std(self, mock_np):
        mock_np.std.return_value.shape = (1, 1, 1)
        mock_axis = Mock()
        self.assertIsInstance(self.element._std(axis=mock_axis), DataElement)
        mock_np.std.assert_called_once_with(self.element.array, axis=mock_axis, keepdims=True)

    def test_flagged_mean(self):
        flag_array = np.zeros((3, 3, 3), dtype=bool)
        flag_array[0, 0, 0] = True
        flag_array[1, 1, 1] = True
        flag_array[2, 2, 2] = True
        flags = FlagList(flags=[FlagElement(array=flag_array)])
        mean = self.element._flagged_mean(axis=0, flags=flags)
        self.assertEqual(3, len(mean.array.shape))
        expect = np.asarray([[13.5, 10., 11.],
                             [12., 13., 14.],
                             [15., 16., 12.5]])
        np.testing.assert_array_equal(expect, mean.squeeze)


    def test_flagged_median(self):
        flag_array = np.zeros((3, 3, 3), dtype=bool)
        flag_array[0, 0, 0] = True
        flag_array[1, 1, 1] = True
        flag_array[2, 2, 2] = True
        flags = FlagList(flags=[FlagElement(array=flag_array)])
        median = self.element._flagged_median(axis=0, flags=flags)
        self.assertEqual(3, len(median.array.shape))
        expect = np.asarray([[13.5, 10., 11.],
                             [12., 13., 14.],
                             [15., 16., 12.5]])
        np.testing.assert_array_equal(expect, median.squeeze)


    def test_flagged_kurtosis(self):
        flag_array = np.zeros((3, 3, 3), dtype=bool)
        flag_array[0, 0, 0] = True
        flag_array[1, 1, 1] = True
        flag_array[2, 2, 2] = True
        flags = FlagList(flags=[FlagElement(array=flag_array)])
        kurtosis = self.element._flagged_kurtosis(axis=0,flags=flags)
        self.assertEqual(3, len(kurtosis.array.shape))
        expect = np.asarray([[-2. , -1.5, -1.5],
                             [-1.5, -2. , -1.5],
                             [-1.5, -1.5, -2. ]])
        np.testing.assert_array_equal(expect, kurtosis.squeeze)

    def test_flagged_std(self):
        flag_array = np.zeros((3, 3, 3), dtype=bool)
        flag_array[0, 0, 0] = True
        flag_array[1, 1, 1] = True
        flag_array[2, 2, 2] = True
        flags = FlagList(flags=[FlagElement(array=flag_array)])
        std = self.element._flagged_std(axis=0, flags=flags)
        self.assertEqual(3, len(std.array.shape))
        expect = np.asarray([[20.25, 54., 54.],
                             [54., 81., 54.],
                             [54., 54., 20.25]]) ** (1 / 2)
        np.testing.assert_array_equal(expect, std.squeeze)
