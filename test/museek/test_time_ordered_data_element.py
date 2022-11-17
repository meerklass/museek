import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from museek.time_ordered_data_element import TimeOrderedDataElement


class TestTimeOrderedDataElement(unittest.TestCase):
    def setUp(self):
        self.shape = (3, 3, 3)
        self.mock_parent = MagicMock()
        self.array = np.resize(np.arange(27), (3, 3, 3))
        self.element = TimeOrderedDataElement(array=self.array, parent=self.mock_parent)

    def test_init_expect_raise(self):
        array = np.zeros((2, 2))
        self.assertRaises(ValueError, TimeOrderedDataElement, array=array, parent=self.mock_parent)

    def test_mul_when_parents_different_expect_raise(self):
        element_2 = TimeOrderedDataElement(array=np.zeros(self.shape), parent=MagicMock())
        self.assertRaises(ValueError, self.element.__mul__, element_2)

    def test_mul_when_other_is_time_ordered_data_element(self):
        element_2 = TimeOrderedDataElement(array=np.ones(self.shape) * 2, parent=self.mock_parent)
        multiplied = self.element * element_2
        expect = np.resize(np.arange(27), self.shape) * 2
        np.testing.assert_array_equal(expect, multiplied._array)
        self.assertEqual(self.mock_parent, multiplied._parent)

    def test_mul_when_other_is_numpy_array(self):
        multiplied = self.element * np.ones(self.shape) * 3
        expect = np.resize(np.arange(27), self.shape) * 3
        np.testing.assert_array_equal(expect, multiplied._array)
        self.assertEqual(self.mock_parent, multiplied._parent)

    def test_mul_when_other_is_float(self):
        multiplied = self.element * 2.
        expect = np.resize(np.arange(27), self.shape) * 2.
        np.testing.assert_array_equal(expect, multiplied._array)
        self.assertEqual(self.mock_parent, multiplied._parent)

    @patch('museek.time_ordered_data_element.np')
    def test_getitem(self, mock_np):
        self.assertEqual(mock_np.squeeze.return_value, self.element[0, 1, 2])
        mock_np.squeeze.assert_called_once_with(5)

    @patch.object(TimeOrderedDataElement, 'get_array')
    def test_scan(self, mock_get_array):
        self.element.scan
        mock_get_array.assert_called_once_with(time=self.element._parent.scan_dumps)

    @patch.object(TimeOrderedDataElement, 'get_array')
    def test_track(self, mock_get_array):
        self.element.track
        mock_get_array.assert_called_once_with(time=self.element._parent.track_dumps)

    @patch.object(TimeOrderedDataElement, 'get_array')
    def test_slew(self, mock_get_array):
        self.element.slew
        mock_get_array.assert_called_once_with(time=self.element._parent.slew_dumps)

    @patch.object(TimeOrderedDataElement, 'get_array')
    def test_stop(self, mock_get_array):
        self.element.stop
        mock_get_array.assert_called_once_with(time=self.element._parent.stop_dumps)

    @patch.object(TimeOrderedDataElement, 'get_array')
    def test_full(self, mock_get_array):
        self.element.full
        mock_get_array.assert_called_once()

    def test_mean_when_explicit(self):
        mean = self.element.mean(axis=0)
        self.assertEqual(3, len(mean._array.shape))
        expect = np.asarray([[9., 10., 11.],
                             [12., 13., 14.],
                             [15., 16., 17.]])
        np.testing.assert_array_equal(expect, mean.full)

    @patch('museek.time_ordered_data_element.np')
    def test_mean_when_mocked(self, mock_np):
        mock_np.mean.return_value.shape = (1, 1, 1)
        mock_axis = MagicMock()
        mean = self.element.mean(axis=mock_axis)
        self.assertEqual(mean._array, mock_np.mean.return_value)
        mock_np.mean.assert_called_once_with(self.element._array,
                                             axis=mock_axis,
                                             keepdims=True)

    def test_get(self):
        list_ = [1] * 9 + [2] * 9 + [3] * 9
        array = np.reshape(list_, (3, 3, 3))
        element = TimeOrderedDataElement(array=array, parent=self.mock_parent)
        np.testing.assert_array_equal(np.ones((3, 3)), element.get(time=0).full)
        np.testing.assert_array_equal(np.ones((3, 3)) * 2, element.get(time=1).full)
        expect = np.asarray([[1, 1, 1],
                             [2, 2, 2],
                             [3, 3, 3]])
        np.testing.assert_array_equal(expect, element.get(freq=1).full)
        np.testing.assert_array_equal(expect, element.get(recv=1).full)

    def test_get_when_only_time_given(self):
        shape_ = (10, 11, 3)
        element = TimeOrderedDataElement(array=np.zeros(shape_), parent=self.mock_parent)
        np.testing.assert_array_equal(np.zeros((11, 3)), element.get(time=5).full)

    def test_get_when_three_integers_given(self):
        shape_ = (10, 11, 3)
        element = TimeOrderedDataElement(array=np.zeros(shape_), parent=self.mock_parent)
        self.assertEqual(0, element.get(time=5, freq=10, recv=2).full)

    def test_get_when_lists_given(self):
        shape_ = (10, 11, 3)
        element = TimeOrderedDataElement(array=np.zeros(shape_), parent=self.mock_parent)
        np.testing.assert_array_equal(np.zeros((3, 2, 2)),
                                      element.get(time=[5, 6, 7], freq=[9, 10], recv=[0, 1]).full)

    def test_get_when_slices_given(self):
        shape_ = (10, 11, 3)
        element = TimeOrderedDataElement(array=np.zeros(shape_), parent=self.mock_parent)
        np.testing.assert_array_equal(np.zeros((10, 2, 2)),
                                      element.get(time=slice(None), freq=slice(2, 4), recv=slice(0, 2)).full)

    def test_get_when_minus_1_given(self):
        shape_ = (10, 11, 3)
        element = TimeOrderedDataElement(array=np.zeros(shape_), parent=self.mock_parent)
        np.testing.assert_array_equal(0, element.get(time=-1, freq=-1, recv=-1).full)

    @patch('museek.time_ordered_data_element.np')
    @patch.object(TimeOrderedDataElement, 'get')
    def test_get_array(self, mock_get, mock_np):
        mock_kwargs = MagicMock()
        self.assertEqual(mock_np.squeeze.return_value, self.element.get_array(**mock_kwargs))
        mock_get.assert_called_once_with(**mock_kwargs)
        mock_np.squeeze.assert_called_once_with(mock_get.return_value._array)

    def test_get_array_when_shape_1_1_1(self):
        shape_ = (1, 1, 1)
        element = TimeOrderedDataElement(array=np.zeros(shape_), parent=self.mock_parent)
        result = element.get_array()
        self.assertEqual(0., result)
        self.assertIsInstance(result, float)
