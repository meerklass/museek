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
        mock_axis = MagicMock()
        mean = self.element.mean(axis=mock_axis)
        self.assertEqual(mean._array, mock_np.mean.return_value)
        mock_np.mean.assert_called_once_with(self.element._array,
                                             axis=mock_axis,
                                             keepdims=True)

    def test_get(self):
        mock_array = MagicMock()
        element = TimeOrderedDataElement(array=mock_array, parent=self.mock_parent)
        mock_time = MagicMock()
        mock_freq = MagicMock()
        mock_recv = MagicMock()
        element_get = element.get(time=mock_time, freq=mock_freq, recv=mock_recv)
        self.assertEqual(self.mock_parent, element_get._parent)
        self.assertEqual(mock_array.copy.return_value.__getitem__.return_value, element_get._array)
        mock_array.copy.return_value.__getitem__.assert_called_once_with((mock_time, mock_freq, mock_recv))

    @patch('museek.time_ordered_data_element.isinstance')
    def test_get_when_integers(self, mock_isinstance):
        mock_isinstance.return_value = True
        mock_array = MagicMock()
        element = TimeOrderedDataElement(array=mock_array, parent=self.mock_parent)
        mock_time = MagicMock()
        mock_freq = MagicMock()
        mock_recv = MagicMock()
        element_get = element.get(time=mock_time, freq=mock_freq, recv=mock_recv)
        self.assertEqual(self.mock_parent, element_get._parent)
        self.assertEqual(mock_array.copy.return_value.__getitem__.return_value, element_get._array)
        mock_array.copy.return_value.__getitem__.assert_called_once_with(([mock_time], [mock_freq], [mock_recv]))

    def test_get_when_only_time_given(self):
        mock_array = MagicMock()
        element = TimeOrderedDataElement(array=mock_array, parent=self.mock_parent)
        mock_time = MagicMock()
        element_get = element.get(time=mock_time)
        self.assertEqual(self.mock_parent, element_get._parent)
        self.assertEqual(mock_array.copy.return_value.__getitem__.return_value, element_get._array)
        mock_array.copy.return_value.__getitem__.assert_called_once_with((mock_time, slice(None), slice(None)))

    def test_get_when_both_recv_and_ants_is_given_expect_raise(self):
        mock_array = MagicMock()
        element = TimeOrderedDataElement(array=mock_array, parent=self.mock_parent)
        mock_ants = MagicMock()
        mock_recv = MagicMock()
        self.assertRaises(ValueError, element.get, ants=mock_ants, recv=mock_recv)

    @patch('museek.time_ordered_data_element.np')
    @patch.object(TimeOrderedDataElement, 'get')
    def test_get_array(self, mock_get, mock_np):
        mock_kwargs = MagicMock()
        self.assertEqual(mock_np.squeeze.return_value, self.element.get_array(**mock_kwargs))
        mock_get.assert_called_once_with(**mock_kwargs)
        mock_np.squeeze.assert_called_once_with(mock_get.return_value._array)
