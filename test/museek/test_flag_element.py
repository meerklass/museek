import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from museek.data_element import DataElement
from museek.flag_element import FlagElement


class TestFlagElement(unittest.TestCase):
    def setUp(self):
        flags = [DataElement(array=np.zeros((3, 3, 3))) for _ in range(3)]
        self.flag_element = FlagElement(flags=flags)

    def test_add_flag(self):
        self.flag_element.add_flag(flag=DataElement(array=np.zeros((3, 3, 3))))
        flags = [DataElement(array=np.zeros((3, 3, 3))) for _ in range(4)]
        expect = FlagElement(flags=flags)
        self.assertEqual(expect, self.flag_element)

    def test_combine_when_empty(self):
        self.assertEqual(DataElement(array=np.zeros((3, 3, 3))), self.flag_element.combine())

    def test_combine_when_ones_and_threshold_small(self):
        flags = [DataElement(array=np.ones((3, 3, 3))) for _ in range(3)]
        flag_element = FlagElement(flags=flags)
        self.assertEqual(DataElement(array=np.ones((3, 3, 3))), flag_element.combine(threshold=1))

    def test_combine_when_ones_and_threshold_large(self):
        flags = [DataElement(array=np.ones((3, 3, 3))) for _ in range(3)]
        flag_element = FlagElement(flags=flags)
        self.assertEqual(DataElement(array=np.zeros((3, 3, 3))), flag_element.combine(threshold=4))

    def test_combine_when_different_flags_and_threshold_small(self):
        flags = [DataElement(array=np.ones((3, 3, 3))),
                 DataElement(array=np.zeros((3, 3, 3))),
                 DataElement(array=np.ones((3, 3, 3)))]
        flag_element = FlagElement(flags=flags)
        self.assertEqual(DataElement(array=np.ones((3, 3, 3))), flag_element.combine(threshold=1))

    def test_combine_when_different_flags_and_threshold_large(self):
        flags = [DataElement(array=np.ones((3, 3, 3))),
                 DataElement(array=np.zeros((3, 3, 3))),
                 DataElement(array=np.ones((3, 3, 3)))]
        flag_element = FlagElement(flags=flags)
        self.assertEqual(DataElement(array=np.zeros((3, 3, 3))), flag_element.combine(threshold=3))

    def test_combine_when_one_dump_flagged(self):
        flag_1 = DataElement(array=np.zeros((3, 3, 3)))
        array_ = np.zeros((3, 3, 3))
        array_[1, 1, 1] = 1
        flag_2 = DataElement(array=array_)

        flag_element = FlagElement(flags=[flag_1, flag_2])
        self.assertEqual(flag_2, flag_element.combine(threshold=1))

    @patch.object(FlagElement, '_check_flags')
    def test_get(self, mock_check_flags):
        mock_flag = MagicMock(shape=1)
        mock_flag.get.return_value = mock_flag
        flag_element = FlagElement(flags=[mock_flag])
        kwargs = {'mock': 'mock'}
        self.assertEqual(flag_element._flags[0], flag_element.get(**kwargs)._flags[0])
        mock_flag.get.assert_called_once_with(mock='mock')
        self.assertEqual(2, mock_check_flags.call_count)

    @patch.object(FlagElement, '_check_flag_types')
    @patch.object(FlagElement, '_check_flag_shapes')
    def test_check_flags(self, mock_check_flag_shapes, mock_check_flag_types):
        self.flag_element._check_flags()
        mock_check_flag_shapes.assert_called_once()
        mock_check_flag_types.assert_called_once()

    def test_check_flag_shapes(self):
        self.assertIsNone(self.flag_element._check_flag_shapes())

    def test_check_flag_shapes_expect_raise(self):
        flags = [DataElement(array=np.zeros((3, 3, 3))), DataElement(array=np.zeros((1, 1, 1)))]
        self.assertRaises(ValueError, FlagElement, flags=flags)

    def test_check_flag_types(self):
        self.assertIsNone(self.flag_element._check_flag_types())

    def test_check_flag_types_expect_raise(self):
        flags = [DataElement(array=np.zeros((3, 3, 3))), np.zeros((1, 1, 1))]
        self.assertRaises(ValueError, FlagElement, flags=flags)
