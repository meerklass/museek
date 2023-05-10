import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from museek.data_element import DataElement
from museek.factory.data_element_factory import DataElementFactory
from museek.flag_element import FlagElement


class TestFlagElement(unittest.TestCase):
    def setUp(self):
        flags = [DataElement(array=np.zeros((3, 3, 3))) for _ in range(3)]
        self.flag_element = FlagElement(flags=flags)

    def test_len(self):
        self.assertEqual(3, len(self.flag_element))

    def test_eq(self):
        flags = [DataElement(array=np.zeros((3, 3, 3))) for _ in range(3)]
        self.assertEqual(self.flag_element, FlagElement(flags=flags))

    def test_eq_when_not_equal(self):
        flags = [DataElement(array=np.ones((3, 3, 3))) for _ in range(3)]
        self.assertNotEqual(self.flag_element, FlagElement(flags=flags))

    def test_eq_when_more_flags_expect_not_equal(self):
        flags = [DataElement(array=np.zeros((3, 3, 3))) for _ in range(4)]
        self.assertNotEqual(self.flag_element, FlagElement(flags=flags))

    def test_eq_when_different_shape_expect_not_equal(self):
        flags = [DataElement(array=np.zeros((3, 4, 3))) for _ in range(3)]
        self.assertNotEqual(self.flag_element, FlagElement(flags=flags))

    def test_from_array(self):
        flag_array = np.zeros((3, 3, 3, 3))
        self.assertEqual(self.flag_element, FlagElement.from_array(flag_array, element_factory=DataElementFactory()))

    def test_from_array_when_3_dimensional(self):
        flag_array = np.zeros((3, 3, 3))
        flags = [DataElement(array=np.zeros((3, 3, 3))) for _ in range(1)]
        flag_element = FlagElement(flags=flags)
        self.assertEqual(flag_element, FlagElement.from_array(flag_array, element_factory=DataElementFactory()))

    def test_shape(self):
        self.assertTupleEqual((3, 3, 3), self.flag_element.shape)

    def test_add_flag(self):
        self.flag_element.add_flag(flag=DataElement(array=np.zeros((3, 3, 3))))
        flags = [DataElement(array=np.zeros((3, 3, 3))) for _ in range(4)]
        expect = FlagElement(flags=flags)
        self.assertEqual(expect, self.flag_element)

    def test_add_flag_when_flag_element(self):
        mock_flags = FlagElement(flags=[DataElement(array=np.zeros((3, 3, 3)))])
        self.flag_element.add_flag(flag=mock_flags)
        np.testing.assert_array_equal(mock_flags._flags[0]._array, self.flag_element._flags[0]._array)

    def test_remove_flag(self):
        flag_element = FlagElement(flags=[DataElement(array=np.ones((3, 3, 3)) * i) for i in range(3)])
        flag_element.remove_flag(index=1)
        expect = FlagElement(flags=[DataElement(array=np.ones((3, 3, 3)) * i) for i in [0, 2]])
        self.assertEqual(expect, flag_element)

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

    def test_insert_receiver_flag_when_flag_shape_incorrect_expect_value_error(self):
        mock_flag = DataElement(array=np.ones((3, 3, 2)))
        self.assertRaises(ValueError,
                          self.flag_element.insert_receiver_flag,
                          flag=mock_flag,
                          i_receiver=1,
                          index=2)

    def test_insert_receiver_flag(self):
        mock_flag = DataElement(array=np.ones((3, 3, 1), dtype=bool))
        self.flag_element.insert_receiver_flag(flag=mock_flag, i_receiver=1, index=2)
        self.assertTrue((self.flag_element._flags[0]._array == False).all())
        self.assertTrue((self.flag_element._flags[2].get(recv=0).squeeze == False).all())
        self.assertTrue((self.flag_element._flags[2].get(recv=2).squeeze == False).all())
        self.assertTrue(self.flag_element._flags[2].get(recv=1).squeeze.all())

    def test_insert_receiver_flag_when_one_channel(self):
        flags = [DataElement(array=np.zeros((3, 1, 3))) for _ in range(3)]
        flag_element = FlagElement(flags=flags)

        mock_flag = DataElement(array=np.ones((3, 1, 1), dtype=bool))
        flag_element.insert_receiver_flag(flag=mock_flag, i_receiver=1, index=2)
        self.assertTrue((flag_element._flags[0]._array == False).all())
        self.assertTrue((flag_element._flags[2].get(recv=0).squeeze == False).all())
        self.assertTrue((flag_element._flags[2].get(recv=2).squeeze == False).all())
        self.assertTrue(flag_element._flags[2].get(recv=1).squeeze.all())

    def test_array(self):
        expect = np.zeros((3, 3, 3, 3))
        np.testing.assert_array_equal(expect, self.flag_element.array())

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
