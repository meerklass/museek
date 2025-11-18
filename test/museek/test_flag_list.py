import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_element import FlagElement
from museek.flag_list import FlagList


class TestFlagList(unittest.TestCase):
    def setUp(self):
        flags = [FlagElement(array=np.zeros((3, 3, 3))) for _ in range(3)]
        self.flag_list = FlagList(flags=flags)

    def test_len(self):
        self.assertEqual(3, len(self.flag_list))

    def test_eq(self):
        flags = [FlagElement(array=np.zeros((3, 3, 3))) for _ in range(3)]
        self.assertEqual(self.flag_list, FlagList(flags=flags))

    def test_eq_when_not_equal(self):
        flags = [FlagElement(array=np.ones((3, 3, 3))) for _ in range(3)]
        self.assertNotEqual(self.flag_list, FlagList(flags=flags))

    def test_eq_when_more_flags_expect_not_equal(self):
        flags = [FlagElement(array=np.zeros((3, 3, 3))) for _ in range(4)]
        self.assertNotEqual(self.flag_list, FlagList(flags=flags))

    def test_eq_when_different_shape_expect_not_equal(self):
        flags = [FlagElement(array=np.zeros((3, 4, 3))) for _ in range(3)]
        self.assertNotEqual(self.flag_list, FlagList(flags=flags))

    def test_from_array(self):
        flag_array = np.zeros((3, 3, 3, 3))
        self.assertEqual(
            self.flag_list,
            FlagList.from_array(flag_array, element_factory=FlagElementFactory()),
        )

    def test_from_array_when_3_dimensional(self):
        flag_array = np.zeros((3, 3, 3))
        flags = [FlagElement(array=np.zeros((3, 3, 3))) for _ in range(1)]
        flag_list = FlagList(flags=flags)
        self.assertEqual(
            flag_list,
            FlagList.from_array(flag_array, element_factory=FlagElementFactory()),
        )

    def test_shape(self):
        self.assertTupleEqual((3, 3, 3), self.flag_list.shape)

    def test_add_flag(self):
        self.flag_list.add_flag(flag=FlagElement(array=np.zeros((3, 3, 3))))
        flags = [FlagElement(array=np.zeros((3, 3, 3))) for _ in range(4)]
        expect = FlagList(flags=flags)
        self.assertEqual(expect, self.flag_list)

    def test_add_flag_when_flag_element(self):
        mock_flags = FlagList(flags=[FlagElement(array=np.zeros((3, 3, 3)))])
        self.flag_list.add_flag(flag=mock_flags)
        np.testing.assert_array_equal(
            mock_flags._flags[0].array, self.flag_list._flags[0].array
        )

    def test_remove_flag(self):
        flag_list = FlagList(
            flags=[FlagElement(array=np.ones((3, 3, 3), dtype=bool)) for _ in range(3)]
        )
        flag_list.remove_flag(index=1)
        expect = FlagList(
            flags=[FlagElement(array=np.ones((3, 3, 3), dtype=bool)) for _ in [0, 2]]
        )
        self.assertEqual(expect, flag_list)

    def test_combine_when_empty(self):
        self.assertEqual(
            FlagElement(array=np.zeros((3, 3, 3))), self.flag_list.combine()
        )

    def test_combine_when_ones_and_threshold_small(self):
        flags = [FlagElement(array=np.ones((3, 3, 3))) for _ in range(3)]
        flag_list = FlagList(flags=flags)
        self.assertEqual(
            FlagElement(array=np.ones((3, 3, 3))), flag_list.combine(threshold=1)
        )

    def test_combine_when_ones_and_threshold_large(self):
        flags = [FlagElement(array=np.ones((3, 3, 3))) for _ in range(3)]
        flag_list = FlagList(flags=flags)
        self.assertEqual(
            FlagElement(array=np.zeros((3, 3, 3))), flag_list.combine(threshold=4)
        )

    def test_combine_when_different_flags_and_threshold_small(self):
        flags = [
            FlagElement(array=np.ones((3, 3, 3))),
            FlagElement(array=np.zeros((3, 3, 3))),
            FlagElement(array=np.ones((3, 3, 3))),
        ]
        flag_list = FlagList(flags=flags)
        self.assertEqual(
            FlagElement(array=np.ones((3, 3, 3))), flag_list.combine(threshold=1)
        )

    def test_combine_when_different_flags_and_threshold_large(self):
        flags = [
            FlagElement(array=np.ones((3, 3, 3))),
            FlagElement(array=np.zeros((3, 3, 3))),
            FlagElement(array=np.ones((3, 3, 3))),
        ]
        flag_list = FlagList(flags=flags)
        self.assertEqual(
            FlagElement(array=np.zeros((3, 3, 3))), flag_list.combine(threshold=3)
        )

    def test_combine_when_one_dump_flagged(self):
        flag_1 = FlagElement(array=np.zeros((3, 3, 3)))
        array_ = np.zeros((3, 3, 3))
        array_[1, 1, 1] = 1
        flag_2 = FlagElement(array=array_)

        flag_list = FlagList(flags=[flag_1, flag_2])
        self.assertEqual(flag_2, flag_list.combine(threshold=1))

    @patch.object(FlagList, "_check_flags")
    def test_get(self, mock_check_flags):
        mock_flag = MagicMock(shape=1)
        mock_flag.get.return_value = mock_flag
        flag_list = FlagList(flags=[mock_flag])
        kwargs = {"mock": "mock"}
        self.assertEqual(flag_list._flags[0], flag_list.get(**kwargs)._flags[0])
        mock_flag.get.assert_called_once_with(mock="mock")
        self.assertEqual(2, mock_check_flags.call_count)

    def test_insert_receiver_flag_when_flag_shape_incorrect_expect_value_error(self):
        mock_flag = FlagElement(array=np.ones((3, 3, 2)))
        self.assertRaises(
            ValueError,
            self.flag_list.insert_receiver_flag,
            flag=mock_flag,
            i_receiver=1,
            index=2,
        )

    def test_insert_receiver_flag(self):
        mock_flag = FlagElement(array=np.ones((3, 3, 1), dtype=bool))
        self.flag_list.insert_receiver_flag(flag=mock_flag, i_receiver=1, index=2)
        self.assertTrue((self.flag_list._flags[0].array == False).all())
        self.assertTrue((self.flag_list._flags[2].get(recv=0).squeeze == False).all())
        self.assertTrue((self.flag_list._flags[2].get(recv=2).squeeze == False).all())
        self.assertTrue(self.flag_list._flags[2].get(recv=1).squeeze.all())

    def test_insert_receiver_flag_when_one_channel(self):
        flags = [FlagElement(array=np.zeros((3, 1, 3))) for _ in range(3)]
        flag_list = FlagList(flags=flags)

        mock_flag = FlagElement(array=np.ones((3, 1, 1), dtype=bool))
        flag_list.insert_receiver_flag(flag=mock_flag, i_receiver=1, index=2)
        self.assertTrue((flag_list._flags[0].array == False).all())
        self.assertTrue((flag_list._flags[2].get(recv=0).squeeze == False).all())
        self.assertTrue((flag_list._flags[2].get(recv=2).squeeze == False).all())
        self.assertTrue(flag_list._flags[2].get(recv=1).squeeze.all())

    def test_array(self):
        expect = np.zeros((3, 3, 3, 3))
        np.testing.assert_array_equal(expect, self.flag_list.array)

    @patch.object(FlagList, "_check_flag_types")
    @patch.object(FlagList, "_check_flag_shapes")
    def test_check_flags(self, mock_check_flag_shapes, mock_check_flag_types):
        self.flag_list._check_flags()
        mock_check_flag_shapes.assert_called_once()
        mock_check_flag_types.assert_called_once()

    def test_check_flag_shapes(self):
        self.assertIsNone(self.flag_list._check_flag_shapes())

    def test_check_flag_shapes_expect_raise(self):
        flags = [
            FlagElement(array=np.zeros((3, 3, 3))),
            FlagElement(array=np.zeros((1, 1, 1))),
        ]
        self.assertRaises(ValueError, FlagList, flags=flags)

    def test_check_flag_types(self):
        self.assertIsNone(self.flag_list._check_flag_types())

    def test_check_flag_types_expect_raise(self):
        flags = [FlagElement(array=np.zeros((3, 3, 3))), np.zeros((1, 1, 1))]
        self.assertRaises(ValueError, FlagList, flags=flags)
