import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np

from museek.flag_element import FlagElement


class TestFlagElement(unittest.TestCase):
    def setUp(self):
        self.shape = (3, 3, 3)
        self.mock_parent = MagicMock()
        self.array = np.ones((3, 3, 3), dtype=bool)
        self.element = FlagElement(array=self.array)

    def test_add_when_both_empty(self):
        flag = FlagElement(
            array=np.asarray(
                [[[False, False, False], [False, False, False], [False, False, False]]]
            )
        )
        self.assertEqual(flag + flag, flag)

    def test_add_when_both_full(self):
        flag = FlagElement(
            array=np.asarray(
                [[[True, True, True], [True, True, True], [True, True, True]]]
            )
        )
        self.assertEqual(flag + flag, flag)

    def test_add_when_one_empty_one_full(self):
        flag_1 = FlagElement(
            array=np.asarray(
                [[[False, False, False], [False, False, False], [False, False, False]]]
            )
        )
        flag_2 = FlagElement(
            array=np.asarray(
                [[[True, True, True], [True, True, True], [True, True, True]]]
            )
        )
        self.assertEqual(flag_1 + flag_2, flag_2)

    def test_add_when_complementary(self):
        flag_1 = FlagElement(
            array=np.asarray(
                [[[True, False, False], [False, False, True], [False, True, False]]]
            )
        )
        flag_2 = FlagElement(
            array=np.asarray(
                [[[False, True, True], [True, True, False], [True, False, True]]]
            )
        )
        expect = FlagElement(
            array=np.asarray(
                [[[True, True, True], [True, True, True], [True, True, True]]]
            )
        )
        self.assertEqual(flag_1 + flag_2, expect)

    @patch("museek.flag_element.DataElement")
    @patch("museek.flag_element.np")
    def test_sum(self, mock_np, mock_data_element):
        mock_axis = MagicMock()
        mean = self.element.sum(axis=mock_axis)
        mock_np.sum.assert_called_once_with(
            self.element.array, axis=mock_axis, keepdims=True
        )
        mock_data_element.assert_called_once_with(array=mock_np.sum.return_value)
        self.assertEqual(mean, mock_data_element.return_value)

    @patch.object(np, "logical_or")
    def test_insert_receiver_flag(self, mock_logical_or):
        mock_flag = MagicMock()
        type(mock_flag).shape = PropertyMock(return_value=(3, 3, 1))
        expect = np.zeros((3, 3))
        mock_logical_or.return_value = expect
        self.element.insert_receiver_flag(flag=mock_flag, i_receiver=0)
        np.testing.assert_array_equal(self.element.array[:, :, 0], expect)

    def test_make_boolean_when_true(self):
        result = FlagElement._make_boolean(array=np.array([1]))
        self.assertTrue(isinstance(result[0], bool))
        np.testing.assert_array_equal(np.array([True]), result)

    def test_make_boolean_when_false(self):
        result = FlagElement._make_boolean(array=np.array([0]))
        self.assertTrue(isinstance(result[0], bool))
        np.testing.assert_array_equal(
            np.array([False]), FlagElement._make_boolean(array=np.array([0]))
        )

    def test_make_boolean_when_not_binary_expect_raise(self):
        self.assertRaises(ValueError, FlagElement._make_boolean, np.array([1, 2, 3]))
