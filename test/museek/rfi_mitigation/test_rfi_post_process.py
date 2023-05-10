import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import scipy

from museek.data_element import DataElement
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess


class TestRfiPostProcess(unittest.TestCase):

    @patch('museek.rfi_mitigation.rfi_post_process.FlagElementFactory')
    def setUp(self, mock_flag_element_factory):
        self.mock_struct_size = MagicMock()
        self.mock_new_flag = MagicMock()
        self.mock_initial_flag = MagicMock()
        self.rfi_post_process = RfiPostProcess(new_flag=self.mock_new_flag,
                                               initial_flag=self.mock_initial_flag,
                                               struct_size=self.mock_struct_size)
        self.mock_data_element_factory = mock_flag_element_factory.return_value

    def test_get_flag(self):
        self.assertEqual(self.mock_new_flag, self.rfi_post_process.get_flag())

    @patch.object(scipy.ndimage, 'binary_dilation')
    def test_binary_mask_dilation(self, mock_binary_dilation):
        self.rfi_post_process.binary_mask_dilation()
        self.mock_data_element_factory.create.assert_called_once_with(
            array=mock_binary_dilation.return_value.__getitem__.return_value
        )
        self.assertEqual(self.mock_data_element_factory.create.return_value,
                         self.rfi_post_process.get_flag())
        mock_binary_dilation.assert_called_once_with(self.mock_new_flag.squeeze.__xor__.return_value,
                                                     structure=self.rfi_post_process._struct,
                                                     iterations=5)

    @patch.object(scipy.ndimage, 'binary_closing')
    def test_binary_mask_closing(self, mock_binary_closing):
        self.rfi_post_process.binary_mask_closing()
        self.mock_data_element_factory.create.assert_called_once_with(
            array=mock_binary_closing.return_value.__getitem__.return_value
        )
        self.assertEqual(self.mock_data_element_factory.create.return_value,
                         self.rfi_post_process.get_flag())
        mock_binary_closing.assert_called_once_with(self.mock_new_flag.squeeze,
                                                    structure=self.rfi_post_process._struct,
                                                    iterations=5)

    def test_flag_all_channels(self):
        mock_flag_array = np.array([[[1], [0], [0]],
                                    [[1], [0], [1]],
                                    [[0], [1], [1]]], dtype=bool)
        mock_new_flag = DataElement(array=mock_flag_array)
        rfi_post_process = RfiPostProcess(new_flag=mock_new_flag,
                                          initial_flag=None,
                                          struct_size=self.mock_struct_size)
        rfi_post_process.flag_all_channels(channel_flag_threshold=0.5)
        expect = np.array([[1, 0, 0],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=bool)
        np.testing.assert_array_equal(expect, rfi_post_process.get_flag().squeeze)

    def test_flag_all_time_dumps(self):
        mock_flag_array = np.array([[[1], [0], [0]],
                                    [[1], [0], [1]],
                                    [[0], [1], [1]]], dtype=bool)
        mock_new_flag = DataElement(array=mock_flag_array)
        rfi_post_process = RfiPostProcess(new_flag=mock_new_flag,
                                          initial_flag=None,
                                          struct_size=self.mock_struct_size)
        rfi_post_process.flag_all_time_dumps(time_dump_flag_threshold=0.5)
        expect = np.array([[1, 0, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=bool)
        np.testing.assert_array_equal(expect, rfi_post_process.get_flag().squeeze)
