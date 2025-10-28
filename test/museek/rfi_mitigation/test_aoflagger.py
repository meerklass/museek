import unittest
from unittest.mock import patch, Mock, call

import numpy as np

from museek.factory.data_element_factory import FlagElementFactory
from museek.rfi_mitigation.aoflagger import _sum_threshold_mask, \
    _run_sumthreshold, _apply_kernel, \
    gaussian_filter, get_rfi_mask


class TestAoflagger(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.data = np.random.normal(loc=300, scale=10, size=(200, 100, 10))
        # add some rfi
        self.data[50:60, :] = np.random.normal(loc=3000, scale=10, size=(10, 100, 10))
        self.data[:, 40:45] = np.random.normal(loc=2000, scale=10, size=(200, 5, 10))
        self.mask = np.zeros_like(self.data, dtype=bool)
        # create a masked frame
        self.mask[:12, :, :] = True
        self.mask[-12:, :, :] = True
        self.mask[:, :10, :] = True
        self.mask[:, -10:, :] = True

    @patch.object(np, 'arange')
    @patch.object(FlagElementFactory, 'create')
    @patch('museek.rfi_mitigation.aoflagger.plot_moments')
    @patch('museek.rfi_mitigation.aoflagger._run_sumthreshold')
    def test_get_rfi_mask(self, mock_run_sumthreshold, mock_plot_moments, mock_create, mock_arange):
        mock_data = Mock()
        mock_mask = Mock()
        mock_window_size = Mock()
        mock_sigma = Mock()

        mock_arange.return_value = 1

        rfi_mask = get_rfi_mask(time_ordered=mock_data,
                                mask=mock_mask,
                                first_threshold=1.,
                                threshold_scales=[0.5, 1],
                                smoothing_window_size=mock_window_size,
                                smoothing_sigma=mock_sigma,
                                output_path='output_path')
        mock_run_sumthreshold.assert_has_calls([
            call(data=mock_data.squeeze,
                 initial_mask=mock_mask.squeeze,
                 threshold_scale=0.5,
                 n_iterations=1,
                 thresholds=1,
                 output_path='output_path',
                 smoothing_window_size=mock_window_size,
                 smoothing_sigma=mock_sigma),
            call(data=mock_data.squeeze,
                 initial_mask=mock_run_sumthreshold.return_value,
                 threshold_scale=1.,
                 n_iterations=1,
                 thresholds=1,
                 output_path='output_path',
                 smoothing_window_size=mock_window_size,
                 smoothing_sigma=mock_sigma)
        ])
        self.assertEqual(mock_create.return_value, rfi_mask)
        mock_create.assert_called_once_with(array=mock_run_sumthreshold.return_value[:, :, np.newaxis])
        mock_plot_moments.assert_called_once_with(mock_data.squeeze, 'output_path')

    @patch('museek.rfi_mitigation.aoflagger._apply_kernel')
    def test_gaussian_filter(self, mock_apply_kernel):
        mock_array = Mock()
        mock_mask = Mock()
        filtered = gaussian_filter(array=mock_array,
                                   mask=mock_mask,
                                   window_size=(2, 2),
                                   sigma=(1, 1))
        self.assertEqual(mock_apply_kernel.return_value, filtered)
        e_to_minus_1_half = 0.60653066
        kernel = (np.array([e_to_minus_1_half, 1, e_to_minus_1_half]),
                  np.array([e_to_minus_1_half, 1, e_to_minus_1_half]))
        call_args_dict = mock_apply_kernel.call_args.kwargs
        self.assertEqual(call_args_dict['array'], mock_array)
        self.assertEqual(call_args_dict['mask'], mock_mask)
        self.assertTupleEqual(call_args_dict['even_window_size'], (2, 2))
        for i in range(2):
            np.testing.assert_array_almost_equal(call_args_dict['kernel'][i], kernel[i])

    def test_apply_kernel_when_unity_kernel_expect_identity(self):
        array = np.array([[0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 1, 9, 3, 4],
                          [0, 1, 2, 3, 4],
                          [0, 0, 0, 0, 0]])
        mask = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],
                        dtype=bool)
        kernel = (np.array([0, 1, 0]), np.array([0, 1, 0]))
        smoothed = _apply_kernel(array=array,
                                 mask=mask,
                                 kernel=kernel,
                                 even_window_size=(2, 2))
        np.testing.assert_array_equal(array, smoothed)

    def test_apply_kernel_when_kernel_only_y_direction(self):
        array = np.array([[0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 1, 9, 3, 4],
                          [0, 1, 2, 3, 4],
                          [0, 0, 0, 0, 0]])
        mask = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],
                        dtype=bool)
        kernel = (np.array([0, 1, 0]), np.array([1, 1, 1]))
        smoothed = _apply_kernel(array=array,
                                 mask=mask,
                                 kernel=kernel,
                                 even_window_size=(2, 2))
        expect = np.array([[0., 0., 0., 0., 0., ],
                           [0.5, 0.33333333, 0.33333333, 0., 0., ],
                           [0.5, 0.5, 9., 3.5, 3.5, ],
                           [0.5, 1., 2., 3., 3.5, ],
                           [0., 0., 0., 0., 0., ]])
        np.testing.assert_array_almost_equal(expect, smoothed)

    def test_apply_kernel_when_kernel_only_x_direction(self):
        array = np.array([[0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 1, 9, 3, 4],
                          [0, 1, 2, 3, 4],
                          [0, 0, 0, 0, 0]]).T
        mask = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],
                        dtype=bool).T
        kernel = (np.array([1, 1, 1]), np.array([0, 1, 0]))
        smoothed = _apply_kernel(array=array,
                                 mask=mask,
                                 kernel=kernel,
                                 even_window_size=(2, 2))
        expect = np.array([[0., 0., 0., 0., 0., ],
                           [0.5, 0.33333333, 0.33333333, 0., 0., ],
                           [0.5, 0.5, 9., 3.5, 3.5, ],
                           [0.5, 1., 2., 3., 3.5, ],
                           [0., 0., 0., 0., 0., ]]).T
        np.testing.assert_array_almost_equal(expect, smoothed)

    def test_run_sumthreshold(self):
        data = self.data[:, :, 0]
        mask = self.mask[:, :, 0]
        sum_threshold = _run_sumthreshold(data=data,
                                          initial_mask=mask,
                                          threshold_scale=1.,
                                          n_iterations=[1, 2],
                                          thresholds=[0.2, 0.4],
                                          smoothing_window_size=(32, 32),
                                          smoothing_sigma=(10, 10),
                                          output_path=None)
        clean = np.ma.array(data=data, mask=sum_threshold)
        mean_clean = np.mean(clean)
        self.assertAlmostEqual(mean_clean, 300, 1)
        rfi = np.ma.array(data=data, mask=~(sum_threshold ^ mask))
        mean_rfi = np.mean(rfi)
        self.assertGreater(mean_rfi, 2000)

    def test_sum_threshold_mask(self):
        data = self.data[:, :, 0]
        mask = self.mask[:, :, 0]
        sum_threshold = _sum_threshold_mask(data=data,
                                            mask=mask,
                                            n_iteration=2,
                                            threshold=1000)
        clean = np.ma.array(data=data, mask=sum_threshold)
        mean_clean = np.mean(clean)
        self.assertAlmostEqual(mean_clean, 300, 1)
        rfi = np.ma.array(data=data, mask=~(sum_threshold ^ mask))
        mean_rfi = np.mean(rfi)
        self.assertGreater(mean_rfi, 2000)
