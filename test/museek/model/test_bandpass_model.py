import unittest
from unittest.mock import patch, Mock, MagicMock

import numpy as np
import scipy

from museek.factory.data_element_factory import DataElementFactory
from museek.model.bandpass_model import BandpassModel


class TestBandpassModel(unittest.TestCase):

    def setUp(self):
        self.bandpass_model = BandpassModel(standing_wave_displacements=[1., 2.], legendre_degree=2, plot_name=None)

    @patch('museek.model.bandpass_model.legendre')
    @patch.object(BandpassModel, '_parameters_to_dictionary')
    @patch.object(BandpassModel, '_bandpass_model')
    @patch.object(BandpassModel, '_sinus_parameter_list')
    @patch.object(scipy, 'optimize')
    def test_fit(self,
                 mock_optimize,
                 mock_sinus_parameter_list,
                 mock_bandpass_model,
                 mock_parameters_to_dictionary,
                 mock_legendre):
        frequency_array = np.arange(1000, 1100, 10) * 1e6
        frequencies = DataElementFactory().create(array=frequency_array[np.newaxis, :, np.newaxis])
        wavelength = 2.
        mock_estimator_array = 300 + 0.1 * frequency_array + 0.1 * np.sin(2 * np.pi / (3e8 / 1e6 / wavelength))
        mock_estimator = DataElementFactory().create(array=mock_estimator_array[np.newaxis, :, np.newaxis])
        mock_optimize.curve_fit.return_value = (MagicMock(), MagicMock())
        self.bandpass_model.fit(frequencies=frequencies,
                                estimator=mock_estimator,
                                receiver_path='',
                                calibrator_label='b')
        self.assertEqual(mock_parameters_to_dictionary.return_value, self.bandpass_model.parameters_dictionary)
        self.assertEqual(mock_parameters_to_dictionary.return_value, self.bandpass_model.variances_dictionary)
        self.assertEqual(mock_bandpass_model.return_value.__truediv__.return_value.__sub__.return_value,
                         self.bandpass_model.epsilon)
        mock_legendre.legfit.assert_called_once()
        mock_sinus_parameter_list.assert_called_once()

    @patch('museek.model.bandpass_model.sum')
    @patch('museek.model.bandpass_model.legendre')
    @patch.object(BandpassModel, '_sinus')
    def test_bandpass_model(self, mock_sinus, mock_legendre, mock_sum):
        mock_frequencies = Mock()
        mock_legendre_coefficients = Mock()
        mock_sinus_parameters_list = [Mock()]
        model = self.bandpass_model._bandpass_model(frequencies=mock_frequencies,
                                                    legendre_coefficients=mock_legendre_coefficients,
                                                    sinus_parameters_list=mock_sinus_parameters_list)
        mock_sum.assert_called_once_with([mock_sinus.return_value])
        mock_legendre.legval.assert_called_once_with(mock_frequencies, mock_legendre_coefficients)
        mock_sinus.assert_called_once_with(mock_frequencies, mock_sinus_parameters_list[0])
        self.assertEqual(model, mock_legendre.legval.return_value.__mul__.return_value)

    def test_sinus_expect_zero(self):
        sinus_ = self.bandpass_model._sinus(frequencies=np.asarray([0]),
                                            parameters=(0, 3, 1))
        self.assertEqual(sinus_[0], 0)

    def test_sinus_expect_1(self):
        sinus_ = self.bandpass_model._sinus(frequencies=np.asarray([np.pi / 4 * 1e6]),
                                            parameters=(2 * np.pi, 1, 1))
        self.assertAlmostEqual(sinus_[0], 0, 1)

    def test_sinus_parameter_list(self):
        mock_parameters = [1, 2, 3, 4, 5, 6]
        parameter_list = self.bandpass_model._sinus_parameter_list(parameters=mock_parameters,
                                                                   n_legendre_coefficients=2)
        self.assertTupleEqual((3, 4, 2.0), parameter_list[0])
        self.assertTupleEqual((5, 6, 4.0), parameter_list[1])

    def test_parameters_to_dictionary(self):
        mock_parameters = [1, 2, 3, 4, 5, 6]
        dict_ = self.bandpass_model._parameters_to_dictionary(parameters=mock_parameters,
                                                              n_legendre_coefficients=2)
        expect = {'l_0': 1,
                  'l_1': 2,
                  'wavelength_2.0_phase': 3,
                  'wavelength_2.0_amplitude': 4,
                  'wavelength_4.0_phase': 5,
                  'wavelength_4.0_amplitude': 6}
        self.assertDictEqual(expect, dict_)

    def test_check_parameters(self):
        mock_parameters = [1, 2, 3, 4, 5, 6]
        self.assertIsNone(self.bandpass_model._check_parameters(
            parameters=mock_parameters,
            n_legendre_coefficients=2,
        ))

    def test_check_parameters_expect_value_error(self):
        mock_parameters = [1, 2, 3, 4, 5]
        self.assertRaises(ValueError,
                          self.bandpass_model._check_parameters,
                          parameters=mock_parameters,
                          n_legendre_coefficients=2)
