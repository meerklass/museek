import unittest
from unittest.mock import Mock, patch

import numpy as np

from museek.data_element import DataElement
from museek.noise_diode import NoiseDiode


class TestNoiseDiode(unittest.TestCase):
    @patch.object(NoiseDiode, "_get_noise_diode_settings")
    def setUp(self, mock_get_noise_diode_settings):
        self.mock_dump_period = Mock()
        self.mock_observation_log = ["observation_log"]
        mock_get_noise_diode_settings.return_value = (Mock(), Mock(), Mock())
        self.noise_diode = NoiseDiode(
            dump_period=self.mock_dump_period, observation_log=self.mock_observation_log
        )

    @patch.object(NoiseDiode, "_get_where_noise_diode_is_off")
    @patch.object(NoiseDiode, "_get_noise_diode_cycle_start_times")
    def test_get_noise_diode_off_scan_dumps(
        self, mock_get_noise_diode_cycle_start_times, mock_get_where_noise_diode_is_off
    ):
        mock_timestamps = Mock()
        self.assertEqual(
            mock_get_where_noise_diode_is_off.return_value,
            self.noise_diode.get_noise_diode_off_scan_dumps(timestamps=mock_timestamps),
        )
        mock_get_where_noise_diode_is_off.assert_called_once_with(
            timestamps=mock_timestamps,
            noise_diode_cycle_starts=mock_get_noise_diode_cycle_start_times.return_value,
        )

    @patch.object(NoiseDiode, "_get_noise_diode_settings_from_obs_script")
    def test_get_noise_diode_settings(
        self, mock_get_noise_diode_settings_from_obs_script
    ):
        expect = (Mock(), Mock(), Mock())
        mock_get_noise_diode_settings_from_obs_script.return_value = expect
        self.assertTupleEqual(expect, self.noise_diode._get_noise_diode_settings())

    @patch.object(NoiseDiode, "_get_noise_diode_settings_from_obs_script")
    def test_get_noise_diode_settings_when_none_expect_raise(
        self, mock_get_noise_diode_settings_from_obs_script
    ):
        mock_get_noise_diode_settings_from_obs_script.return_value = (
            None,
            Mock(),
            Mock(),
        )
        self.assertRaises(
            NotImplementedError, self.noise_diode._get_noise_diode_settings
        )

    def test_get_noise_diode_settings_from_obs_script_fails_expect_none(self):
        duration, period, set_at = (
            self.noise_diode._get_noise_diode_settings_from_obs_script()
        )
        self.assertIsNone(duration)
        self.assertIsNone(period)
        self.assertIsNone(set_at)

    def test_get_noise_diode_settings_from_obs_script(self):
        mock_obs_script_log = [
            "INFO Repeat noise diode pattern every 1 2 sec on",
            "INFO Report: Switch noise-diode pattern on at 3",
        ]
        self.noise_diode._observation_log = mock_obs_script_log
        duration, period, set_at = (
            self.noise_diode._get_noise_diode_settings_from_obs_script()
        )
        self.assertEqual(2, duration)
        self.assertEqual(1, period)
        self.assertEqual(3, set_at)

    def test_get_noise_diode_settings_from_2019_obs_script(self):
        mock_obs_script_log = [
            "INFO Request noise diode pattern to repeat every 20.0 sec, with 1.8 sec on and apply pattern to all",
            "WARNING Set noise diode pattern to activate at 1556120503.0, with 5.0 sec lead time",
        ]
        self.noise_diode._observation_log = mock_obs_script_log
        duration, period, set_at = (
            self.noise_diode._get_noise_diode_settings_from_2019_obs_script()
        )
        self.assertEqual(1.8, duration)
        self.assertEqual(20.0, period)
        self.assertEqual(1556120503.0, set_at)

    @patch.object(NoiseDiode, "_get_noise_diode_ratios")
    def test_get_where_noise_diode_is_off(self, mock_get_noise_diode_ratios):
        mock_get_noise_diode_ratios.return_value = np.array([0, 0.1, 0.5])
        np.testing.assert_array_equal(
            np.array([0]),
            self.noise_diode._get_where_noise_diode_is_off(Mock(), Mock()),
        )

    def test_get_noise_diode_ratios_when_duration_short(self):
        mock_timestamps = DataElement(
            array=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])[
                :, np.newaxis, np.newaxis
            ]
        )
        mock_noise_diode_cycle_starts = np.array([1, 4, 7, 10])
        mock_dump_period = 1
        self.noise_diode.duration = 0.1
        ratios = self.noise_diode._get_noise_diode_ratios(
            timestamps=mock_timestamps,
            noise_diode_cycle_starts=mock_noise_diode_cycle_starts,
            dump_period=mock_dump_period,
        )
        expect = np.array([0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0])
        np.testing.assert_array_equal(expect, ratios)

    def test_get_noise_diode_ratios_when_duration_short_and_span_two_timestamps(self):
        mock_timestamps = DataElement(
            array=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])[
                :, np.newaxis, np.newaxis
            ]
        )
        mock_noise_diode_cycle_starts = np.array([1.46, 4.46, 7.46, 10.46])
        mock_dump_period = 1
        self.noise_diode.duration = 0.1
        ratios = self.noise_diode._get_noise_diode_ratios(
            timestamps=mock_timestamps,
            noise_diode_cycle_starts=mock_noise_diode_cycle_starts,
            dump_period=mock_dump_period,
        )
        expect = np.array(
            [0, 0.04, 0.06, 0, 0.04, 0.06, 0, 0.04, 0.06, 0, 0.04, 0.06, 0]
        )
        np.testing.assert_array_almost_equal(expect, ratios)

    def test_get_noise_diode_ratios_when_duration_long_and_span_two_timestamps(self):
        mock_timestamps = DataElement(
            array=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])[
                :, np.newaxis, np.newaxis
            ]
        )
        mock_noise_diode_cycle_starts = np.array([3, 6, 9])
        mock_dump_period = 1
        self.noise_diode.duration = 1.1
        ratios = self.noise_diode._get_noise_diode_ratios(
            timestamps=mock_timestamps,
            noise_diode_cycle_starts=mock_noise_diode_cycle_starts,
            dump_period=mock_dump_period,
        )
        expect = np.array([0, 0, 0, 0.5, 0.6, 0, 0.5, 0.6, 0, 0.5, 0.6, 0, 0])
        np.testing.assert_array_almost_equal(expect, ratios)

    def test_get_noise_diode_cycle_start_times(self):
        self.noise_diode.first_set_at = 0
        self.noise_diode.period = 3
        cycle_start_times = self.noise_diode._get_noise_diode_cycle_start_times(
            timestamps=np.array([15])
        )
        np.testing.assert_array_equal(np.array([0, 3, 6, 9, 12]), cycle_start_times)

    def test_get_noise_diode_cycle_start_times_when_period_long(self):
        self.noise_diode.first_set_at = 0
        self.noise_diode.period = 30
        cycle_start_times = self.noise_diode._get_noise_diode_cycle_start_times(
            timestamps=np.array([15])
        )
        np.testing.assert_array_equal(np.array([0]), cycle_start_times)

    def test_get_noise_diode_cycle_start_times_timestamps_too_early(self):
        self.noise_diode.first_set_at = 10
        self.noise_diode.period = 3
        self.assertRaises(
            ValueError,
            self.noise_diode._get_noise_diode_cycle_start_times,
            timestamps=np.array([5]),
        )
