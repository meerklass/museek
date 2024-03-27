import unittest
from unittest.mock import Mock, patch, call

import numpy as np
from mock import MagicMock

from museek.util.clustering import Clustering
from museek.util.track_pointing_iterator import TrackPointingIterator


class TestTrackPointingIterator(unittest.TestCase):

    def test_init(self):
        mock_track_data = MagicMock()
        mock_receiver = Mock()
        track_pointting_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                         receiver=mock_receiver,
                                                         n_pointings=0,
                                                         plot_dir=None,
                                                         n_centre_observations=0,
                                                         distance_threshold=0,
                                                         scan_start=0,
                                                         scan_end=1)
        self.assertIsInstance(track_pointting_iterator._clustering, Clustering)
        self.assertEqual(track_pointting_iterator._features, mock_track_data.timestamps.squeeze)
        antenna_index = mock_receiver.antenna_index.return_value
        mock_track_data.right_ascension.get.assert_called_once_with(recv=antenna_index)
        mock_track_data.declination.get.assert_called_once_with(recv=antenna_index)

    def test_init_expect_raise(self):
        mock_track_data = Mock()
        mock_receiver = Mock()
        self.assertRaises(ValueError,
                          TrackPointingIterator,
                          track_data=mock_track_data,
                          receiver=mock_receiver,
                          n_pointings=0,
                          plot_dir=None,
                          n_centre_observations=0,
                          calibrator_observation_labels=['a', 'b'],
                          n_calibrator_observations=1,
                          distance_threshold=0,
                          scan_start=0,
                          scan_end=1)

    @patch.object(TrackPointingIterator, '_target_dumps_two_calibrators')
    @patch.object(TrackPointingIterator, '_two_calibrator_observations')
    @patch.object(TrackPointingIterator, '_single_dish_calibrators')
    @patch.object(Clustering, 'split_pointings')
    def test_iterate_when_two_calibrator_observations(self,
                                                      mock_split_pointings,
                                                      mock_single_dish_calibrators,
                                                      mock_two_calibrator_observations,
                                                      mock_target_dumps_two_calibrators):
        mock_two_calibrator_observations.return_value = True
        mock_single_dish_calibrators.return_value = [MagicMock(), MagicMock()]
        mock_split_pointings.return_value = (MagicMock(), MagicMock())
        mock_track_data = MagicMock()
        mock_receiver = Mock()
        mock_labels = ['a', 'b']
        track_pointting_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                         receiver=mock_receiver,
                                                         n_pointings=0,
                                                         plot_dir=None,
                                                         n_centre_observations=0,
                                                         distance_threshold=0,
                                                         calibrator_observation_labels=mock_labels,
                                                         scan_start=0,
                                                         scan_end=1)

        for i, (w, x, y, z) in enumerate(track_pointting_iterator.iterate()):
            self.assertEqual(mock_labels[i], w)
            self.assertEqual(mock_single_dish_calibrators.return_value[i], x)
            self.assertEqual(mock_split_pointings.return_value[0], y)
            self.assertEqual(mock_split_pointings.return_value[1], z)
        mock_split_pointings.assert_has_calls(
            [call(coordinate_1=track_pointting_iterator._right_ascension.get().squeeze,
                  coordinate_2=track_pointting_iterator._declination.get().squeeze,
                  timestamps=track_pointting_iterator._timestamps.get().squeeze,
                  n_pointings=track_pointting_iterator._n_pointings,
                  n_centre_observations=track_pointting_iterator._n_centre_observations,
                  distance_threshold=track_pointting_iterator._distance_threshold)] * 2
        )
        mock_single_dish_calibrators.assert_called_once_with(
            target_dumps_list=mock_target_dumps_two_calibrators.return_value,
            n_calibrator_pointings=-1
        )

    @patch.object(TrackPointingIterator, '_target_dumps_one_calibrator')
    @patch.object(TrackPointingIterator, '_two_calibrator_observations')
    @patch.object(TrackPointingIterator, '_single_dish_calibrators')
    @patch.object(Clustering, 'split_pointings')
    def test_iterate_when_one_calibrator_observation(self,
                                                     mock_split_pointings,
                                                     mock_single_dish_calibrators,
                                                     mock_two_calibrator_observations,
                                                     mock_target_dumps_one_calibrator):
        mock_two_calibrator_observations.return_value = False
        mock_single_dish_calibrators.return_value = [MagicMock(), MagicMock()]
        mock_split_pointings.return_value = (MagicMock(), MagicMock())
        mock_track_data = MagicMock()
        mock_receiver = Mock()
        mock_labels = ['a', 'b']
        track_pointting_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                         receiver=mock_receiver,
                                                         n_pointings=0,
                                                         plot_dir=None,
                                                         n_centre_observations=0,
                                                         distance_threshold=0,
                                                         calibrator_observation_labels=mock_labels,
                                                         scan_start=0,
                                                         scan_end=1)

        for i, (w, x, y, z) in enumerate(track_pointting_iterator.iterate()):
            self.assertEqual(mock_labels[i], w)
            self.assertEqual(mock_single_dish_calibrators.return_value[i], x)
            self.assertEqual(mock_split_pointings.return_value[0], y)
            self.assertEqual(mock_split_pointings.return_value[1], z)
        mock_split_pointings.assert_has_calls(
            [call(coordinate_1=track_pointting_iterator._right_ascension.get().squeeze,
                  coordinate_2=track_pointting_iterator._declination.get().squeeze,
                  timestamps=track_pointting_iterator._timestamps.get().squeeze,
                  n_pointings=track_pointting_iterator._n_pointings,
                  n_centre_observations=track_pointting_iterator._n_centre_observations,
                  distance_threshold=track_pointting_iterator._distance_threshold)] * 2
        )
        mock_single_dish_calibrators.assert_called_once_with(
            target_dumps_list=mock_target_dumps_one_calibrator.return_value,
            n_calibrator_pointings=-1
        )

    @patch.object(Clustering, 'split_pointings')
    @patch.object(TrackPointingIterator, '_single_dish_calibrators')
    @patch.object(TrackPointingIterator, '_target_dumps_one_calibrator')
    @patch.object(TrackPointingIterator, '_two_calibrator_observations')
    def test_iterate_when_times_none(self,
                                     mock_two_calibrator_observations,
                                     mock_target_dumps_one_calibrator,
                                     mock_single_dish_calibrators,
                                     mock_split_pointings):
        mock_two_calibrator_observations.return_value = False
        mock_single_dish_calibrators.return_value = [None, None]
        mock_split_pointings.return_value = (MagicMock(), MagicMock())
        mock_track_data = MagicMock()
        mock_receiver = Mock()
        mock_labels = ['a', 'b']
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir=None,
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        calibrator_observation_labels=mock_labels,
                                                        scan_start=0,
                                                        scan_end=1)
        count = 0
        for expect, label in zip(track_pointing_iterator.iterate(), mock_labels):
            count += 1
            self.assertEqual(expect[0], label)
            for i in range(1, 4):
                self.assertIsNone(expect[i])
        self.assertEqual(2, count)

    def test_target_dumps_one_calibrator_when_after(self):
        mock_track_data = MagicMock()
        mock_track_data.timestamps.squeeze = np.asarray([2, 3])
        mock_receiver = Mock()
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir=None,
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        scan_start=0,
                                                        scan_end=1)
        dumps = track_pointing_iterator._target_dumps_one_calibrator()
        self.assertIsNone(dumps[0])
        self.assertEqual(dumps[1], range(0, 1))

    def test_target_dumps_one_calibrator_when_before(self):
        mock_track_data = MagicMock()
        mock_track_data.timestamps.squeeze = np.asarray([2, 3])
        mock_receiver = Mock()
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir=None,
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        scan_start=4,
                                                        scan_end=5)
        dumps = track_pointing_iterator._target_dumps_one_calibrator()
        self.assertIsNone(dumps[1])
        self.assertEqual(dumps[0], range(0, 1))

    def test_target_dumps_two_calibrators_when_if(self):
        mock_track_data = MagicMock()
        mock_track_data.timestamps.squeeze = np.asarray([0, 0.1, 1.1, 1.2])
        mock_receiver = Mock()
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir=None,
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        scan_start=0,
                                                        scan_end=1)
        target_dumps = track_pointing_iterator._target_dumps_two_calibrators()
        self.assertEqual(target_dumps[0], range(0, 1))
        self.assertEqual(target_dumps[1], range(2, 3))

    @patch.object(Clustering, 'ordered_dumps_of_coherent_clusters')
    def test_target_dumps_two_calibrators_when_else(self, mock_ordered_dumps_of_coherent_clusters):
        mock_track_data = MagicMock()
        mock_track_data.timestamps.squeeze = np.asarray([0, 0.1, 0.4, 0.5])
        mock_receiver = Mock()
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir=None,
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        scan_start=0,
                                                        scan_end=1)
        target_dumps = track_pointing_iterator._target_dumps_two_calibrators()
        self.assertEqual(target_dumps, mock_ordered_dumps_of_coherent_clusters.return_value)

    def test_two_calibrator_observations_when_true(self):
        mock_track_data = MagicMock()
        mock_track_data.timestamps.squeeze = np.asarray([0, 0.1, 1.1, 1.2])
        mock_receiver = Mock()
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir=None,
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        scan_start=0,
                                                        scan_end=1)
        self.assertTrue(track_pointing_iterator._two_calibrator_observations())

    def test_two_calibrator_observations_when_true(self):
        mock_track_data = MagicMock()
        mock_track_data.timestamps.squeeze = np.asarray([0, 0.1, 0.4, 0.5])
        mock_receiver = Mock()
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir=None,
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        scan_start=0,
                                                        scan_end=1)
        self.assertFalse(track_pointing_iterator._two_calibrator_observations())

    @patch('museek.util.track_pointing_iterator.plt')
    def test_single_dish_calibrators_when_only_after(self, mock_plt):
        mock_track_data = MagicMock()
        mock_receiver = Mock()
        mock_labels = ['a', 'b']
        mock_target_dumps_list = [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
        mock_track_data.timestamps.squeeze = np.asarray([0, 2, 4, 6, 12, 14, 16, 116, 118, 216])
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir='',
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        calibrator_observation_labels=mock_labels,
                                                        scan_start=0,
                                                        scan_end=1,
                                                        pointing_slewing_thresholds=(4, 30))
        result = track_pointing_iterator._single_dish_calibrators(target_dumps_list=mock_target_dumps_list,
                                                                  n_calibrator_pointings=2)
        mock_plt.scatter.assert_called()
        self.assertIsNone(result[0])
        self.assertEqual(result[1], range(1, 6))

    @patch('museek.util.track_pointing_iterator.plt')
    def test_single_dish_calibrators_when_only_before(self, mock_plt):
        mock_track_data = MagicMock()
        mock_receiver = Mock()
        mock_labels = ['a', 'b']
        mock_target_dumps_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [9]]
        mock_track_data.timestamps.squeeze = np.asarray([0, 2, 4, 6, 12, 14, 16, 116, 118, 216])
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir='',
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        calibrator_observation_labels=mock_labels,
                                                        scan_start=0,
                                                        scan_end=1,
                                                        pointing_slewing_thresholds=(4, 30))
        result = track_pointing_iterator._single_dish_calibrators(target_dumps_list=mock_target_dumps_list,
                                                                  n_calibrator_pointings=2)
        mock_plt.scatter.assert_called()
        self.assertIsNone(result[1])
        self.assertEqual(result[0], range(0, 6))

    def test_single_dish_calibrators_when_only_one_calibrator_seen(self):
        mock_track_data = MagicMock()
        mock_receiver = Mock()
        mock_labels = ['a', 'b']
        mock_target_dumps_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [9]]
        mock_track_data.timestamps.squeeze = np.asarray([0, 2, 4, 6, 12, 18, 19, 20, 21, 22])
        track_pointing_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                        receiver=mock_receiver,
                                                        n_pointings=0,
                                                        plot_dir='',
                                                        n_centre_observations=0,
                                                        distance_threshold=0,
                                                        calibrator_observation_labels=mock_labels,
                                                        scan_start=0,
                                                        scan_end=1,
                                                        pointing_slewing_thresholds=(4, 30))
        result = track_pointing_iterator._single_dish_calibrators(target_dumps_list=mock_target_dumps_list,
                                                                  n_calibrator_pointings=3)
        self.assertEqual(range(3, 4), result[0])
        self.assertIsNone(result[1])


if __name__ == '__main__':
    unittest.main()
