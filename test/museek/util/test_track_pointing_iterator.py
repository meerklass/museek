import unittest
from unittest.mock import Mock, patch, call

from museek.util.clustering import Clustering
from museek.util.track_pointing_iterator import TrackPointingIterator


class TestTrackPointingIterator(unittest.TestCase):
    def test_init(self):
        mock_track_data = Mock()
        mock_receiver = Mock()
        track_pointting_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                         receiver=mock_receiver,
                                                         receiver_index=0,
                                                         n_pointings=0,
                                                         n_centre_observations=0,
                                                         distance_threshold=0)
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
                          receiver_index=0,
                          n_pointings=0,
                          n_centre_observations=0,
                          calibrator_observation_labels=['a', 'b'],
                          n_calibrator_observations=1,
                          distance_threshold=0)

    @patch.object(Clustering, 'split_pointings')
    @patch.object(Clustering, 'ordered_dumps_of_coherent_clusters')
    def test_iterate(self, mock_ordered_dumps_of_coherent_clusters, mock_split_pointings):
        mock_ordered_dumps_of_coherent_clusters.return_value = [Mock(), Mock()]
        mock_split_pointings.return_value = (Mock(), Mock())
        mock_track_data = Mock()
        mock_receiver = Mock()
        mock_labels = ['a', 'b']
        track_pointting_iterator = TrackPointingIterator(track_data=mock_track_data,
                                                         receiver=mock_receiver,
                                                         receiver_index=0,
                                                         n_pointings=0,
                                                         n_centre_observations=0,
                                                         distance_threshold=0,
                                                         calibrator_observation_labels=mock_labels)

        for i, (w, x, y, z) in enumerate(track_pointting_iterator.iterate()):
            self.assertEqual(mock_labels[i], w)
            self.assertEqual(mock_ordered_dumps_of_coherent_clusters.return_value[i], x)
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
