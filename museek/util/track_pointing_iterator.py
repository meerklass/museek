from typing import Iterator

import numpy as np

from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData
from museek.util.clustering import Clustering


class TrackPointingIterator:
    """ Util class to iterate through the different calibrator pointings of an observation. """

    def __init__(self,
                 track_data: TimeOrderedData,
                 receiver: Receiver,
                 receiver_index: int,
                 n_calibrator_observations: int = 2,
                 calibrator_observation_labels: list[str] = ('before_scan', 'after_scan'),
                 n_pointings: int = 5,
                 n_centre_observations: int = 5,
                 distance_threshold: float = 5.):
        """
        Initialise
        :param track_data: data of the calibrator observation in tracking mode
        :param receiver: `Receiver` object
        :param receiver_index: index of `receiver`
        :param n_calibrator_observations: number of calibrator observations, typically 2
        :param calibrator_observation_labels: `tuple` of `str` labels for the calibrator observations
        :param n_pointings: number of pointings, typically 5 for centre, up, left, down and right off-centre
        :param n_centre_observations: number of pointings onto the centre
        :param distance_threshold: forwarded to `Clustering`
        :raise ValueError: if `n_calibrator_observations` is not equal to the `len` of `calibrator_observation_labels`
        """
        if len(calibrator_observation_labels) != n_calibrator_observations:
            raise ValueError(f'The length of `calibrator_observation_labels` must match `n_calibrator_observations`,'
                             f' got {calibrator_observation_labels} and {n_calibrator_observations}')
        self._features = track_data.timestamps.squeeze
        self._n_clusters_before_after = n_calibrator_observations
        self._before_or_after_labels = calibrator_observation_labels
        self._clustering = Clustering()
        antenna_index = receiver.antenna_index(receivers=track_data.receivers)
        self._right_ascension = track_data.right_ascension.get(recv=antenna_index)
        self._declination = track_data.declination.get(recv=antenna_index)
        self._timestamps = track_data.timestamps
        self._n_pointings = n_pointings
        self._n_centre_observations = n_centre_observations
        self._distance_threshold = distance_threshold

    def iterate(self) -> Iterator[tuple[str, list[np.ndarray], np.ndarray]]:
        """
        Iterate through the pointings and yield a `tuple` of calibrator pointing label, calibrator pointing time
        dumps, `list` of subpointing time dumps (e.g. off-centre pointings) and the correspoinding pointing centre
        coordinates.
        """
        target_dumps_list = self._clustering.ordered_dumps_of_coherent_clusters(
            features=self._features,
            n_clusters=self._n_clusters_before_after
        )
        for before_or_after, times in zip(self._before_or_after_labels, target_dumps_list):
            right_ascension = self._right_ascension.get(time=times)
            declination = self._declination.get(time=times)
            timestamps = self._timestamps.get(time=times)
            times_list, pointing_centres = self._clustering.split_pointings(
                coordinate_1=right_ascension.squeeze,
                coordinate_2=declination.squeeze,
                timestamps=timestamps.squeeze,
                n_pointings=self._n_pointings,
                n_centre_observations=self._n_centre_observations,
                distance_threshold=self._distance_threshold,
            )
            yield before_or_after, times, times_list, pointing_centres
