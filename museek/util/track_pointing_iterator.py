from typing import Iterator

import numpy as np

from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData
from museek.util.clustering import Clustering
from matplotlib import pyplot as plt
import os


class TrackPointingIterator:
    """ Util class to iterate through the different calibrator pointings of an observation. """

    def __init__(self,
                 track_data: TimeOrderedData,
                 receiver: Receiver,
                 plot_dir: str | None,
                 scan_start: float,
                 scan_end: float,
                 n_calibrator_observations: int = 2,
                 calibrator_observation_labels: list[str] = ('before_scan', 'after_scan'),
                 n_pointings: int = 5,
                 n_centre_observations: int = 3,
                 distance_threshold: float = 5.,
                 pointing_slewing_thresholds: tuple[float, float] = (7., 15.),
                 min_max_pointing_time: tuple[float, float] = (55., 80.)):
        """
        Initialise
        :param track_data: data of the calibrator observation in tracking mode
        :param receiver: `Receiver` object
        :param plot_dir: optional directory to store diagnostic plots
        :param scan_start: time dump [s] of scan observation start
        :param scan_end: time dump [s] of scan observation end
        :param n_calibrator_observations: number of calibrator observations, typically 2, i.e. 1 before 1 after scan
        :param calibrator_observation_labels: `tuple` of `str` labels for the calibrator observations
        :param n_pointings: number of pointings, typically 5 for centre, up, left, down and right off-centre
        :param n_centre_observations: number of pointings onto the centre of the calibrator
        :param distance_threshold: forwarded to `Clustering`, pointings beyond this threshold are considered outliers
        :param pointing_slewing_thresholds: `tuple` of lower and upper `float` seconds thresholds for telescope slew 
                                            movement in between single dish calibrator pointings
        :param min_max_pointing_time: `tuple` of minimum and maximum `float` seconds observation times for individual
                                      single dish calibrator pointings
        :raise ValueError: if `n_calibrator_observations` is not equal to the `len` of `calibrator_observation_labels`
        """
        if len(calibrator_observation_labels) != n_calibrator_observations:
            raise ValueError(f'The length of `calibrator_observation_labels` must match `n_calibrator_observations`,'
                             f' got {calibrator_observation_labels} and {n_calibrator_observations}')
        self._features = track_data.timestamps.squeeze
        self._features_diff = self._features[1:] - self._features[:-1]
        self._n_clusters_before_after = n_calibrator_observations
        self._calibrator_observation_labels = calibrator_observation_labels
        self._clustering = Clustering()
        antenna_index = receiver.antenna_index(receivers=track_data.receivers)
        self._right_ascension = track_data.right_ascension.get(recv=antenna_index)
        self._declination = track_data.declination.get(recv=antenna_index)
        self._timestamps = track_data.timestamps
        self._n_pointings = n_pointings
        self._n_centre_observations = n_centre_observations
        self._distance_threshold = distance_threshold
        self._scan_start = scan_start
        self._scan_end = scan_end
        self._scan_observation_duration = self._scan_end - self._scan_start

        self._pointing_slewing_thresholds = pointing_slewing_thresholds
        self._min_max_pointing_time = min_max_pointing_time

        self._plot_dir = plot_dir

    def iterate(self) -> Iterator[tuple[str, range, list[np.ndarray], np.ndarray]]:
        """
        Iterate through the pointings and yield a `tuple` of calibrator pointing label, calibrator pointing time
        dumps, `list` of subpointing time dumps (e.g. off-centre pointings) and the correspoinding pointing centre
        coordinates.
        """
        if self._two_calibrator_observations():
            target_dumps_list = self._target_dumps_two_calibrators()
        else:
            target_dumps_list = self._target_dumps_one_calibrator()
        target_dumps_list = self._single_dish_calibrators(
            target_dumps_list=target_dumps_list,
            n_calibrator_pointings=self._n_pointings + self._n_centre_observations - 1,
        )
        for label, times in zip(self._calibrator_observation_labels, target_dumps_list):
            if times is None:
                print(f'No calibrator found for {label}.')
                yield label, None, None, None
                continue
            right_ascension = self._right_ascension.get(time=times)
            declination = self._declination.get(time=times)
            timestamps = self._timestamps.get(time=times)
            pointing_times_list, pointing_centres = self._clustering.split_pointings(
                coordinate_1=right_ascension.squeeze,
                coordinate_2=declination.squeeze,
                timestamps=timestamps.squeeze,
                n_pointings=self._n_pointings,
                n_centre_observations=self._n_centre_observations,
                distance_threshold=self._distance_threshold,
            )
            yield label, times, pointing_times_list, pointing_centres

    def _target_dumps_one_calibrator(self) -> list[range | None]:
        """
        Return `list` of `optional` `range` of dump indices for the case when only one calibrator is observed.
        This method decides wether the tracking data is closer to the scanning start or end and associates the
        calibrator observation with before or after the scanning.
        """
        target_dumps_list = [None, None]
        # is scan end or scan start closer to calibration start?
        calibrator_index = np.argmin(np.abs(np.asarray([self._scan_start, self._scan_end]) - self._features[0]))
        target_dumps_list[calibrator_index] = range(len(self._features)-1)
        return target_dumps_list

    def _target_dumps_two_calibrators(self) -> list[range]:
        """
        Return `list` of `range` of dump indices for the case when two calibrators are observed.
        This method looks for a gap in the track data timestamps and looks like it can accommodate the entire 
        scanning observation and takes this as a splitting point into before and after the scanning.
        """
        max_diff = max(self._features_diff)  # max difference between samples, in seconds
        if max_diff > 0.6*self._scan_observation_duration:
            argmax = np.argmax(self._features_diff)
            target_dumps_list = [range(0, argmax), range(argmax+1, len(self._features)-1)]
        else:
            target_dumps_list = self._clustering.ordered_dumps_of_coherent_clusters(
                features=self._features,
                n_clusters=self._n_clusters_before_after
            )
        return target_dumps_list

    def _two_calibrator_observations(self) -> bool:
        """ Return `True` if two calibrator observations are found. """
        max_diff = max(self._features_diff)  # max difference between samples, in seconds
        if max_diff < 0.6 * self._scan_observation_duration:  # at least 1 hour gap for scanning observation
            return False
        return True

    def _single_dish_calibrators(
        self,
        target_dumps_list: list[range],
        n_calibrator_pointings: int = 7,
    ) -> list[range | None]:
        """
        Returns a `list` of `range`s of target dump indices contained in `target_dumps_list` that belong to single dish
        calibrators only. It is assumed that single dish calibrators always have pointings onto the centre and to the
        right, left, up and down.
        :param target_dumps_list: `list` of `range`s defining the calibrator dumps to be weeded for single dish
        :param n_calibrator_pointings: number of observations of the single dish calibrator, defaults to 7 for
                                          right, centre, up, centre, left, centre, down
        """
        result = []
        lower, upper = self._pointing_slewing_thresholds
        min_pointing_time, max_pointing_time = self._min_max_pointing_time

        for dumps, label in zip(target_dumps_list, self._calibrator_observation_labels):
            if dumps is None:
                print(f'No calibrators found {label}- continue ...')
                result.append(None)
                continue
            features = self._features[dumps]
            features_diff = np.asarray(features[1:] - features[:-1])
            target_change_indices = np.where(features_diff >= upper)[0]
            pointing_change_indices = np.where((features_diff > lower) & (features_diff < upper))[0]
            pointing_change_dumps = np.asarray([features[i] for i in pointing_change_indices])
            pointing_change_dumps_diff = pointing_change_dumps[1:] - pointing_change_dumps[:-1]

            if not all((pointing_change_dumps_diff > min_pointing_time)
                       & (pointing_change_dumps_diff < max_pointing_time)):
                print(
                    f'WARNING: not all calibrator pointings are observed a minimum of {min_pointing_time} '
                    's and maximum of {max_pointing_time} s. Got max and min {max(pointing_change_dumps_diff)} '
                    'and {(pointing_change_dumps_diff)} s'
                )

            try:
                start_end = (min(pointing_change_indices), max(pointing_change_indices))
                if target_change_indices.size != 0:  # not empty array means we are seeing more than one calibrator
                    if all(start_end[0] < target_change_indices):
                        start_end = (0, min(target_change_indices))
                    elif all(start_end[1] > target_change_indices):
                        start_end = (max(target_change_indices)+1, len(features)-1)
                valid_indices = (start_end[0], start_end[1])
            except ValueError:
                valid_indices = None

            if valid_indices is not None and max(features_diff[range(*valid_indices)]) > upper:
                print('WARNING: something went wrong')

            if self._plot_dir is not None:
                plt.scatter(features[:-1], features_diff)
                for i in pointing_change_indices:
                    plt.axvline(features[i])
                if valid_indices is not None:
                    plt.axvline(features[valid_indices[0]], color='black')
                    plt.axvline(features[valid_indices[1]], color='black')
                plot_name = f'track_pointing_iterator_single_dish_calibrators_{label}.png'
                plt.xlabel('time [s]')
                plt.ylabel('time in between track pointings [s]')
                plt.savefig(os.path.join(self._plot_dir, plot_name))
                plt.close()

            if len(pointing_change_indices) != n_calibrator_pointings - 1 or valid_indices is None:
                print(f'No single dish calibrator found {label} - continue ...')
                result.append(None)
            else:
                result.append(range(dumps[valid_indices[0]], dumps[valid_indices[1]]))
        return result
