import os

import numpy as np
from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.clustering import Clustering


class BandpassPlugin(AbstractPlugin):
    """ Example plugin to help later development of the bandpass correction. """

    def __init__(self,
                 target_channels: range | list[int],
                 pointing_threshold: float,
                 n_pointings: int,
                 n_centre_observations: int):
        """
        Initialise
        :param target_channels: `list` or `range` of channel indices to be examined
        :param pointing_threshold: deviations up to this tolerance from the pointing are accepted
        :param n_pointings: number of pointings per calibrator, usually 5 (centre, up, right, down, left)
        :param n_centre_observations: number of on-centre calibrator observations
        """
        super().__init__()
        self.target_channels = target_channels
        self.pointing_threshold = pointing_threshold
        self.n_pointings = n_pointings
        self.n_centre_observations = n_centre_observations

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, track_data: TimeOrderedData, output_path: str):
        """
        Split the tracking data all different pointings and create pointing and spectra plots.
        :param track_data: the time ordered data containing the observation's tracking part
        :param output_path: path to store results
        """

        mega = 1e6
        track_data.load_visibility_flags_weights()
        target_dumps_list = Clustering().ordered_dumps_of_coherent_clusters(features=track_data.timestamps.squeeze,
                                                                            n_clusters=2)

        for i_receiver, receiver in enumerate(track_data.receivers):
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)

            for before_or_after, times in zip(['before_scan', 'after_scan'], target_dumps_list):
                right_ascension = track_data.right_ascension.get(recv=i_receiver // 2,
                                                                 time=times).squeeze
                declination = track_data.declination.get(recv=i_receiver // 2,
                                                         time=times).squeeze
                timestamps = track_data.timestamps.get(time=times).squeeze

                times_list, pointing_centres = Clustering().split_pointings(
                    coordinate_1=right_ascension,
                    coordinate_2=declination,
                    timestamps=timestamps,
                    n_pointings=self.n_pointings,
                    n_centre_observations=self.n_centre_observations,
                    distance_threshold=self.pointing_threshold
                )
                centre_times = times_list[0]
                for i, (t, p) in enumerate(zip(times_list, pointing_centres)):
                    color = 'black'
                    label_1 = ''
                    label_2 = ''
                    if i == 0:
                        label_2 = 'on-centre dumps'
                    if i <= 2:
                        color = 'red'
                    elif i == 3:
                        label_1 = 'pointing centres'
                        label_2 = 'off-centre dumps'
                    plt.scatter(right_ascension[t], declination[t], color=color, label=label_2)
                    plt.scatter(p[0], p[1], color=color, marker='x', s=100, label=label_1)
                plt.legend()
                plt.xlabel('RA [deg]')
                plt.ylabel('Dec [deg]')
                plt.savefig(os.path.join(receiver_path, f'track_pointing_{before_or_after}.png'))
                plt.close()

                track_centre_times = list(np.asarray(times)[centre_times])
                target_visibility_centre = track_data.visibility.get(recv=i_receiver,
                                                                     freq=self.target_channels,
                                                                     time=track_centre_times)
                bandpass_centre = target_visibility_centre.mean(axis=0)
                if track_data.gain_solution is not None:
                    gain_solution = track_data.gain_solution.get(
                        recv=i_receiver,
                        freq=self.target_channels,
                        time=track_centre_times
                    )
                    corrected_target_visibility_centre = track_data.corrected_visibility().get(
                        recv=i_receiver,
                        freq=self.target_channels,
                        time=track_centre_times
                    )
                    corrected_bandpass = corrected_target_visibility_centre.mean(axis=0)
                    plt.figure(figsize=(12, 8))
                    ax1 = plt.subplot(3, 1, 1)
                else:
                    plt.figure(figsize=(8, 8))
                    ax1 = plt.subplot(1, 1, 1)

                plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / mega,
                         bandpass_centre.squeeze[1:],
                         label='on-centre observation')
                plt.xlabel('frequency [MHz]')
                plt.ylabel('intensity')
                plt.legend()

                if track_data.gain_solution is not None:
                    gain_solution_mean = gain_solution.mean(axis=0).squeeze
                    plt.subplot(3, 1, 2, sharex=ax1)
                    plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / mega,
                             corrected_bandpass.squeeze[1:],
                             label='on-centre observation')
                    plt.xlabel('frequency [MHz]')
                    plt.ylabel('level2 calibrated intensity')
                    plt.legend()

                    plt.subplot(3, 1, 3, sharex=ax1)
                    plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / mega,
                             gain_solution_mean[1:],
                             label='on-centre observation')
                    plt.xlabel('frequency [MHz]')
                    plt.ylabel('level2 gain solution')
                    plt.legend()

                plt.suptitle(f'Calibrator tracking receiver {receiver.name}')
                plt.savefig(os.path.join(receiver_path, f'bandpass_during_tracking_{before_or_after}.png'))
                plt.close()
