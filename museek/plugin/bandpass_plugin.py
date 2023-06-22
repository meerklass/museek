import os

import numpy as np
from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.clustering import Clustering

MEGA = 1e6


class BandpassPlugin(AbstractPlugin):
    """ Example plugin to help later development of the bandpass correction. """

    def __init__(self,
                 target_channels: range | list[int] | None,
                 pointing_threshold: float,
                 n_pointings: int,
                 n_centre_observations: int):
        """
        Initialise
        :param target_channels: optional `list` or `range` of channel indices to be examined, if `None`, all are used
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

        track_data.load_visibility_flags_weights()
        target_dumps_list = Clustering().ordered_dumps_of_coherent_clusters(features=track_data.timestamps.squeeze,
                                                                            n_clusters=2)
        pointing_labels = ['on centre 1',
                           'off centre top',
                           'on centre 2',
                           'off centre right',
                           'on centre 3',
                           'off centre down',
                           'on centre 4',
                           'off centre left']
        for i_receiver, receiver in enumerate(track_data.receivers):
            try:
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
                    self.plot_pointings(right_ascension=right_ascension,
                                        declination=declination,
                                        times_list=times_list,
                                        pointing_centres=pointing_centres,
                                        receiver_path=receiver_path,
                                        before_or_after=before_or_after)

                    bandpasses_dict, corrected_bandpasses_dict, track_times_dict = self.get_bandpasses_dicts(
                        track_data=track_data,
                        times_list=times_list,
                        times=times,
                        pointing_labels=pointing_labels,
                        i_receiver=i_receiver
                    )

                    self.plot_all_bandpasses(receiver=receiver,
                                             receiver_path=receiver_path,
                                             before_or_after=before_or_after,
                                             track_data=track_data,
                                             bandpasses_dict=bandpasses_dict,
                                             corrected_bandpasses_dict=corrected_bandpasses_dict)

                    self.plot_bandpass(track_data=track_data,
                                       bandpass=bandpasses_dict['on centre 1'] - bandpasses_dict['off centre top'],
                                       label='on centre 1 - off centre top',
                                       receiver_path=receiver_path,
                                       before_or_after=before_or_after)

            except ValueError:
                print(f'Receiver {receiver.name} failed to process. Continue...')
                continue

    def plot_bandpass(self, track_data, bandpass, label, receiver_path, before_or_after):
        plt.figure(figsize=(16, 8))
        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / MEGA,
                 bandpass.squeeze[1:],
                 label=label)
        plt.legend()
        plt.xlabel('frequency [MHz]')
        plt.ylabel('intensity')
        plt.savefig(os.path.join(receiver_path, f'bandpass_{label}_{before_or_after}.png'))
        plt.close()

    @staticmethod
    def plot_pointings(right_ascension,
                       declination,
                       times_list,
                       pointing_centres,
                       receiver_path,
                       before_or_after):
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

    def get_bandpasses_dicts(self, track_data, times_list, times, pointing_labels, i_receiver):
        bandpasses_dict = dict()
        corrected_bandpasses_dict = dict()
        track_times_dict = dict()
        for i_label, pointing_times in enumerate(times_list):
            label = pointing_labels[i_label]
            track_times = list(np.asarray(times)[pointing_times])
            track_times_dict[label] = track_times
            target_visibility = track_data.visibility.get(recv=i_receiver,
                                                          freq=self.target_channels,
                                                          time=track_times)
            flags = track_data.flags.get(recv=i_receiver,
                                         freq=self.target_channels,
                                         time=track_times)

            bandpass_pointing = target_visibility.mean(axis=0, flags=flags)
            bandpasses_dict[label] = bandpass_pointing

            if track_data.gain_solution is not None:
                corrected_target_visibility_centre = track_data.corrected_visibility().get(
                    recv=i_receiver,
                    freq=self.target_channels,
                    time=track_times
                )
                corrected_bandpass = corrected_target_visibility_centre.mean(axis=0)
                corrected_bandpasses_dict[label] = corrected_bandpass
            else:
                corrected_bandpasses_dict[label] = None
        return bandpasses_dict, corrected_bandpasses_dict, track_times_dict

    def plot_all_bandpasses(self,
                            receiver,
                            receiver_path,
                            before_or_after,
                            track_data,
                            bandpasses_dict,
                            corrected_bandpasses_dict):
        plt.figure(figsize=(16, 8))
        ax1 = plt.subplot(1, 1, 1)
        for key in bandpasses_dict.keys():
            if corrected_bandpasses_dict[key] is not None:
                plt.close()
                plt.figure(figsize=(19, 8))
                ax1 = plt.subplot(3, 1, 1)

        for key in bandpasses_dict.keys():

            plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / MEGA,
                     bandpasses_dict[key].squeeze[1:],
                     label=key)
            plt.xlabel('frequency [MHz]')
            plt.ylabel('intensity')
            plt.legend()

            if corrected_bandpasses_dict[key] is not None:
                gain_solution_mean = corrected_bandpasses_dict[key].mean(axis=0).squeeze
                plt.subplot(3, 1, 2, sharex=ax1)
                plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / MEGA,
                         corrected_bandpasses_dict[key].squeeze[1:],
                         label=key)
                plt.xlabel('frequency [MHz]')
                plt.ylabel('level2 calibrated intensity')
                plt.legend()

                plt.subplot(3, 1, 3, sharex=ax1)
                plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / MEGA,
                         gain_solution_mean[1:],
                         label='on-centre observation')
                plt.xlabel('frequency [MHz]')
                plt.ylabel('level2 gain solution')
                plt.legend()

        plt.suptitle(f'Calibrator tracking receiver {receiver.name}')
        plt.savefig(os.path.join(receiver_path, f'bandpass_during_tracking_{before_or_after}.png'))
        plt.close()
