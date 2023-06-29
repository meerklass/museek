import os
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import legendre

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
                                             track_data=track_data,
                                             bandpasses_dict=bandpasses_dict,
                                             corrected_bandpasses_dict=corrected_bandpasses_dict,
                                             plot_name=f'bandpass_during_tracking_{before_or_after}.png')
                    self.plot_bandpass(track_data=track_data,
                                       bandpass=bandpasses_dict['on centre 1'] + bandpasses_dict['off centre top'],
                                       label='on centre 1 + off centre top',
                                       receiver_path=receiver_path,
                                       before_or_after=before_or_after)

                    self.plot_fft(track_data=track_data,
                                  bandpasses_dict=bandpasses_dict,
                                  receiver_path=receiver_path,
                                  before_or_after=before_or_after)

                    mean = (bandpasses_dict['on centre 1'] + bandpasses_dict['off centre top']) / 2
                    epsilon = self.epsilon(track_data=track_data, mean=mean, receiver_path=receiver_path)

                    no_wiggles_bandpasses_dict = {}
                    for key, value in bandpasses_dict.items():
                        no_wiggles_bandpasses_dict[key] = value / (1 + epsilon)

                    self.plot_all_bandpasses(
                        receiver=receiver,
                        receiver_path=receiver_path,
                        track_data=track_data,
                        bandpasses_dict=no_wiggles_bandpasses_dict,
                        corrected_bandpasses_dict=corrected_bandpasses_dict,
                        plot_name=f'no_wiggles_bandpass_during_tracking_{before_or_after}.png'
                    )



            except ValueError:
                print(f'Receiver {receiver.name} failed to process. Continue...')
                raise

    def plot_bandpass(self, track_data, bandpass, label, receiver_path, before_or_after):
        plt.figure(figsize=(16, 8))
        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                 bandpass,
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

            bandpass_pointing = target_visibility.mean(axis=0, flags=flags).squeeze
            bandpasses_dict[label] = bandpass_pointing

            if track_data.gain_solution is not None:
                corrected_target_visibility_centre = track_data.corrected_visibility().get(
                    recv=i_receiver,
                    freq=self.target_channels,
                    time=track_times
                )
                corrected_bandpass = corrected_target_visibility_centre.mean(axis=0).squeeze
                corrected_bandpasses_dict[label] = corrected_bandpass
            else:
                corrected_bandpasses_dict[label] = None
        return bandpasses_dict, corrected_bandpasses_dict, track_times_dict

    def plot_all_bandpasses(self,
                            receiver,
                            receiver_path,
                            track_data,
                            bandpasses_dict,
                            corrected_bandpasses_dict,
                            plot_name):
        plt.figure(figsize=(16, 8))
        ax1 = plt.subplot(1, 1, 1)
        for key in bandpasses_dict.keys():
            if corrected_bandpasses_dict[key] is not None:
                plt.close()
                plt.figure(figsize=(19, 8))
                ax1 = plt.subplot(3, 1, 1)

        for key in bandpasses_dict.keys():

            plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                     bandpasses_dict[key],
                     label=key)
            plt.xlabel('frequency [MHz]')
            plt.ylabel('intensity')
            plt.legend()

            if corrected_bandpasses_dict[key] is not None:
                gain_solution_mean = corrected_bandpasses_dict[key].mean(axis=0).squeeze
                plt.subplot(3, 1, 2, sharex=ax1)
                plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                         corrected_bandpasses_dict[key],
                         label=key)
                plt.xlabel('frequency [MHz]')
                plt.ylabel('level2 calibrated intensity')
                plt.legend()

                plt.subplot(3, 1, 3, sharex=ax1)
                plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                         gain_solution_mean[1:],
                         label='on-centre observation')
                plt.xlabel('frequency [MHz]')
                plt.ylabel('level2 gain solution')
                plt.legend()

        plt.suptitle(f'Calibrator tracking receiver {receiver.name}')
        plt.savefig(os.path.join(receiver_path, plot_name))
        plt.close()

    def plot_fft(self, track_data, bandpasses_dict, receiver_path, before_or_after):
        mean = (bandpasses_dict['on centre 1'] + bandpasses_dict['off centre top']) / 2
        # mean -= np.mean(mean)
        difference = bandpasses_dict['on centre 4'] - bandpasses_dict['off centre right']
        # quantity = difference / mean
        label = 'fft'
        # norm_channel = 1122  # when using everything
        norm_channel = 0  # when using cosmo frequencies only
        # quantity =  (bandpasses_dict['on centre 4'] + bandpasses_dict['off centre right'])

        plt.figure(figsize=(16, 8))

        # plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / MEGA,
        #          quantity.squeeze[1:] / quantity.get(freq=norm_channel).squeeze,
        #          color='green',
        #          label=label)
        #
        # plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / MEGA,
        #          difference.squeeze[1:] / difference.get(freq=norm_channel).squeeze,
        #          color='red',
        #          label=label)
        # plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / MEGA,
        #          mean[1:] / mean.get(freq=norm_channel).squeeze,
        #          color='blue',
        #          label=label)
        plt.subplot(4, 1, 1)
        fft = np.fft.fft(mean)
        fft_freq = np.fft.fftfreq(
            len(mean),
            (track_data.frequencies.get(freq=1).squeeze - track_data.frequencies.get(freq=0).squeeze) / MEGA
        )
        filtered_fft = deepcopy(fft)
        max_freq = 0.13  # delta is around 0.0245
        min_freq = 0.09
        filtered_fft[(np.abs(fft_freq) < max_freq) & (np.abs(fft_freq) > min_freq)] = 0
        # filtered_fft[(min_freq < np.abs(fft_freq)) & (np.abs(fft_freq) < max_freq)] = 0

        # filtered_fft[:band] = 0
        # filtered_fft[band + bandwidth:] = 0

        plt.plot(fft_freq, np.abs(fft), label='fft')
        plt.plot(fft_freq, np.abs(filtered_fft), label='fft filtered', ls=':')
        plt.ylim((0, 300))
        plt.ylabel('fft')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                 mean,
                 label='mean')
        plt.legend()

        filtered_mean = np.fft.ifft(filtered_fft)

        plt.subplot(4, 1, 3)
        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                 filtered_mean.real,
                 label='filtered mean')

        plt.ylabel('intensity')

        plt.legend()
        plt.xlabel('frequency [MHz]')

        plt.subplot(4, 1, 4)
        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                 mean - filtered_mean.real,
                 label='difference mean and filtered')
        plt.legend()

        plt.savefig(os.path.join(receiver_path, f'bandpass_{label}_{before_or_after}.png'))
        plt.close()

    def epsilon(self, track_data, mean, receiver_path):

        legendre_degree = 4
        coefficients = legendre.legfit(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                                       mean,
                                       legendre_degree)
        legendre_fit = legendre.legval(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                                       coefficients)
        plt.figure(figsize=(16, 12))
        plt.subplot(3, 1, 1)
        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                 mean,
                 label='mean')
        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                 legendre_fit,
                 label='legendre fit')
        plt.ylabel('intensity')
        plt.xlabel('frequency [MHz]')
        plt.legend()

        plt.subplot(3, 1, 2)
        epsilon_noisy = mean / legendre_fit - 1  # epsilon = signal / signal_no_wiggles - 1

        fft = np.fft.fft(epsilon_noisy)
        fft_freq = np.fft.fftfreq(
            len(epsilon_noisy),
            (track_data.frequencies.get(freq=1).squeeze - track_data.frequencies.get(freq=0).squeeze) / MEGA
        )

        filtered_fft = deepcopy(fft)
        min_freq = 0.15
        filtered_fft[np.abs(fft_freq) > min_freq] = 0
        epsilon = np.fft.ifft(filtered_fft)

        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                 epsilon_noisy.real,
                 label='epsilon = mean / legendre - 1')
        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                 epsilon.real,
                 label='lowpass of epsilon =  mean / legendre - 1')
        plt.ylabel('intensity')
        plt.xlabel('frequency [MHz]')

        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze / MEGA,
                 mean / (1 + epsilon.real),
                 label='mean without wiggles')
        plt.ylabel('intensity')
        plt.xlabel('frequency [MHz]')

        plt.legend()
        plt.savefig(os.path.join(receiver_path, 'legendre_fit_fft_denoise_epsilon.png'))
        plt.close()

        plt.plot(fft_freq[fft_freq > 0], np.abs(fft[fft_freq > 0]))
        plt.xlabel('µs')
        plt.ylabel('fft of epsilon before denoising')
        plt.xlim((0., 0.5))
        plt.axvline(min_freq, label='delay cut to remove noise')
        plt.legend()
        plt.savefig(os.path.join(receiver_path, 'fft_epsilon.png'))
        plt.close()

        return epsilon.real
