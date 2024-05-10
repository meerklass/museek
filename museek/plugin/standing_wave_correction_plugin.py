import os
from typing import Callable

import numpy as np
import scipy
from matplotlib import pyplot as plt

from definitions import MEGA
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.swings import Swings


class StandingWaveCorrectionPlugin(AbstractPlugin):
    """ Experimental plugin to apply the standing wave correction to the data"""

    def __init__(self,
                 target_channels: range | list[int] | None):
        """
        Initialise
        :param target_channels: optional `list` or `range` of channel indices to be examined, if `None`, all are used
        """
        super().__init__()
        self.target_channels = target_channels

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [
            Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
            Requirement(location=ResultEnum.STANDING_WAVE_CALIBRATOR_LABEL, variable='calibrator_label'),
            Requirement(location=ResultEnum.STANDING_WAVE_BANDPASS_ESTIMATOR, variable='bandpass_estimators')
        ]

    def run(self,
            scan_data: TimeOrderedData,
            output_path: str,
            calibrator_label: str,
            bandpass_estimators: list[DataElement]):
        """
        Run the plugin, i.e. apply the standing wave correction.
        :param scan_data: the time ordered data containing the observation's scanning part
        :param output_path: path to store results
        :param calibrator_label: `str` label to identify the standing wave calibrator that was used
        :param bandpass_estimators:
        """
        for i_receiver, receiver in enumerate(scan_data.receivers):
            print(f'Working on {receiver}...')
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            antenna_index = receiver.antenna_index(receivers=scan_data.receivers)
            frequencies = scan_data.frequencies.get(freq=self.target_channels)
            self.plot_individual_swings(scan_data=scan_data,
                                        antenna_index=antenna_index,
                                        receiver_index=i_receiver,
                                        target_channels=self.target_channels,
                                        bandpass_estimators=bandpass_estimators,
                                        receiver_path=receiver_path)

            self.plot_azimuth_bins(scan_data=scan_data,
                                   i_receiver=i_receiver,
                                   antenna_index=antenna_index,
                                   target_channels=self.target_channels,
                                   frequencies=frequencies,
                                   bandpass_estimator=bandpass_estimators,
                                   receiver_path=receiver_path)

    @staticmethod
    def azimuth_digitizer(azimuth: DataElement) -> tuple[np.ndarray, np.ndarray]:
        """ Digitize the `azimuth`. """
        bins = np.linspace(azimuth.min(axis=0).squeeze, azimuth.max(axis=0).squeeze, 50)
        return np.digitize(azimuth.squeeze, bins=bins), bins

    def plot_individual_swings(self,
                               scan_data,
                               antenna_index,
                               receiver_index,
                               target_channels,
                               bandpass_estimators,
                               receiver_path):
        """ Make plots with each individual swing as one line. """
        swing_turnaround_dumps = Swings.swing_turnaround_dumps(
            azimuth=scan_data.azimuth.get(recv=antenna_index)
        )
        fig = plt.figure(figsize=(8, 12))

        ax = fig.subplots(2, 1)
        jet = plt.get_cmap('jet')
        colors = jet(np.arange(len(swing_turnaround_dumps)) / len(swing_turnaround_dumps))

        line_width = 0.5

        for i in range(len(swing_turnaround_dumps) - 1):
            if i < len(swing_turnaround_dumps) // 2:
                bandpass_estimator_index = 0
            else:
                bandpass_estimator_index = 1
            bandpass_estimator = bandpass_estimators[bandpass_estimator_index]

            times = range(swing_turnaround_dumps[i], swing_turnaround_dumps[i + 1])
            flags = scan_data.flags.get(time=times,
                                        freq=target_channels,
                                        recv=receiver_index)
            mean_bandpass = scan_data.visibility.get(time=times,
                                                     freq=target_channels,
                                                     recv=receiver_index).mean(axis=0, flags=flags)
            bandpass = mean_bandpass.squeeze / mean_bandpass.squeeze[0]
            fit_frequencies = scan_data.frequencies.get(freq=target_channels).squeeze / MEGA

            corrected = bandpass / bandpass_estimator.get(freq=target_channels, recv=receiver_index).squeeze  # this should be constant at 1
            corrected = self.correct_linear(array=corrected, frequencies=fit_frequencies)

            residual = (corrected - 1) * 100

            ax[0].plot(fit_frequencies,
                       corrected,
                       label=f'swing {i}',
                       c=colors[i],
                       lw=line_width)
            ax[1].plot(fit_frequencies,
                       residual,
                       label=f'swing {i}',
                       c=colors[i],
                       lw=line_width)

        ax[0].set_xlabel('frequency [MHz]')
        ax[0].set_ylabel('corrected intensity')
        ax[1].set_xlabel('frequency [MHz]')
        ax[1].set_ylabel('correction residual [%]')
        plot_name = 'standing_wave_correction_scanning_swings.png'
        plt.savefig(os.path.join(receiver_path, plot_name))
        plt.close()

    def correct_linear(self, array: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        """
        Remove any slope from `array` using `scipy.optimize.curve_fit` and return the result.
        :param array: `numpy` array depending on `frequencies`
        :param frequencies: in MHz
        :return: `array` with a linear slope removed
        """
        line_fit = scipy.optimize.curve_fit(self.line, frequencies, array, p0=(1, 0))
        line = self.line(frequencies, *line_fit[0])
        return array / line

    @staticmethod
    def line(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """ Return line with offset `b` and gradient `a` at value `x`. """
        return a * x + b

    def plot_azimuth_bins(self,
                          scan_data,
                          i_receiver,
                          antenna_index,
                          target_channels,
                          frequencies,
                          bandpass_estimators,
                          receiver_path):
        """ Old plotting function to check for azimuth dependence. """

        azimuth_digitized, azimuth_bins = self.azimuth_digitizer(azimuth=scan_data.azimuth.get(recv=antenna_index))
        corrected_azimuth_binned_bandpasses = []
        for index in range(len(azimuth_bins)):
            time_dumps = np.where(azimuth_digitized == index)[0]
            visibility = scan_data.visibility.get(time=time_dumps,
                                                  freq=target_channels,
                                                  recv=i_receiver)
            flag = scan_data.flags.get(time=time_dumps,
                                       freq=target_channels,
                                       recv=i_receiver)
            corrected_azimuth_binned_bandpasses.append(
                visibility.mean(axis=0, flags=flag).squeeze / bandpass_estimators[0].get(freq=target_channels,
                                                                                         recv=i_receiver).squeeze
                )

        plt.figure(figsize=(8, 6))
        for i, bandpass in enumerate(corrected_azimuth_binned_bandpasses):
            label = ''
            if i % 7 == 0:
                label = f'azimuth {azimuth_bins[i]:.1f}'
            plt.plot(frequencies.squeeze / MEGA, bandpass, label=label)

        plt.legend()
        plt.xlabel('frequency [MHz]')
        plt.ylabel('intensity')
        plot_name = 'standing_wave_correction_scanning_azimuth_bins.png'
        plt.savefig(os.path.join(receiver_path, plot_name))
        plt.close()
