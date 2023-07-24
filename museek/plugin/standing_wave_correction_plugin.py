import os
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.data_element import DataElement
from museek.enum.result_enum import ResultEnum
from museek.plugin.bandpass_plugin import MEGA
from museek.time_ordered_data import TimeOrderedData


class StandingWaveCorrectionPlugin(AbstractPlugin):

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [
            Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
            Requirement(location=ResultEnum.STANDING_WAVE_CHANNELS, variable='target_channels'),
            Requirement(location=ResultEnum.STANDING_WAVE_EPSILON_FUNCTION_DICT, variable='epsilon_function_dict')
        ]

    def run(self,
            scan_data: TimeOrderedData,
            output_path: str,
            target_channels: range | list[int],
            epsilon_function_dict: dict[dict[Callable]]):
        """ DOC """
        before_or_after = 'before_scan'

        for i_receiver, receiver in enumerate(scan_data.receivers):
            print(f'Working on {receiver}...')
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            antenna_index = receiver.antenna_index(receivers=scan_data.receivers)
            frequencies = scan_data.frequencies.get(freq=target_channels)
            epsilon = epsilon_function_dict[receiver.name][before_or_after](frequencies)
            self.plot_individual_swings(scan_data=scan_data,
                                        antenna_index=antenna_index,
                                        i_receiver=i_receiver,
                                        before_or_after=before_or_after,
                                        target_channels=target_channels,
                                        epsilon=epsilon,
                                        receiver_path=receiver_path)

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
                corrected_azimuth_binned_bandpasses.append(visibility.mean(axis=0, flags=flag).squeeze / (1 + epsilon))

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

    @staticmethod
    def swing_turnaround_dumps(azimuth: DataElement) -> list[int]:
        """ DOC """
        sign = np.sign(np.diff(azimuth.squeeze))
        sign_change = ((np.roll(sign, 1) - sign) != 0).astype(bool)
        return np.where(sign_change)[0]

    @staticmethod
    def azimuth_digitizer(azimuth: DataElement) -> tuple[np.ndarray, np.ndarray]:
        """ DOC """
        bins = np.linspace(azimuth.min(axis=0).squeeze, azimuth.max(axis=0).squeeze, 50)
        return np.digitize(azimuth.squeeze, bins=bins), bins

    def plot_individual_swings(self,
                               scan_data,
                               antenna_index,
                               i_receiver,
                               before_or_after,
                               target_channels,
                               epsilon,
                               receiver_path):
        swing_turnaround_dumps = self.swing_turnaround_dumps(
            azimuth=scan_data.azimuth.get(recv=antenna_index)
        )
        fig = plt.figure(figsize=(8, 12))

        ax = fig.subplots(2, 1)
        ax[1].plot(scan_data.timestamp_dates.squeeze, scan_data.temperature.squeeze)

        for i, dump in enumerate(swing_turnaround_dumps):
            ax[1].text(scan_data.timestamp_dates.get(time=dump).squeeze,
                       scan_data.temperature.get(time=dump).squeeze,
                       f'{i}',
                       fontsize='x-small')
        if before_or_after == 'after_scan':
            correction_dump = scan_data.timestamps.shape[0] - 1
        else:
            correction_dump = 0
        ax[1].text(scan_data.timestamp_dates.get(time=correction_dump).squeeze,
                   scan_data.temperature.get(time=correction_dump).squeeze,
                   'model fit',
                   fontsize='small')

        ax[1].set_xlabel('time')
        ax[1].set_ylabel('temperature')

        for i in range(len(swing_turnaround_dumps) - 1):
            times = range(swing_turnaround_dumps[i], swing_turnaround_dumps[i + 1])
            flags = scan_data.flags.get(time=times,
                                        freq=target_channels,
                                        recv=i_receiver)
            mean_bandpass = scan_data.visibility.get(time=times,
                                                     freq=target_channels,
                                                     recv=i_receiver).mean(axis=0, flags=flags)
            frequencies = scan_data.frequencies.get(freq=target_channels)

            corrected = mean_bandpass.squeeze / (1 + epsilon)

            ax[0].plot(frequencies.squeeze / MEGA,
                       corrected,
                       label=f'swing {i}')
            if i % 5 == 0:
                ax[0].text(frequencies.squeeze[0] / MEGA,
                           corrected[0],
                           f'{i}',
                           fontsize='x-small')

        ax[0].set_xlabel('frequency [MHz]')
        ax[0].set_ylabel('intensity')
        plot_name = 'standing_wave_correction_scanning_swings.png'
        plt.savefig(os.path.join(receiver_path, plot_name))
        plt.close()
