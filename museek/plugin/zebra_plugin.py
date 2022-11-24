import itertools
import os
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.struct import Struct
from museek.enum.result_enum import ResultEnum
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData


class ZebraPlugin(AbstractPlugin):
    suspected_zebra_peaks = [102.5, 105, 107.5, 110.25]

    def __init__(self, ctx: Struct):
        super().__init__(ctx=ctx)
        self.zebra_channel = self.config.zebra_channel

    def set_requirements(self):
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, data: TimeOrderedData, output_path: str):

        data.load_visibility_flags_weights()

        for receiver in data.receivers:
            self.create_visibility_fft_plot(data=data, receiver=receiver, output_path=output_path)

        # self.create_turnaround_plots(data=data, output_path=output_path)
        #
        # self.create_visibility_vs_azimuth_plots(data=data,
        #                                         output_path=output_path,
        #                                         reference_channel=self.zebra_channel)

    @staticmethod
    def get_turn_around_dumps(d_azimuth_d_time: np.ndarray) -> np.ndarray:
        derivative_changes_sign = [d_azimuth_d_time[i] * d_azimuth_d_time[i + 1] < 0
                                   for i in range(len(d_azimuth_d_time) - 1)]
        turnaround_dumps = np.argwhere(derivative_changes_sign)
        if turnaround_dumps[0] != 0:
            turnaround_dumps = np.insert(turnaround_dumps, 0, 0)
        return turnaround_dumps

    @staticmethod
    def get_d_azimuth_d_time(timestamps: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
        return np.array([(y2 - y0) / (x2 - x0) for x2, x0, y2, y0 in zip(timestamps[2:],
                                                                         timestamps,
                                                                         azimuth[2:],
                                                                         azimuth)])

    @staticmethod
    def get_swing_direction(d_azimuth_d_time: np.ndarray, index: int) -> str:
        if d_azimuth_d_time[index] > 0:
            return 'right'
        return 'left'

    def create_turnaround_plots(self, data: TimeOrderedData, output_path: str):

        for antenna_index, antenna in enumerate(data.antennas):
            azimuth = data.azimuth.get(recv=antenna_index).scan
            d_azimuth_d_time = self.get_d_azimuth_d_time(timestamps=data.timestamps.scan, azimuth=azimuth)
            turn_around_dumps = self.get_turn_around_dumps(d_azimuth_d_time=d_azimuth_d_time)
            plt.figure(figsize=(12, 4))
            plt.plot(data.timestamp_dates.scan, azimuth)
            for dump in turn_around_dumps:
                plt.axvline(data.timestamp_dates.scan[dump])
            plt.savefig(os.path.join(output_path, f'zebra_{antenna.name}_azimuth_swing_turnaround.png'))
            plt.close()

    def create_visibility_vs_azimuth_plots(
            self,
            data: TimeOrderedData,
            reference_channel: int,
            output_path: str
    ):

        n_turn_around = self.get_n_turn_around(data=data)
        count = itertools.count()
        swing_direction = 'right'
        start = 0

        for start_index in range(n_turn_around - 1):
            plt.figure(figsize=(12, 12))

            for receiver_index, receiver in enumerate(data.receivers):
                antenna_index = data.antenna_index_of_receiver(receiver=receiver)
                azimuth = data.azimuth.get(recv=antenna_index).scan
                d_azimuth_d_time = self.get_d_azimuth_d_time(timestamps=data.timestamps.scan, azimuth=azimuth)
                turn_around_dumps = self.get_turn_around_dumps(d_azimuth_d_time=d_azimuth_d_time)

                start = turn_around_dumps[start_index]
                end = turn_around_dumps[start_index + 1]
                if start != 0:  # otherwise these indices are plotted twice
                    start += 1

                swing_direction = self.get_swing_direction(d_azimuth_d_time=d_azimuth_d_time, index=start)

                visibility_to_plot = data.visibility.get(freq=reference_channel,
                                                         recv=receiver_index).scan[start: end]
                plt.subplot(311)
                plt.plot(azimuth[start: end], visibility_to_plot / np.mean(visibility_to_plot), label=receiver.name)

                fft_visibility, fft_freq = self.get_fft_and_freq(azimuth=azimuth[start: end],
                                                                 visibility=visibility_to_plot[:, np.newaxis])
                plt.subplot(312)
                plt.plot(fft_freq[2:], np.abs(fft_visibility[2:]), label=receiver.name)
                plt.axvline(1 / (self.suspected_zebra_peaks[1] - self.suspected_zebra_peaks[0]), color='black', ls=':')
                plt.xlabel('~1/az')
                plt.ylabel('fft')

                plt.subplot(313)
                plt.plot(fft_freq[2:], np.angle(fft_visibility[2:]), label=receiver.name)
                plt.axvline(1 / (self.suspected_zebra_peaks[1] - self.suspected_zebra_peaks[0]), color='black', ls=':')
                plt.ylabel('phase [deg]')
                plt.xlabel('~1/az')

            plt.subplot(311)
            for peak in self.suspected_zebra_peaks:
                plt.axvline(peak, color='black', ls=':')

            plt.legend()
            plt.title(f'Swings {swing_direction}, starting at {data.timestamp_dates.scan[start]}')
            plt.ylabel('visibility')
            plt.xlabel('az')
            plt.savefig(
                os.path.join(output_path,
                             f'zebra_ch{reference_channel}_swing_{swing_direction}_{next(count)}.png')
            )
            plt.close()

    def get_n_turn_around(self, data: TimeOrderedData) -> int:
        azimuth = data.azimuth.get(recv=0).scan
        d_azimuth_d_time = self.get_d_azimuth_d_time(timestamps=data.timestamps.scan, azimuth=azimuth)
        return len(self.get_turn_around_dumps(d_azimuth_d_time=d_azimuth_d_time))

    def create_visibility_fft_plot(self, receiver: Receiver, data: TimeOrderedData, output_path: str):
        antenna_index = data.antenna_index_of_receiver(receiver=receiver)
        receiver_index = data.receivers.index(receiver)
        visibility = data.visibility.get(recv=receiver_index).scan
        azimuth = data.azimuth.get(recv=antenna_index).scan
        d_azimuth_d_time = self.get_d_azimuth_d_time(timestamps=data.timestamps.scan, azimuth=azimuth)
        turn_around_dumps = self.get_turn_around_dumps(d_azimuth_d_time=d_azimuth_d_time)

        count = itertools.count()
        for start, end in zip(turn_around_dumps[:-1], turn_around_dumps[1:]):
            fft_visibility, fft_freq = self.get_fft_and_freq(azimuth=azimuth[start:end],
                                                             visibility=visibility[start: end])
            swing_direction = self.get_swing_direction(d_azimuth_d_time=d_azimuth_d_time, index=start)

            inverse_zebra_scale = 1 / (self.suspected_zebra_peaks[1] - self.suspected_zebra_peaks[0])
            zebra_scale_index = np.argmin(abs(fft_freq - inverse_zebra_scale))

            if next(copy(count)) == 35:
                self.save_zebra_phase_and_position(data=data,
                                                   receiver=receiver,
                                                   phase_index=zebra_scale_index,
                                                   fft=fft_visibility,
                                                   output_path=output_path)

            fft_visibility = np.abs(fft_visibility[2:])
            fft_freq = fft_freq[2:]
            for i_frequency in range(visibility.shape[1]):
                fft_mean = np.mean(fft_visibility[:, i_frequency])
                if fft_mean != 0:
                    fft_visibility[:, i_frequency] = fft_visibility[:, i_frequency] / fft_mean

            zebra_scale_index = np.argmin(abs(fft_freq - inverse_zebra_scale))

            plt.imshow(fft_visibility.T, aspect='auto')
            xticks = range(len(fft_freq))[::5][:-1]
            plt.xticks(xticks, np.round(fft_freq[::5][:-1], 2))
            plt.axvline(zebra_scale_index, color='black', ls=':')
            plt.axhline(488, color='black', ls=':', alpha=0.5)
            plt.xlabel('~1/az')
            plt.ylabel('channel')
            start_time = data.timestamp_dates[start]
            plt.title(f'fft visibility. swing {swing_direction} at {start_time}')

            plt.savefig(
                os.path.join(output_path,
                             f'zebra_{receiver}_swing_{swing_direction}_{next(count)}.png')
            )
            plt.close()

    @staticmethod
    def get_fft_and_freq(visibility: np.ndarray, azimuth: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray]:
        """ DOC """
        n_visibility, n_frequency = visibility.shape
        channels = range(n_frequency)

        azimuth_sort = np.argsort(azimuth)
        azimuth = azimuth[azimuth_sort]
        visibility = visibility[azimuth_sort]

        interp_azimuth = np.linspace(azimuth[0], azimuth[-1], n_visibility)
        interp_visibility = np.zeros((n_visibility, n_frequency))

        for i_frequency in channels:
            interp_visibility[:, i_frequency] = np.interp(interp_azimuth,
                                                          azimuth,
                                                          visibility[:, i_frequency])

        fft_visibility = rfft(interp_visibility, axis=0)

        fft_freq = rfftfreq(interp_visibility.shape[0], abs(interp_azimuth[1] - interp_azimuth[0]))
        return fft_visibility, fft_freq

    def save_zebra_phase_and_position(self,
                                      data: TimeOrderedData,
                                      receiver: Receiver,
                                      phase_index,
                                      fft,
                                      output_path: str):
        antenna = data.antenna(receiver=receiver)
        position = antenna.position_wgs84

        fft = fft[:, self.zebra_channel]
        if np.abs(fft[phase_index]) <= 10 * np.mean(np.abs(fft[phase_index+1:])):
            print('Warning: the zebra mode should be dominant but is not.')
        phase = np.angle(fft)[phase_index]
        file_path = os.path.join(output_path, 'dish_phase_location.txt')

        line = f'{antenna.name} | {position} | {phase}\n'

        if os.path.exists(file_path):
            with open(file_path, 'r') as in_file:
                lines = in_file.readlines()
        else:
            lines = []

        if line in lines:
            return

        line_found = False
        for i, line_from_file in enumerate(lines):
            if antenna.name in line_from_file:
                lines[i] = line
                line_found = True
        if not line_found:
            lines.append(line)

        with open(file_path, 'w+') as out_file:
            for line in lines:
                out_file.write(line)
