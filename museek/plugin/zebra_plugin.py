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

        # for receiver in data.receivers:
        #     self.create_visibility_fft_plot(data=data, receiver=receiver, output_path=output_path)

        # self.create_turnaround_plots(data=data, output_path=output_path)
        #
        # self.create_visibility_vs_azimuth_plots(data=data,
        #                                         output_path=output_path,
        #                                         reference_channel=self.zebra_channel)

        self.create_plots_integrated_power(data=data,
                                           output_path=output_path,
                                           zebra_channel=self.config.zebra_channel)

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

                swing_visibility = data.visibility.get(freq=reference_channel,
                                                       recv=receiver_index).scan[start: end]
                swing_azimuth = azimuth[start: end]
                plt.subplot(311)
                plt.plot(swing_azimuth, swing_visibility / np.mean(swing_visibility), label=receiver.name)

                zebra_scale = self.suspected_zebra_peaks[1] - self.suspected_zebra_peaks[0]
                fft_visibility, fft_freq = self.get_fft_and_freq(azimuth=swing_azimuth,
                                                                 visibility=swing_visibility[:, np.newaxis])
                zebra_index = self.zebra_index(fft=fft_visibility, freq=fft_freq, inverse_zebra_scale=1 / zebra_scale)
                plt.subplot(312)
                plt.plot(fft_freq[2:], np.abs(fft_visibility[2:]), label=receiver.name)
                plt.axvline(1 / zebra_scale, color='black', ls=':')
                plt.scatter(1/zebra_scale, zebra_index, marker='x', color='black')
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
        if np.abs(fft[phase_index]) <= 10 * np.mean(np.abs(fft[phase_index + 1:])):
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

    def create_plots_integrated_power(self, data: TimeOrderedData, output_path: str, zebra_channel: int):
        """
        DOC
        """
        # times = range(1600, len(data.timestamps.scan))
        times = data.scan_dumps

        visibility = data.visibility.get_array(recv=0, time=times)
        frequencies = data.frequencies.get_array()
        frequency_bin_width = frequencies[1] - frequencies[0]
        total_power = np.trapz(visibility, x=frequencies, axis=1)

        # zebra_channels = range(475, 498)
        zebra_channels = range(350, 498)
        # satelite_channels = range(1350, 2100)
        rfi_free_channels = range(2500, 3000)

        zebra_visibility = data.visibility.get_array(freq=zebra_channels, time=times)
        zebra_frequencies = [frequencies[channel] for channel in zebra_channels]
        zebra_power = np.trapz(zebra_visibility, x=zebra_frequencies, axis=1)

        rfi_free_visibility = data.visibility.get_array(freq=rfi_free_channels, time=times)
        rfi_free_frequencies = [frequencies[channel] for channel in rfi_free_channels]
        rfi_free_power = np.trapz(rfi_free_visibility, x=rfi_free_frequencies, axis=1)

        zebra_frequency_fraction = (frequencies[zebra_channels[-1]] - frequencies[zebra_channels[0]]) / (
                frequencies[-1] - frequencies[0])

        # plt.semilogy(data.timestamp_dates.scan, total_power)
        # plt.semilogy(data.timestamp_dates.scan, zebra_power)

        plt.plot(data.timestamp_dates.get_array(time=times), zebra_power / total_power)
        plt.show()


        start_index = 2000
        end_index = -10
        mean_zebra = np.mean(zebra_power[start_index:end_index])
        mean_total = np.mean(total_power[start_index:end_index])
        mean_rfi_free = np.mean(rfi_free_power[start_index:end_index])
        zebra_power_ratio = zebra_power / total_power
        zebra_power_ratio_mean = np.mean(zebra_power_ratio[start_index:end_index])

        zebra_normalized = (zebra_power - mean_zebra) / np.max(zebra_power[start_index:end_index] - mean_zebra)

        total_normalized = (total_power - mean_total) / np.max(total_power[start_index:end_index] - mean_total)
        zebra_ratio_normalized = (zebra_power_ratio - zebra_power_ratio_mean) / (
            np.max(zebra_power_ratio[start_index:end_index] - zebra_power_ratio_mean)
        )
        # rfi_free_normalized = rfi_free_power / np.max(rfi_free_power[start_index:end_index])
        rfi_free_normalized = (rfi_free_power - mean_rfi_free) / np.max(
            abs(rfi_free_power[start_index:end_index] - mean_rfi_free))
        # rfi_free_normalized *= np.max(zebra_power_ratio[start_index:end_index])
        # rfi_free_normalized = rfi_free_normalized + np.mean

        plt.figure(figsize=(16, 6))
        # plt.plot(data.timestamp_dates.get_array(time=times),zebra_normalized,label='zebra power')
        # plt.plot(data.timestamp_dates.get_array(time=times),total_normalized,label='total power')
        plt.plot(data.timestamp_dates.get_array(time=times), rfi_free_normalized, label='mostly rfi free power')
        plt.plot(data.timestamp_dates.get_array(time=times),
                 zebra_ratio_normalized,
                 label='power from 958 MHz RFI / total power')
        plt.xlim((data.timestamp_dates.get_array(time=times)[start_index],
                  data.timestamp_dates.get_array(time=times)[end_index]))
        plt.ylim((-1.2, 1.2))
        # plt.ylim((0.0, 0.3))
        plt.legend()
        plt.show()

        plt.scatter(rfi_free_power[start_index:end_index], zebra_power_ratio[start_index:end_index], color='black', s=0.1)
        plt.xlabel('RFI free power')
        plt.ylabel('Fraction of total power from channels around 958 MHz')
        # plt.xlim((2.1e10, 2.3e10))
        plt.show()

        import scipy

        correlation = scipy.signal.correlate(rfi_free_power[start_index:end_index],
                                             zebra_power_ratio[start_index + 100:start_index + 300], 'valid')
        lags = scipy.signal.correlation_lags(len(rfi_free_power[start_index:end_index]),
                                             len(zebra_power_ratio[start_index + 100:end_index + 300]), 'valid')
        correlation /= max(correlation)
        plt.plot(lags, correlation)
        plt.show()

        plt.imshow(data.visibility.get_array(time=times).T, aspect='auto', norm='log')
        plt.axhline(zebra_channels[0], color='black')
        plt.axhline(zebra_channels[-1], color='black')
        plt.show()

        test_channels = range(370, 500)
        plt.imshow(data.visibility.get_array(time=times, freq=test_channels).T, aspect='auto', norm='log', cmap='gray')

    @staticmethod
    def zebra_index(fft, freq, inverse_zebra_scale) -> float:
        arg_min = np.argmin(abs(freq-inverse_zebra_scale))
        return abs(fft[arg_min])

    
