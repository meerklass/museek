import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.data_element import DataElement
from museek.enum.result_enum import ResultEnum
from museek.flag_factory import FlagFactory
from museek.time_ordered_data import TimeOrderedData
from museek.visualiser import plot_time_ordered_data_map


class ZebraRemoverPlugin(AbstractPlugin):
    def __init__(self,
                 reference_frequency: float,
                 gsm_downlink_range: tuple[float, float],
                 do_create_maps_of_frequency: bool,
                 satellite_free_dump_dict: dict,
                 grid_size: tuple[int, int] = (60, 60)):
        """
        Initialise
        :param reference_channel: the index of the reference channel, should be mostly rfi free before flagging
        :param zebra_channels: `list` or `range` of channel indices affected by the emission from the vanwyksvlei tower
        """
        super().__init__()
        self.reference_frequency = reference_frequency
        self.gsm_downlink_range = gsm_downlink_range
        self.do_create_maps_of_frequency = do_create_maps_of_frequency
        self.satellite_free_dump_dict = satellite_free_dump_dict
        self.grid_size = grid_size

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, scan_data: TimeOrderedData, output_path: str):
        mega = 1e6
        best_fit_gradient = 5.6
        scan_data.load_visibility_flags_weights()
        timestamp_dates = scan_data.timestamp_dates.squeeze

        # mask point sources
        # the same for all dishes
        # TODO: this is NOT the same for all dishes, but the point source masks must be created individually
        #  for each dish and then added up
        point_source_mask = FlagFactory().get_point_source_mask(shape=scan_data.visibility.shape,
                                                                right_ascension=scan_data.right_ascension,
                                                                declination=scan_data.declination)
        scan_data.flags.add_flag(point_source_mask)

        # set rfi free channels
        # rfi_free_channels = [3000, 3001]
        # rfi_free_channels = [600, 601]  # cosmology target
        rfi_free_frequency_edges = [981.390625, 981.599609375]

        # reference_frequency = scan_data.frequencies.get(freq=self.reference_channel).squeeze
        reference_channel = self.channel_from_frequency(frequency=self.reference_frequency,
                                                        frequencies=scan_data.frequencies)

        # manually remove the satellites:
        start_index, end_index = self.satellite_free_dump_dict[scan_data.name.split('_')[0]]
        if end_index == 'end':
            end_index = len(timestamp_dates)
        times = range(start_index, end_index)

        # fit a straight line to the scatter plot
        # def fitting_function(parameter, offset, gradient_):
        #     return self.fitting_wrapper(parameter=parameter,
        #                                 offset=offset,
        #                                 gradient=gradient_,
        #                                 repetitions=rfi_free_visibility.shape[1],
        #                                 function=self.straight_line).flatten()

        def fitting_function(parameter, a, b, c):
            return self.fitting_wrapper(parameter=parameter,
                                        function=self.half_polynomial_half_constant,
                                        repetitions=rfi_free_visibility.shape[1],
                                        a=a,
                                        b=b,
                                        c=c).flatten()

        for i_receiver, receiver in enumerate(scan_data.receivers):
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            antenna = scan_data.antenna(receiver=receiver)
            i_antenna = scan_data.antennas.index(antenna)
            channel_visibility = scan_data.visibility.get(recv=i_receiver, time=times, freq=reference_channel)
            right_ascension = scan_data.right_ascension.get(recv=i_antenna, time=times)

            #############################3
            # fix for UHF
            def convert_right_ascension(right_ascension_: np.ndarray) -> np.ndarray:
                return np.asarray([[timestamp_ra if 0 < timestamp_ra < 180 else timestamp_ra + 360
                                    for timestamp_ra in dish_ra]
                                   for dish_ra in right_ascension_])
            right_ascension = DataElement(convert_right_ascension(right_ascension_=right_ascension.squeeze[:,np.newaxis, np.newaxis]))

            #####################################



            declination = scan_data.declination.get(recv=i_antenna, time=times)
            flags = scan_data.flags.get(recv=i_receiver, time=times, freq=reference_channel)

            print(f'{receiver.name} flag sums are {sum(flags._flags[0].squeeze)} and {sum(flags._flags[1].squeeze)}')

            frequencies = scan_data.frequencies.squeeze
            zebra_channel_a = self.channel_from_frequency(self.gsm_downlink_range[0], frequencies=scan_data.frequencies)
            zebra_channel_b = self.channel_from_frequency(self.gsm_downlink_range[1], frequencies=scan_data.frequencies)
            zebra_channels = range(zebra_channel_a, zebra_channel_b)
            zebra_frequencies = [frequencies[channel] for channel in zebra_channels]
            zebra_visibility = scan_data.visibility.get(freq=zebra_channels, time=times, recv=i_receiver)
            zebra_power = np.trapz(zebra_visibility.squeeze, x=zebra_frequencies, axis=1)
            zebra_power_normalizer = 1e11


            rfi_free_channel_a = self.channel_from_frequency(frequency=rfi_free_frequency_edges[0], frequencies=scan_data.frequencies)
            rfi_free_channel_b = self.channel_from_frequency(frequency=rfi_free_frequency_edges[1], frequencies=scan_data.frequencies)
            rfi_free_channels = list(range(rfi_free_channel_a, rfi_free_channel_b+1))

            rfi_free_visibility = scan_data.visibility.get(freq=rfi_free_channels, time=times, recv=i_receiver)
            rfi_free_frequencies = [frequencies[channel] for channel in rfi_free_channels]
            rfi_free_normalizer = rfi_free_visibility.mean(axis=(0, 1)).squeeze

            np.savez(os.path.join(receiver_path, 'gain_model_fit_data.npz'),
                     zebra_power=zebra_power,
                     rfi_free_visibility=rfi_free_visibility._array,
                     right_ascension=right_ascension._array,
                     declination=declination._array,
                     flags=[flag._array for flag in flags._flags],
                     rfi_free_frequencies=np.asarray(rfi_free_frequencies))

            extent = [0,
                      len(scan_data.timestamps.squeeze),
                      scan_data.frequencies.get(freq=-1).squeeze / mega,
                      scan_data.frequencies.get(freq=0).squeeze / mega]

            plt.imshow(scan_data.visibility.get(recv=i_receiver).squeeze.T,
                       aspect='auto',
                       extent=extent)
            plt.axhline(scan_data.frequencies.get(freq=reference_channel).squeeze / mega,
                        xmin=times[0] / len(timestamp_dates),
                        xmax=times[-1] / len(timestamp_dates))
            plt.axhline(scan_data.frequencies.get(freq=rfi_free_channels[0]).squeeze / mega,
                        xmin=times[0] / len(timestamp_dates),
                        xmax=times[-1] / len(timestamp_dates))
            plt.axhline(scan_data.frequencies.get(freq=rfi_free_channels[-1]).squeeze / mega,
                        xmin=times[0] / len(timestamp_dates),
                        xmax=times[-1] / len(timestamp_dates))
            plt.xlabel('dump index')
            plt.ylabel('frequency [MHz]')
            plt.savefig(os.path.join(receiver_path, f'waterfall.png'))
            plt.close()

            # self.plot_band_pass(i_receiver=i_receiver,
            #                     scan_data=scan_data,
            #                     zebra_power=zebra_power,
            #                     times=times,
            #                     receiver_path=receiver_path)

            fit = curve_fit(f=fitting_function,
                            xdata=zebra_power / zebra_power_normalizer,
                            ydata=rfi_free_visibility.squeeze.flatten() / rfi_free_normalizer,
                            p0=[333., -13., 25.])

            line_for_plot = self.half_polynomial_half_constant(zebra_power / zebra_power_normalizer, *fit[0])
            line_for_fix = self.half_polynomial_half_constant(zebra_power / zebra_power_normalizer, 1, *fit[0][1:])
            if any(line_for_fix < 1):
                print('WARNING, zebra cleaning seems to add new power to the signal.')

            killed_zebra = channel_visibility * (1 / line_for_fix[:, np.newaxis, np.newaxis])

            if self.do_create_maps_of_frequency:
                for i_channel, channel in enumerate(zebra_channels):
                    plt.figure(figsize=(6, 12))
                    plt.subplot(2, 1, 1)
                    extent = [scan_data.timestamps[start_index],
                              scan_data.timestamps[end_index - 1],
                              scan_data.frequencies.get(freq=zebra_channels[-1]).squeeze / mega,
                              scan_data.frequencies.get(freq=zebra_channels[0]).squeeze / mega]
                    image = plt.imshow(scan_data.visibility.get(recv=i_receiver,
                                                                freq=zebra_channels,
                                                                time=times).squeeze.T,
                                       aspect='auto',
                                       norm='log',
                                       cmap='gist_ncar',
                                       extent=extent)
                    plt.colorbar(image)
                    plt.axhline(scan_data.frequencies.get(freq=channel).squeeze / mega, color='red')

                    plt.subplot(2, 1, 2)
                    plot_time_ordered_data_map(right_ascension=right_ascension,
                                               declination=declination,
                                               visibility=scan_data.visibility.get(freq=channel,
                                                                                   time=times,
                                                                                   recv=i_receiver),
                                               grid_size=self.grid_size,
                                               flags=flags)

                    plt.savefig(os.path.join(
                        receiver_path,
                        f'map_of_frequency{scan_data.frequencies.get(freq=channel).squeeze / mega:.0f}MHz.png')
                    )
                    plt.close()

            azimuth = scan_data.azimuth.get(recv=i_antenna, time=times).squeeze
            plt.figure(figsize=(18, 12))
            plt.subplot(2, 3, 1)
            plt.title('zebra power map')
            plot_time_ordered_data_map(right_ascension=right_ascension,
                                       declination=declination,
                                       visibility=DataElement(array=zebra_power[:, np.newaxis, np.newaxis]),
                                       flags=flags,
                                       grid_size=self.grid_size)
            plt.xlabel('Right ascension')
            plt.ylabel('Declination')

            plt.subplot(2, 3, 2)
            plt.plot(azimuth, zebra_power)
            plt.xlabel('azimuth [deg]')
            plt.ylabel('zebra power')

            plt.subplot(2, 3, 3)
            extent = [scan_data.timestamps[start_index],
                      scan_data.timestamps[end_index - 1],
                      scan_data.frequencies.get(freq=zebra_channels[-1]).squeeze / mega,
                      scan_data.frequencies.get(freq=zebra_channels[0]).squeeze / mega]
            image = plt.imshow(scan_data.visibility.get(recv=i_receiver,
                                                        freq=zebra_channels,
                                                        time=times).squeeze.T,
                               aspect='auto',
                               extent=extent,
                               cmap='gist_ncar',
                               norm='log')
            plt.colorbar(image)
            plt.xlabel('timestamp [s]')
            plt.ylabel('frequency [MHz]')
            plt.title('"zebra" channels')

            plt.subplot(2, 3, 4)
            plot_time_ordered_data_map(right_ascension=right_ascension,
                                       declination=declination,
                                       visibility=channel_visibility,
                                       flags=flags,
                                       grid_size=self.grid_size)
            plt.xlabel('Right ascension')
            plt.ylabel('Declination')
            plt.title(f'raw visibility {self.reference_frequency / mega :.1f} MHz')

            plt.subplot(2, 3, 5)
            plot_time_ordered_data_map(right_ascension=right_ascension,
                                       declination=declination,
                                       visibility=killed_zebra,
                                       flags=flags,
                                       grid_size=self.grid_size)
            plt.xlabel('Right ascension')
            plt.ylabel('Declination')
            plt.title(f'model parameters {fit[0]}')
            print(f'model parameters {fit[0]}')

            zebra_power_sort_args = np.argsort(zebra_power)
            plt.subplot(2, 3, 6)
            for i in range(rfi_free_visibility.shape[1]):
                plt.scatter(zebra_power,
                            rfi_free_visibility.squeeze[:, i] / rfi_free_normalizer,
                            color='black',
                            s=0.01)
            plt.plot(zebra_power[zebra_power_sort_args],
                     line_for_plot[zebra_power_sort_args],
                     color='black',
                     label='uncorrected')

            for i in range(rfi_free_visibility.shape[1]):
                plt.scatter(zebra_power,
                            rfi_free_visibility.squeeze[:, i] / line_for_fix / rfi_free_normalizer,
                            color='red',
                            s=0.1)
            plt.plot(zebra_power[zebra_power_sort_args],
                     line_for_plot[zebra_power_sort_args] / line_for_fix[zebra_power_sort_args],
                     color='red',
                     label='excess power removed')
            plt.xlabel(
                f'Power integrated from {zebra_frequencies[0] / 1e6:.0f} to {zebra_frequencies[-1] / 1e6:.0f} MHz')
            plt.ylabel(f'Raw signal from {rfi_free_frequencies[0] / 1e6:.0f} to {rfi_free_frequencies[1] / 1e6:.0f}'
                       f' MHz, mostly RFI free')
            plt.legend()

            plt.savefig(os.path.join(receiver_path, f'zebra_correction_matrix_plot.png'))
            plt.close()

            # # for i, gradient in enumerate(np.linspace(20, 35)):
            # for i, gradient in enumerate(np.linspace(3, 6)):
            #     # line_ = self.two_lines(zebra_power / zebra_power_max, fit[0][0], gradient, 34, fit[0][3])
            #
            #     line_ = self.straight_line(zebra_power * 1e-10, fit[0][0], gradient)
            #     normalized_line = line_ / line_[np.argmin(zebra_power)]
            #
            #     killed_zebra = channel_visibility * (1 / normalized_line)[:, np.newaxis, np.newaxis]
            #     plot_time_ordered_data_map(right_ascension=right_ascension,
            #                                declination=declination,
            #                                visibility=killed_zebra,
            #                                flags=flags,
            #                                grid_size=self.grid_size)
            #     plt.title(f'line gradient {gradient:.3f}')
            #     plot_name = f'zebra_removal_{i}.png'
            #     plt.savefig(os.path.join(receiver_path, plot_name))
            #     plt.close()

    @staticmethod
    def straight_line(parameter, offset, gradient):
        return offset + gradient * parameter

    @staticmethod
    def fitting_wrapper(parameter, repetitions: int, function, **kwargs):
        line_ = function(parameter=parameter, **kwargs)
        return np.tile(line_[:, np.newaxis], (1, repetitions))

    @staticmethod
    def polynomial_2(parameter, a, b, c):
        return a + b * parameter + c * parameter ** 2

    def half_polynomial_half_constant(self, parameter, a, b, c):
        result = self.polynomial_2(parameter=parameter, a=a, b=b, c=c)
        if c != 0:
            minimum_x = -b / c / 2
            result[parameter <= minimum_x] = self.polynomial_2(parameter=minimum_x, a=a, b=b, c=c)
        return result

    @staticmethod
    def plot_band_pass(i_receiver: int,
                       scan_data: TimeOrderedData,
                       zebra_power: np.ndarray,
                       times: range,
                       receiver_path: str):
        mega = 1e6

        turn_on_start = 710
        turn_on_end = 730

        zebra_mean_tower_on = np.mean(zebra_power[turn_on_end:])
        zebra_power_high_indices = np.where(zebra_power > zebra_mean_tower_on)[0]

        # clean_channels=range(2700, 3150)
        clean_channels = range(570, 765)  # cosmology target

        extent = [scan_data.timestamps[times[0]],
                  scan_data.timestamps[times[-1]],
                  scan_data.frequencies.get(freq=clean_channels[-1]).squeeze / mega,
                  scan_data.frequencies.get(freq=clean_channels[0]).squeeze / mega]
        image = plt.imshow(scan_data.visibility.get(recv=i_receiver,
                                                    freq=clean_channels,
                                                    time=times).squeeze.T,
                           aspect='auto',
                           extent=extent,
                           cmap='gist_ncar',
                           norm='log')
        plt.colorbar(image)
        plt.xlabel('timestamp [s]')
        plt.ylabel('frequency [MHz]')
        plt.title('"clean" channels')
        plt.savefig(os.path.join(receiver_path, 'waterfall_target_channels.png'))
        plt.close()

        clean_tower_off = scan_data.visibility.get(recv=i_receiver,
                                                   freq=clean_channels,
                                                   time=times[:turn_on_start])

        clean_tower_on = scan_data.visibility.get(recv=i_receiver,
                                                  freq=clean_channels,
                                                  time=times[turn_on_end:])

        clean_tower_high = scan_data.visibility.get(recv=i_receiver,
                                                    freq=clean_channels,
                                                    time=[times[i] for i in zebra_power_high_indices])

        bandpass_tower_off = clean_tower_off.mean(axis=0).squeeze
        bandpass_tower_on = clean_tower_on.mean(axis=0).squeeze
        bandpass_tower_high = clean_tower_high.mean(axis=0).squeeze

        plt.figure(figsize=(6, 12))
        plt.subplot(2, 1, 1)
        plt.plot(scan_data.frequencies.get(freq=clean_channels).squeeze[1:] / mega, bandpass_tower_off[1:],
                 label='tower off')
        plt.plot(scan_data.frequencies.get(freq=clean_channels).squeeze[1:] / mega, bandpass_tower_on[1:],
                 label='tower on')
        plt.plot(scan_data.frequencies.get(freq=clean_channels).squeeze[1:] / mega, bandpass_tower_high[1:],
                 label='tower extra high')
        plt.xlabel('frequency [MHz]')
        plt.ylabel('mean bandpass')
        plt.legend()

        bandpass_diff = (bandpass_tower_on - bandpass_tower_off) / bandpass_tower_off * 100
        plt.subplot(2, 1, 2)
        plt.plot(scan_data.frequencies.get(freq=clean_channels).squeeze[1:] / mega, bandpass_diff[1:])
        plt.xlabel('frequency [MHz]')
        plt.ylabel('bandpass (tower on - tower off) / tower off %')
        plt.savefig(os.path.join(receiver_path, 'bandpass_during_scanning.png'))
        plt.close()

    @staticmethod
    def channel_from_frequency(frequency: float, frequencies: DataElement) -> int:
        return np.atleast_1d(np.argmin(abs(frequencies.get().squeeze - frequency*1e6)))[0]
