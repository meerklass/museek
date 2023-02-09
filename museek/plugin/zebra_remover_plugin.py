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
                 reference_channel: int,
                 zebra_channels: range | list[int],
                 do_create_maps_of_frequency: bool,
                 satellite_free_dump_dict: dict,
                 grid_size: tuple[int, int] = (60, 60)):
        """
        Initialise
        :param reference_channel: the index of the reference channel, should be mostly rfi free before flagging
        :param zebra_channels: `list` or `range` of channel indices affected by the emission from the vanwyksvlei tower
        """
        super().__init__()
        self.reference_channel = reference_channel
        self.zebra_channels = zebra_channels
        self.do_create_maps_of_frequency = do_create_maps_of_frequency
        self.satellite_free_dump_dict = satellite_free_dump_dict
        self.grid_size = grid_size

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, scan_data: TimeOrderedData, output_path: str):
        mega = 1e6
        scan_data.load_visibility_flags_weights()
        timestamp_dates = scan_data.timestamp_dates.squeeze

        # mask point sources
        point_source_mask = FlagFactory().get_point_source_mask(shape=scan_data.visibility.shape,
                                                                right_ascension=scan_data.right_ascension,
                                                                declination=scan_data.declination)
        scan_data.flags.add_flag(point_source_mask)

        # set rfi free channels
        rfi_free_channels = [3000, 3001]

        # manually remove the satellites:
        start_index, end_index = self.satellite_free_dump_dict[scan_data.name.split('_')[0]]
        if end_index == 'end':
            end_index = len(timestamp_dates)
        times = range(start_index, end_index)

        # fit a straight line to the scatter plot
        def fitting_function(parameter, offset, gradient_):
            return self.straight_line_fitting_wrapper(parameter=parameter,
                                                      offset=offset,
                                                      gradient=gradient_,
                                                      repetitions=rfi_free_visibility.shape[1]).flatten()

        for i_receiver, receiver in enumerate(scan_data.receivers):
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            antenna = scan_data.antenna(receiver=receiver)
            i_antenna = scan_data.antennas.index(antenna)
            channel_visibility = scan_data.visibility.get(recv=i_receiver, time=times, freq=self.reference_channel)
            right_ascension = scan_data.right_ascension.get(recv=i_antenna, time=times)
            declination = scan_data.declination.get(recv=i_antenna, time=times)
            flags = scan_data.flags.get(recv=i_receiver, time=times, freq=self.reference_channel)

            frequencies = scan_data.frequencies.squeeze
            zebra_frequencies = [frequencies[channel] for channel in self.zebra_channels]
            zebra_visibility = scan_data.visibility.get(freq=self.zebra_channels, time=times, recv=i_receiver)
            zebra_power = np.trapz(zebra_visibility.squeeze, x=zebra_frequencies, axis=1)
            zebra_power_max = np.max(zebra_power)

            rfi_free_visibility = scan_data.visibility.get(freq=rfi_free_channels, time=times, recv=i_receiver)
            rfi_free_frequencies = [frequencies[channel] for channel in rfi_free_channels]

            extent = [0,
                      len(scan_data.timestamps.squeeze),
                      scan_data.frequencies.get(freq=-1).squeeze / mega,
                      scan_data.frequencies.get(freq=0).squeeze / mega]

            plt.imshow(scan_data.visibility.get(recv=i_receiver).squeeze.T,
                       aspect='auto',
                       extent=extent)
            plt.axhline(scan_data.frequencies.get(freq=self.reference_channel).squeeze / mega,
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

            fit = curve_fit(f=fitting_function,
                            xdata=zebra_power / zebra_power_max,
                            ydata=rfi_free_visibility.squeeze.flatten(),
                            p0=[0., 5.])
            line_ = self.straight_line(zebra_power / zebra_power_max, *fit[0])
            normalized_line = line_ / line_[np.argmin(zebra_power)]  # divide by the lowest rfi power value
            if any(normalized_line < 1):
                print('WARNING, zebra cleaning seems to add new power to the signal.')

            # for i in range(rfi_free_visibility.shape[1]):
            #     plt.scatter(zebra_power,
            #                 rfi_free_visibility.squeeze[:, i],
            #                 color='black',
            #                 s=0.01)
            # plt.plot(zebra_power, line_, color='black', label='uncorrected')
            #
            # for i in range(rfi_free_visibility.shape[1]):
            #     plt.scatter(zebra_power,
            #                 rfi_free_visibility.squeeze[:, i] / normalized_line,
            #                 color='red',
            #                 s=0.1)
            # plt.plot(zebra_power, line_ / normalized_line, color='red', label='excess power removed')
            # plt.xlabel(f'Power integrated from {zebra_frequencies[0] / 1e6:.0f} to {zebra_frequencies[-1] / 1e6:.0f} MHz')
            # plt.ylabel(f'Raw signal from {rfi_free_frequencies[0] / 1e6:.0f} to {rfi_free_frequencies[1] / 1e6:.0f}'
            #            f' MHz, mostly RFI free')
            # plt.legend()
            # plt.show()

            killed_zebra = channel_visibility * (1 / normalized_line[:, np.newaxis, np.newaxis])

            # plt.figure(figsize=(6, 18))
            # plt.subplot(2, 1, 1)
            # plot_time_ordered_data_map(right_ascension=right_ascension,
            #                            declination=declination,
            #                            visibility=killed_zebra,
            #                            flags=flags,
            #                            grid_size=self.grid_size)
            # plt.title('linear zebra model correction')
            # plt.subplot(2, 1, 2)
            # plot_time_ordered_data_map(right_ascension=right_ascension,
            #                            declination=declination,
            #                            visibility=channel_visibility,
            #                            flags=flags,
            #                            grid_size=self.grid_size)
            # plt.title('raw visibility')
            # plt.show()

            if self.do_create_maps_of_frequency:
                for i_channel, channel in enumerate(self.zebra_channels):
                    plt.figure(figsize=(6, 12))
                    plt.subplot(2, 1, 1)
                    extent = [scan_data.timestamps[start_index],
                              scan_data.timestamps[end_index - 1],
                              scan_data.frequencies.get(freq=self.zebra_channels[-1]).squeeze / mega,
                              scan_data.frequencies.get(freq=self.zebra_channels[0]).squeeze / mega]
                    image = plt.imshow(scan_data.visibility.get(recv=i_receiver,
                                                                freq=self.zebra_channels,
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

            plt.subplot(2, 3, 2)
            plt.plot(azimuth, zebra_power)
            plt.xlabel('azimuth [deg]')
            plt.ylabel('zebra power')

            plt.subplot(2, 3, 3)
            extent = [scan_data.timestamps[start_index],
                      scan_data.timestamps[end_index - 1],
                      scan_data.frequencies.get(freq=self.zebra_channels[-1]).squeeze / mega,
                      scan_data.frequencies.get(freq=self.zebra_channels[0]).squeeze / mega]
            image = plt.imshow(scan_data.visibility.get(recv=i_receiver,
                                                        freq=self.zebra_channels,
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
            plt.title('raw visibility')

            plt.subplot(2, 3, 5)
            plot_time_ordered_data_map(right_ascension=right_ascension,
                                       declination=declination,
                                       visibility=killed_zebra,
                                       flags=flags,
                                       grid_size=self.grid_size)
            plt.title(f'linear model correction offset {fit[0][0]:.2f} gradient {fit[0][1]:.2f}')

            plt.subplot(2, 3, 6)
            for i in range(rfi_free_visibility.shape[1]):
                plt.scatter(zebra_power,
                            rfi_free_visibility.squeeze[:, i],
                            color='black',
                            s=0.01)
            plt.plot(zebra_power, line_, color='black', label='uncorrected')

            for i in range(rfi_free_visibility.shape[1]):
                plt.scatter(zebra_power,
                            rfi_free_visibility.squeeze[:, i] / normalized_line,
                            color='red',
                            s=0.1)
            plt.plot(zebra_power, line_ / normalized_line, color='red', label='excess power removed')
            plt.xlabel(
                f'Power integrated from {zebra_frequencies[0] / 1e6:.0f} to {zebra_frequencies[-1] / 1e6:.0f} MHz')
            plt.ylabel(f'Raw signal from {rfi_free_frequencies[0] / 1e6:.0f} to {rfi_free_frequencies[1] / 1e6:.0f}'
                       f' MHz, mostly RFI free')
            plt.legend()

            plt.savefig(os.path.join(receiver_path, f'zebra_correction_matrix_plot.png'))
            plt.close()

            for i, gradient in enumerate(np.linspace(0, 1.5)):
                line_ = self.straight_line(zebra_power * 1e-10, fit[0][0], gradient)
                normalized_line = line_ / line_[np.argmin(zebra_power)]

                killed_zebra = channel_visibility * (1 / normalized_line)[:, np.newaxis, np.newaxis]
                plot_time_ordered_data_map(right_ascension=right_ascension,
                                           declination=declination,
                                           visibility=killed_zebra,
                                           flags=flags,
                                           grid_size=self.grid_size)
                plt.title(f'line gradient {gradient:.3f}')
                plot_name = f'zebra_removal_{i}.png'
                plt.savefig(os.path.join(receiver_path, plot_name))
                plt.close()

    @staticmethod
    def straight_line(parameter, offset, gradient):
        return offset + gradient * parameter

    def straight_line_fitting_wrapper(self, parameter, offset, gradient, repetitions: int):
        line_ = self.straight_line(parameter=parameter, offset=offset, gradient=gradient)
        return np.tile(line_[:, np.newaxis], (1, repetitions))
