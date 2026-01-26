import numpy as np
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from museek.enums.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.visualiser import plot_time_ordered_data_map


class ZebraRemoverPlugin(AbstractPlugin):
    def __init__(self, reference_channel: int, zebra_channels: range | list[int]):
        """
        Initialise
        :param reference_channel: the index of the reference channel,
            should be mostly rfi free before flagging
        :param zebra_channels: `list` or `range` of channel indices
            affected by the emission from the vanwyksvlei tower
        """
        super().__init__()
        self.reference_channel = reference_channel
        self.zebra_channels = zebra_channels

    def set_requirements(self):
        """Set the requirements."""
        self.requirements = [
            Requirement(location=ResultEnum.SCAN_DATA, variable="scan_data"),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable="output_path"),
        ]

    def run(self, scan_data: TimeOrderedData, output_path: str):
        scan_data.load_visibility_flags_weights(polars=["HH", "VV"])
        timestamp_dates = scan_data.timestamp_dates.squeeze

        # set rfi free channels
        rfi_free_channels = [3000, 3001]

        # manually remove the satellites:
        start_index = 1500
        end_index = len(timestamp_dates)
        times = range(start_index, end_index)

        channel_visibility = scan_data.visibility.get(
            recv=0, time=times, freq=self.reference_channel
        )
        right_ascension = scan_data.right_ascension.get(recv=0, time=times)
        declination = scan_data.declination.get(recv=0, time=times)
        flags = scan_data.flags.get(recv=0, time=times, freq=self.reference_channel)

        frequencies = scan_data.frequencies.squeeze
        zebra_frequencies = [frequencies[channel] for channel in self.zebra_channels]
        zebra_visibility = scan_data.visibility.get(
            freq=self.zebra_channels, time=times
        )
        zebra_power = np.trapezoid(
            zebra_visibility.squeeze, x=zebra_frequencies, axis=1
        )
        zebra_power_max = np.max(zebra_power)

        rfi_free_visibility = scan_data.visibility.get(
            freq=rfi_free_channels, time=times
        )
        rfi_free_frequencies = [frequencies[channel] for channel in rfi_free_channels]

        plt.imshow(scan_data.visibility.get(recv=0).squeeze.T, aspect="auto")
        plt.axhline(
            self.reference_channel, xmin=times[0] / len(timestamp_dates), xmax=1
        )
        plt.axhline(rfi_free_channels[0], xmin=times[0] / len(timestamp_dates), xmax=1)
        plt.axhline(rfi_free_channels[-1], xmin=times[0] / len(timestamp_dates), xmax=1)
        plt.show()

        # fit a straight line to the scatter plot
        def fitting_function(parameter, offset, gradient_):
            return self.straight_line_fitting_wrapper(
                parameter=parameter,
                offset=offset,
                gradient=gradient_,
                repetitions=rfi_free_visibility.shape[1],
            ).flatten()

        fit = curve_fit(
            f=fitting_function,
            xdata=zebra_power / zebra_power_max,
            ydata=rfi_free_visibility.squeeze.flatten(),
            p0=[1.0, 1.0],
        )
        line_ = self.straight_line(zebra_power / zebra_power_max, *fit[0])
        normalized_line = (
            line_ / line_[np.argmin(zebra_power)]
        )  # divide by the lowest rfi power value
        if any(normalized_line < 1):
            print("WARNING, zebra cleaning seems to add new power to the signal.")

        for i in range(rfi_free_visibility.shape[1]):
            plt.scatter(
                zebra_power, rfi_free_visibility.squeeze[:, i], color="black", s=0.01
            )
        plt.plot(zebra_power, line_, color="black", label="uncorrected")

        for i in range(rfi_free_visibility.shape[1]):
            plt.scatter(
                zebra_power,
                rfi_free_visibility.squeeze[:, i] / normalized_line,
                color="red",
                s=0.1,
            )
        plt.plot(
            zebra_power,
            line_ / normalized_line,
            color="red",
            label="excess power removed",
        )
        plt.xlabel(
            f"Power integrated from {zebra_frequencies[0] / 1e6:.0f} to "
            f"{zebra_frequencies[-1] / 1e6:.0f} MHz"
        )
        plt.ylabel(
            f"Raw signal from {rfi_free_frequencies[0] / 1e6:.0f} to "
            f"{rfi_free_frequencies[1] / 1e6:.0f} MHz, mostly RFI free"
        )
        plt.legend()
        plt.show()

        killed_zebra = channel_visibility * (
            1 / normalized_line[:, np.newaxis, np.newaxis]
        )

        plt.figure(figsize=(6, 12))
        plt.subplot(2, 1, 1)
        plot_time_ordered_data_map(
            right_ascension=right_ascension,
            declination=declination,
            visibility=killed_zebra,
            flags=flags,
        )
        plt.title("linear zebra model correction")
        plt.subplot(2, 1, 2)
        plot_time_ordered_data_map(
            right_ascension=right_ascension,
            declination=declination,
            visibility=channel_visibility,
            flags=flags,
        )
        plt.title("raw visibility")
        plt.show()

        # for i, gradient in enumerate(np.linspace(fit[0][1] * 0.1, fit[0][1] * 3)):
        #     line_ = self.straight_line(zebra_power * 1e-10, fit[0][0], gradient)
        #     normalized_line = line_ / line_[np.argmin(zebra_power)]
        #
        #     killed_zebra = channel_visibility * (1 / normalized_line)[:, np.newaxis, np.newaxis]
        #     plot_time_ordered_data_map(right_ascension=right_ascension,
        #                                declination=declination,
        #                                visibility=killed_zebra,
        #                                flags=flags)
        #     plt.title(f'line gradient {gradient:.3f}')
        #     plot_name = f'zebra_removal_{i}.png'
        #     import os
        #     plt.savefig(os.path.join(output_path, plot_name))
        #     plt.close()

    @staticmethod
    def straight_line(parameter, offset, gradient):
        return offset + gradient * parameter

    def straight_line_fitting_wrapper(
        self, parameter, offset, gradient, repetitions: int
    ):
        line_ = self.straight_line(
            parameter=parameter, offset=offset, gradient=gradient
        )
        return np.tile(line_[:, np.newaxis], (1, repetitions))
