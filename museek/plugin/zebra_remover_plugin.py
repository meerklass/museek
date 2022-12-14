import os

from astropy import units
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.enum.result_enum import ResultEnum
from museek.katcali_visualiser import plot_data
from museek.mask import point_mask_list
from museek.time_ordered_data import TimeOrderedData
from matplotlib import pyplot as plt
import healpy
import numpy as np


class ZebraRemoverPlugin(AbstractPlugin):
    def __init__(self):
        super().__init__()

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, scan_data: TimeOrderedData, output_path: str):
        channel = 3000
        timestamp_dates = scan_data.timestamp_dates.squeeze
        start_index = 1500
        end_index = len(timestamp_dates)
        times = range(start_index, end_index)
        scan_data.load_visibility_flags_weights()

        # visibility = scan_data.visibility.get(recv=0).squeeze
        # plt.imshow(visibility.T, aspect='auto')
        # plt.axhline(channel, xmin=times[0] / len(timestamp_dates), xmax=1)
        # plt.show()

        channel_visibility = scan_data.visibility.get(recv=0, time=times, freq=channel).squeeze
        right_ascension = scan_data.right_ascension.get(recv=0, time=times).squeeze
        declination = scan_data.declination.get(recv=0, time=times).squeeze

        point_sources = np.loadtxt(os.path.join(os.path.dirname(__file__), '../../data/radio_point_sources.txt'))
        data_points = SkyCoord(right_ascension * units.deg, declination * units.deg, frame='icrs')
        angle_threshold = .5
        point_source_mask_list = point_mask_list(mask_points=point_sources,
                                                 data_points=data_points,
                                                 angle_threshold=angle_threshold)

        point_source_mask = np.zeros_like(channel_visibility, dtype=bool)
        point_source_mask[point_source_mask_list] = True

        zebra_channels = range(350, 498)
        frequencies = scan_data.frequencies.squeeze
        zebra_frequencies = [frequencies[channel] for channel in zebra_channels]
        zebra_visibility = scan_data.visibility.get(freq=zebra_channels).squeeze
        zebra_power = np.trapz(zebra_visibility, x=zebra_frequencies, axis=1)

        rfi_free_channels = range(2500, 3000)
        rfi_free_visibility = scan_data.visibility.get(freq=rfi_free_channels).squeeze
        rfi_free_frequencies = [frequencies[channel] for channel in rfi_free_channels]
        rfi_free_power = np.trapz(rfi_free_visibility, x=rfi_free_frequencies, axis=1)

        fit = curve_fit(self.straight_line,
                        zebra_power[start_index:end_index]*1e-10,
                        rfi_free_power[start_index:end_index]*1e-10,
                        p0=[1., 1.])
        line_ = self.straight_line(zebra_power[start_index:end_index]*1e-10, *fit[0])*1e10

        plt.scatter(zebra_power[start_index:end_index],
                    rfi_free_power[start_index:end_index],
                    color='black',
                    s=0.1)
        plt.plot(zebra_power[start_index:end_index], line_, color='black')
        plt.xlabel(f'Power integrated from {zebra_frequencies[0] / 1e6:.0f} to {zebra_frequencies[-1] / 1e6:.0f} MHz')
        plt.ylabel(f'Power from {rfi_free_frequencies[0] / 1e6:.0f} to {rfi_free_frequencies[1] / 1e6:.0f}'
                   f' MHz, mostly RFI free')
        plt.show()

        plt.scatter(zebra_power[start_index:end_index],
                    rfi_free_power[start_index:end_index]/line_,
                    color='black',
                    s=0.1)
        plt.plot(zebra_power[start_index:end_index], line_/line_, color='black')
        plt.xlabel(f'Power integrated from {zebra_frequencies[0] / 1e6:.0f} to {zebra_frequencies[-1] / 1e6:.0f} MHz')
        plt.ylabel(f'Power from {rfi_free_frequencies[0] / 1e6:.0f} to {rfi_free_frequencies[1] / 1e6:.0f}'
                   f' MHz, mostly RFI free')
        plt.show()

        killed_zebra = channel_visibility/line_
        killed_zebra *= np.mean(channel_visibility)/np.mean(killed_zebra)

        plt.figure(figsize=(6,12))
        plt.subplot(2,1,1)
        plot_data(right_ascension, declination, killed_zebra, flags=[point_source_mask])
        plt.title('linear zebra model gain multiplier')
        plt.subplot(2,1,2)
        plot_data(right_ascension, declination, channel_visibility, flags=[point_source_mask])
        plt.title('raw visibility')

        plt.show()



        # for i, gradient in enumerate(np.linspace(fit[0][1]*0.1, fit[0][1]*3)):
        #     line_ = self.straight_line(zebra_power[start_index:end_index] * 1e-10, fit[0][0], gradient) * 1e10
        #     killed_zebra = channel_visibility / line_
        #     killed_zebra *= np.mean(channel_visibility) / np.mean(killed_zebra)
        #
        #     plot_data(right_ascension, declination, killed_zebra, flags=[point_source_mask])
        #     plt.title(f'line gradient {gradient}')
        #     plot_name = f'zebra_removal_{i}.png'
        #     plt.savefig(os.path.join(output_path, plot_name))
        #     plt.close()

    def ra_dec_to_index(self, declination, right_ascension, nside):
        return healpy.pixelfunc.ang2pix(nside,
                                        self.declination_to_phi(declination),
                                        self.right_ascension_to_theta(right_ascension))

    @staticmethod
    def declination_to_phi(declination):
        return np.radians(-declination + 90.)

    @staticmethod
    def right_ascension_to_theta(right_ascension):
        return np.radians(360. - right_ascension)

    @staticmethod
    def longitude_range(right_ascension, margin=1):
        return [360 - max(right_ascension) - margin, 360 - min(right_ascension) + margin]

    @staticmethod
    def latitude_range(declination, margin=1):
        return [min(declination) - margin, max(declination) + margin]

    @staticmethod
    def straight_line(parameter, offset, gradient):
        return offset + gradient * parameter
