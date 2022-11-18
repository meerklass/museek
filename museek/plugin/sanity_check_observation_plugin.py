import itertools

import matplotlib.pylab as plt
from astropy import coordinates, units

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.struct import Struct
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter


class SanityCheckObservationPlugin(AbstractPlugin):
    """
    A plugin for sanity checks specific for MeerKLASS observations.
    The scope is the production of plots and tests of the scanning route.
    """

    def __init__(self, ctx: Struct | None):
        super().__init__(ctx=ctx)

        self.output_path = None
        self.plot_name_template = 'plot_sanity_check_observation_{plot_count}.png'
        self.report_file_name = 'sanity_check_observation_report.md'
        self.reference_receiver_index = self.config.reference_receiver_index
        self.plot_count = itertools.count()
        self.report_writer = None

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, data: TimeOrderedData, output_path: str):
        """
        Runs the observation sanity check.
        Produces a selection of plots and runs a couple of checks.
        :param data: the `TimeOrderedData` object to check
        :param output_path: the path to store the results and plots
        """
        self.output_path = output_path
        report_writer = ReportWriter(output_path=output_path,
                                     report_name=self.report_file_name,
                                     data_name=data.name,
                                     plugin_name=self.name)
        self.report_writer = report_writer

        # declarations
        frequencies = data.frequencies
        timestamp_dates = data.timestamp_dates
        reference_receiver = data.receivers[self.reference_receiver_index]
        reference_antenna = data.antenna(receiver=reference_receiver)

        # coordinates
        sky_coordinates = coordinates.SkyCoord(data.azimuth.get_array() * units.deg,
                                               data.elevation.get_array() * units.deg,
                                               frame='altaz')

        # mean over dishes
        dish_mean_azimuth = data.azimuth.mean(axis=-1)
        dish_mean_elevation = data.elevation.mean(axis=-1)
        dish_mean_ra = data.right_ascension.mean(axis=-1)
        dish_mean_dec = data.declination.mean(axis=-1)

        # reference coordinates
        reference_elevation = data.elevation.get(recv=self.reference_receiver_index)
        reference_azimuth = data.azimuth.get(recv=self.reference_receiver_index)
        reference_right_ascension = data.right_ascension.get(recv=self.reference_receiver_index)
        reference_declination = data.declination.get(recv=self.reference_receiver_index)

        # conversions
        mega = 1e6

        # start
        report_writer.print_to_report(data)
        report_writer.print_to_report(data.obs_script_log)
        report_writer.print_to_report([f'Number of available antennas: {len(data.all_antennas)}',
                                       f'dump period: {data.dump_period}',
                                       f'Frequencies from {frequencies.get(freq=0).full / mega:.1f} ',
                                       f'\t \t to {frequencies.get(freq=-1).full / mega:.1f} MHz',
                                       f'Observation start time: {timestamp_dates[0]}\n ',
                                       f'\t \t and duration: {timestamp_dates[-1] - timestamp_dates[0]}'])

        # create plots
        plt.figure(figsize=(8, 4))
        plt.plot(reference_right_ascension.full,
                 reference_declination.full,
                 '.-')
        plt.xlabel('ra')
        plt.ylabel('dec')
        self.savefig(description=f'Scanning route of entire observation. '
                                 f'Reference antenna {reference_antenna.name}.')

        plt.figure(figsize=(8, 4))
        plt.plot(sky_coordinates.az, sky_coordinates.alt, '.-')
        plt.xlabel('az')
        plt.ylabel('el')
        self.savefig(description=f'Scanning route of entire observation. '
                                 f'Reference antenna {reference_antenna.name}.')

        plt.figure(figsize=(8, 4))
        plt.subplots_adjust(hspace=.2)
        plt.subplot(211)
        plt.plot(timestamp_dates.scan, reference_azimuth.scan, '.')
        plt.ylabel('az [deg]')
        plt.subplot(212)
        plt.plot(timestamp_dates.scan,
                 reference_elevation.scan,
                 '.')
        plt.xlabel('time')
        plt.ylabel('el [deg]')
        self.savefig('Azimuth and elevation vs time, during scanning. ')

        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(hspace=.5)
        plt.subplot(411)
        for i_antenna in range(len(data.antennas)):
            plt.plot(timestamp_dates.scan,
                     data.azimuth.get(recv=i_antenna).scan - dish_mean_azimuth.scan)
            plt.ylabel('az [deg]')
            plt.xlabel('time')

        plt.subplot(412)
        for i_antenna in range(len(data.antennas)):
            plt.plot(timestamp_dates.scan,
                     data.elevation.get(recv=i_antenna).scan - dish_mean_elevation.scan)
        plt.ylabel('el - mean')
        plt.xlabel('time')

        plt.subplot(413)
        for i_antenna in range(len(data.antennas)):
            plt.plot(timestamp_dates.scan, data.right_ascension.get(recv=i_antenna).scan - dish_mean_ra.scan)
        plt.xlabel('time')
        plt.ylabel('ra - mean')

        plt.subplot(414)
        for i_antenna in range(len(data.antennas)):
            plt.plot(timestamp_dates.scan, data.declination.get(recv=i_antenna).scan - dish_mean_dec.scan)
        plt.xlabel('time')
        plt.ylabel('dec - mean')

        self.savefig('All coordinates minus their mean with time. All dishes.')

        plt.hist(data.elevation.scan.flatten(), bins=200)
        plt.xlabel('elevation')
        self.savefig(description='Elevation histogram of all dishes during scan.')

        plt.figure(figsize=(8, 5))
        plt.subplot(311)
        plt.plot(timestamp_dates.full, data.temperature.full)
        plt.ylabel('temperature')
        plt.subplot(312)
        plt.plot(timestamp_dates.full, data.humidity.full)
        plt.ylabel('humidity')
        plt.subplot(313)
        plt.plot(timestamp_dates.full, data.pressure.full)
        plt.xlabel('time')
        plt.ylabel('pressure')
        self.savefig('Weather')

    def savefig(self, description: str = 'description'):
        """ Save a figure and embed it in the report with `description`. """
        count = next(self.plot_count)
        plot_name = self.plot_name_template.format(plot_count=count)

        plt.savefig(self.output_path + plot_name)
        plt.close()
        self.report_writer.write_plot_description_to_report(description=description, plot_name=plot_name)
