import itertools

import matplotlib.pylab as plt
from ivory.enum.context_storage_enum import ContextStorageEnum
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result

from museek.antenna_sanity.constant_elevation_scans import ConstantElevationScans
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.antenna_sanity.from_log import FromLog


class SanityCheckObservationPlugin(AbstractPlugin):
    """
    A plugin for sanity checks specific for MeerKLASS observations.
    The scope is the production of plots and tests of the scanning route.
    """

    def __init__(self,
                 reference_receiver_index: int,
                 elevation_sum_square_difference_threshold: float,
                 elevation_square_difference_threshold: float,
                 elevation_antenna_standard_deviation_threshold=1e-2):
        """
        Initialise the plugin.
        :param reference_receiver_index: index of the receiver to use primarily for the sanity check
        :param elevation_sum_square_difference_threshold: threshold on the total sum of squared differences between
                                                          antenna pointing elevations and the mean
        :param elevation_square_difference_threshold: threshold on the squared difference between one antenna's
                                                      pointing elevation and the overall mean
        :param elevation_antenna_standard_deviation_threshold: threshold on the standard deviation
                                                               on antenna pointing elevation
        """
        super().__init__()

        self.output_path = None
        self.plot_name_template = 'plot_sanity_check_observation_{plot_count}.png'
        self.report_file_name = 'sanity_check_observation_report.md'
        self.reference_receiver_index = reference_receiver_index
        self.plot_count = itertools.count()
        self.report_writer = None

        self.elevation_sum_square_difference_threshold = elevation_sum_square_difference_threshold
        self.elevation_square_difference_threshold = elevation_square_difference_threshold
        self.elevation_antenna_standard_deviation_threshold = elevation_antenna_standard_deviation_threshold

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='all_data'),
                             Requirement(location=ResultEnum.SCAN_DATA,
                                         variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, scan_data: TimeOrderedData, all_data: TimeOrderedData, output_path: str):
        """
        Runs the observation sanity check.
        Produces a selection of plots and runs a couple of checks.
        :param scan_data: the `TimeOrderedData` object referring to the scanning part
        :param all_data: the `TimeOrderedData` object referring to the complete observation
        :param output_path: the path to store the results and plots
        """
        self.output_path = output_path
        report_writer = ReportWriter(output_path=output_path,
                                     report_name=self.report_file_name,
                                     data_name=scan_data.name,
                                     plugin_name=self.name)
        self.report_writer = report_writer

        frequencies = scan_data.frequencies
        timestamp_dates = scan_data.timestamp_dates
        mega = 1e6

        straggler_list = FromLog(
            obs_script_log=scan_data.obs_script_log).straggler_list()

        # start
        report_writer.print_to_report(scan_data)
        report_writer.print_to_report(scan_data.obs_script_log)
        report_writer.print_to_report([f'Number of available antennas: {len(scan_data.all_antennas)}',
                                       f'dump period: {scan_data.dump_period}',
                                       f'Frequencies from {frequencies.get(freq=0).squeeze / mega:.1f} ',
                                       f'\t \t to {frequencies.get(freq=-1).squeeze / mega:.1f} MHz',
                                       f'Observation start time: {timestamp_dates[0]}\n ',
                                       f'\t \t and duration: {timestamp_dates[-1] - timestamp_dates[0]}',
                                       f'Number of stragglers during scan: {len(straggler_list)}',
                                       f'Straggler antenna list dusing scan: {straggler_list}'])

        self.check_elevation(data=scan_data, report_writer=report_writer)
        self.create_plots_of_complete_observation(data=all_data)
        self.create_plots_of_scan_data(data=scan_data)

        self.set_result(result=Result(
            location=ContextStorageEnum.DIRECTORY, result=output_path))
        self.set_result(result=Result(
            location=ContextStorageEnum.FILE_NAME, result='context.pickle'))

    def savefig(self, description: str = 'description'):
        """ Save a figure and embed it in the report with `description`. """
        count = next(self.plot_count)
        plot_name = self.plot_name_template.format(plot_count=count)

        plt.savefig(self.output_path + plot_name)
        plt.close()
        self.report_writer.write_plot_description_to_report(
            description=description, plot_name=plot_name)

    def check_elevation(self, data: TimeOrderedData, report_writer: ReportWriter):
        """
        Look for dishes that have non-constant elevation during scanning and write results to the report.
        :param data: the `TimeOrderedData` to check
        :param report_writer: the `ReportWriter` object to handle the report
        """
        report_writer.write_to_report(lines=[
            '## Elevation constancy check',
            f'performed with summed square difference threshold {self.elevation_sum_square_difference_threshold} deg^2 '
            f'and per-timestamp square difference threshold {self.elevation_square_difference_threshold} deg^2.'
        ])
        bad_antennas = ConstantElevationScans.get_antennas_with_non_constant_elevation(
            data=data,
            threshold=self.elevation_antenna_standard_deviation_threshold
        )
        if bad_antennas:
            report_writer.write_to_report(
                lines=['The following antennas fail the test: '])
            report_writer.print_to_report(bad_antennas)

    def create_plots_of_complete_observation(self, data: TimeOrderedData):
        """ DOC """

        straggler_list = FromLog(
            obs_script_log=data.obs_script_log).straggler_list()
        reference_receiver_index = self.reference_receiver_index
        reference_check = str(data.receivers[reference_receiver_index])[:-1]
        while (reference_check in straggler_list):
            reference_receiver_index += 1
            reference_check = str(
                data.receivers[reference_receiver_index])[:-1]
        reference_receiver = data.receivers[reference_receiver_index]
        reference_antenna = data.antenna(receiver=reference_receiver)

        plt.figure(figsize=(8, 4))
        plt.plot(data.right_ascension.get(recv=self.reference_receiver_index).squeeze,
                 data.declination.get(
                     recv=self.reference_receiver_index).squeeze,
                 '.-')
        plt.xlabel('ra')
        plt.ylabel('dec')
        self.savefig(description=f'Pointing route of entire observation. '
                                 f'Reference antenna {reference_antenna.name}.')

        plt.figure(figsize=(8, 5))
        plt.subplot(311)
        plt.plot(data.timestamp_dates.squeeze, data.temperature.squeeze)
        plt.ylabel('temperature')
        plt.subplot(312)
        plt.plot(data.timestamp_dates.squeeze, data.humidity.squeeze)
        plt.ylabel('humidity')
        plt.subplot(313)
        plt.plot(data.timestamp_dates.squeeze, data.pressure.squeeze)
        plt.xlabel('time')
        plt.ylabel('pressure')
        self.savefig('Weather')

    def create_plots_of_scan_data(self, data: TimeOrderedData):
        """ Create all observation diagnostic plots for `data` and embed them in the report. """

        timestamp_dates = data.timestamp_dates

        # check validity of reference receiver
        straggler_list = FromLog(
            obs_script_log=data.obs_script_log).straggler_list()
        reference_receiver_index = self.reference_receiver_index
        reference_check = str(data.receivers[reference_receiver_index])[:-1]
        while (reference_check in straggler_list):
            reference_receiver_index += 1
            reference_check = str(
                data.receivers[reference_receiver_index])[:-1]
        reference_receiver = data.receivers[reference_receiver_index]

        # reference coordinates
        reference_elevation = data.elevation.get(recv=reference_receiver_index)
        reference_azimuth = data.azimuth.get(recv=reference_receiver_index)

        # create no straggler list
        straggler_list_indexes = [
            int(data._antenna_name_list.index(ii)) for ii in straggler_list]
        no_straggler_indexes = [int(ii)
                                for ii in range(len(data._antenna_name_list))]
        no_straggler_list = data._antenna_name_list.copy()
        for jj in straggler_list_indexes:
            no_straggler_indexes.remove(jj)
        for jj in straggler_list:
            no_straggler_list.remove(jj)

        # mean over no straggler dishes
        dish_mean_azimuth = data.azimuth[:, :,
                                         no_straggler_indexes].mean(axis=-1)
        dish_mean_elevation = data.elevation[:,
                                             :, no_straggler_indexes].mean(axis=-1)
        dish_mean_ra = data.right_ascension[:,
                                            :, no_straggler_indexes].mean(axis=-1)
        dish_mean_dec = data.declination[:, :,
                                         no_straggler_indexes].mean(axis=-1)

        plt.plot(data.right_ascension.get(recv=reference_receiver_index).squeeze,
                 data.declination.get(recv=reference_receiver_index).squeeze,
                 '.-')
        plt.xlabel('ra')
        plt.ylabel('dec')
        self.savefig(description=f'Pointing route of entire scan. '
                                 f'Reference antenna {data.antenna(receiver=reference_receiver).name}.')

        plt.figure(figsize=(8, 4))
        plt.plot(data.azimuth.squeeze, data.elevation.squeeze, '.-')
        plt.xlabel('az')
        plt.ylabel('el')
        self.savefig(description=f'Entire scanning route. '
                                 f'All antennas.')

        plt.figure(figsize=(8, 4))
        plt.plot(data.azimuth.squeeze[:, straggler_list_indexes],
                 data.elevation.squeeze[:, straggler_list_indexes], '.-')
        plt.legend(straggler_list)
        plt.xlabel('az')
        plt.ylabel('el')
        self.savefig(description=f'Entire scanning route. '
                                 f'Straggler(s)')

        plt.figure(figsize=(8, 4))
        plt.plot(data.azimuth.squeeze[:, no_straggler_indexes],
                 data.elevation.squeeze[:, no_straggler_indexes], '.-')
        plt.xlabel('az')
        plt.ylabel('el')
        self.savefig(description=f'Entire scanning route. '
                                 f'Antennas without straggler(s)')

        plt.figure(figsize=(8, 4))
        plt.subplots_adjust(hspace=.2)
        plt.subplot(211)
        plt.plot(timestamp_dates.squeeze, reference_azimuth.squeeze, '.')
        plt.ylabel('az [deg]')
        plt.subplot(212)
        plt.plot(timestamp_dates.squeeze,
                 reference_elevation.squeeze,
                 '.')
        plt.xlabel('time')
        plt.ylabel('el [deg]')
        self.savefig('Azimuth and elevation vs time, during scanning. ')

        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(hspace=.5)
        plt.subplot(411)
        for i_antenna in no_straggler_indexes:
            plt.plot(timestamp_dates.squeeze,
                     data.azimuth.get(recv=i_antenna).squeeze - dish_mean_azimuth)
            plt.ylabel('az [deg]')
            plt.xlabel('time')

        plt.subplot(412)
        for i_antenna in no_straggler_indexes:
            plt.plot(timestamp_dates.squeeze,
                     data.elevation.get(recv=i_antenna).squeeze - dish_mean_elevation)
        plt.ylabel('el - mean')
        plt.xlabel('time')

        plt.subplot(413)
        for i_antenna in no_straggler_indexes:
            plt.plot(timestamp_dates.squeeze, data.right_ascension.get(
                recv=i_antenna).squeeze - dish_mean_ra)
        plt.xlabel('time')
        plt.ylabel('ra - mean')

        plt.subplot(414)
        for i_antenna in no_straggler_indexes:
            plt.plot(timestamp_dates.squeeze, data.declination.get(
                recv=i_antenna).squeeze - dish_mean_dec)
        plt.xlabel('time')
        plt.ylabel('dec - mean')

        self.savefig(
            'All coordinates minus their mean with time. All dishes (excluding stragglers).')

        plt.hist(
            data.elevation.squeeze[:, no_straggler_indexes].flatten(), bins=200)
        plt.xlabel('elevation')
        self.savefig(
            description='Elevation histogram of all dishes during scan (excluding stragglers).')
