import os
from typing import Generator

import numpy as np
import datetime

from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import git_version_info, consecutive_subsets


class PointSourceCalibrationPlugin(AbstractParallelJoblibPlugin):
    """
    Plugin to calculate gain calibration using point source calibrators (e.g., HydraA, PictorA).

    This plugin processes track_data (calibrator pointings) to derive gain solutions
    by comparing measured visibility to expected flux from known calibrator sources.
    """

    def __init__(self,
                 calibrator_models: dict,
                 flag_combination_threshold: int,
                 do_store_context: bool,
                 **kwargs):
        """
        Initialize the plugin.

        :param calibrator_models: Dictionary mapping calibrator names to their flux models
                                  e.g., {'HydraA': model_func, 'PictorA': model_func}
        :param flag_combination_threshold: Threshold for combining sets of flags, usually 1
        :param do_store_context: If True, context is stored to disc after finishing
        """
        super().__init__(**kwargs)
        self.calibrator_models = calibrator_models
        self.flag_combination_threshold = flag_combination_threshold
        self.do_store_context = do_store_context

    def set_requirements(self):
        """Set the requirements for the plugin."""
        self.requirements = [
            Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
            Requirement(location=ResultEnum.CALIBRATOR_VALIDATED_PERIODS, variable='calibrator_validated_periods'),
            Requirement(location=ResultEnum.CALIBRATOR_DUMP_INDICES, variable='calibrator_dump_indices'),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
            Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
            Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')
        ]

    def map(self,
            track_data: TimeOrderedData,
            calibrator_validated_periods: list,
            calibrator_dump_indices: dict,
            flag_report_writer: ReportWriter,
            output_path: str,
            block_name: str) -> Generator[tuple, None, None]:
        """
        Yield data for each receiver to be processed in parallel.

        :param track_data: Time ordered data containing calibrator tracking observations
        :param calibrator_validated_periods: List of validated calibrator periods (e.g., ['before_scan', 'after_scan'])
        :param calibrator_dump_indices: Dict mapping periods to dump indices
        :param flag_report_writer: Report writer for flagging info
        :param output_path: Path to store results
        :param block_name: Name of the observation block
        :yield: Tuple of (receiver_path, calibrator_periods, dump_indices, visibility, flags, frequencies, dumps)
        """
        track_data.load_visibility_flags_weights(polars='auto')
        initial_flags = track_data.flags.combine(threshold=self.flag_combination_threshold)
        freq = track_data.frequencies.squeeze  # Frequencies in Hz
        dumps = np.array(track_data._dumps())  # Absolute dump indices in track_data

        for i_receiver, receiver in enumerate(track_data.receivers):
            receiver_path = os.path.join(output_path, receiver.name)
            if not os.path.isdir(receiver_path):
                os.makedirs(receiver_path)

            visibility_recv = track_data.visibility.get(recv=i_receiver)
            initial_flag_recv = initial_flags.get(recv=i_receiver)

            yield (receiver_path, calibrator_validated_periods, calibrator_dump_indices,
                   visibility_recv, initial_flag_recv, freq, dumps)

    def run_job(self, anything: tuple) -> np.ndarray:
        """
        Calculate gain solution for one receiver.

        :param anything: Tuple from map() containing receiver data
        :return: Gain solution array for this receiver
        """
        (receiver_path, calibrator_validated_periods, calibrator_dump_indices,
         visibility_recv, initial_flag_recv, freq, dumps) = anything

        # Initialize gain solution array (time, frequency, 1 receiver)
        gain_solution = np.ones(visibility_recv.squeeze.shape, dtype=complex)

        # Process each calibrator period
        for period in calibrator_validated_periods:
            calibrator_dump_indices_consecutive_subsets = consecutive_subsets(
                calibrator_dump_indices[period]
            )

            # Process each pointing separately
            for subset in calibrator_dump_indices_consecutive_subsets:
                select = np.isin(dumps, subset)  # Boolean mask for this subset
                vis_pointing = visibility_recv.squeeze[select]
                flag_pointing = initial_flag_recv.squeeze[select]

                # TODO: Get expected calibrator model flux
                # model_flux = self._get_calibrator_model(period, freq)

                # TODO: Calculate gain = measured / model
                # gain_pointing = self._calculate_gain(vis_pointing, flag_pointing, model_flux)

                # TODO: Store gain solution
                # gain_solution[select, :] = gain_pointing

        return gain_solution

    def gather_and_set_result(self,
                              result_list: list[np.ndarray],
                              track_data: TimeOrderedData,
                              calibrator_validated_periods: list,
                              calibrator_dump_indices: dict,
                              flag_report_writer: ReportWriter,
                              output_path: str,
                              block_name: str):
        """
        Combine gain solutions from all receivers and set results.

        :param result_list: List of gain solution arrays, one per receiver
        :param track_data: Time ordered data
        :param calibrator_validated_periods: Validated periods
        :param calibrator_dump_indices: Calibrator dump indices
        :param flag_report_writer: Report writer
        :param output_path: Output path
        :param block_name: Block name
        """
        # Combine all receiver gain solutions into one array (time, freq, receivers)
        gain_solution_array = np.array(result_list, dtype='complex').transpose(1, 2, 0)

        # TODO: Set gain solution on track_data
        # track_data.set_gain_solution(gain_solution_array, mask_array)

        # Write report
        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = [
            '...........................',
            f'Running PointSourceCalibrationPlugin with MuSEEK version: {branch} ({commit})',
            f'Finished at {current_datetime.strftime("%Y-%m-%d %H:%M:%S")}',
            f'Calibrator periods processed: {calibrator_validated_periods}'
        ]
        flag_report_writer.write_to_report(lines)

        # Set results
        self.set_result(result=Result(location=ResultEnum.TRACK_DATA, result=track_data, allow_overwrite=True))

        if self.do_store_context:
            context_file_name = 'point_source_calibration_plugin.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=output_path)
