import os
from copy import deepcopy

from definitions import ROOT_DIR
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.enums.scan_state_enum import ScanStateEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import flag_percent_recv, git_version_info
import datetime


class ScanTrackSplitPlugin(AbstractPlugin):
    """
    Plugin to split the scanning and tracking part from the data.
    For the scanning part and calibrator tracking parts new `TimeOrderedData` objects are created.
    """

    def __init__(self, do_delete_unsplit_data: bool, do_store_context: bool):
        """
        Initialise with `do_delete_unsplit_data`, a switch that determines wether the object containing the entire
        data should be deleted to save memory.
        :param do_delete_unsplit_data: switch that determines wether the data should be deleted after split
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        """
        super().__init__()
        self.do_delete_unsplit_data = do_delete_unsplit_data
        self.do_store_context = do_store_context

    def set_requirements(self):
        """ Only requirement is the data. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')]

    def run(self, data: TimeOrderedData, block_name: str, output_path: str, flag_report_writer: ReportWriter):
        """
        Split `data` into scanning and tracking part and save.
        If `self.do_delete_unsplit_data` is `True`, `data` is deleted.
        :param data: the complete time ordered data
        :param block_name: name of the observation block
        :param flag_report_writer: report of the flagged fraction
        """
        scan_data = deepcopy(data)
        scan_data.set_data_elements(scan_state=ScanStateEnum.SCAN)

        scan_observation_start, scan_observation_end = self._observation_start_end(data=scan_data)

        track_data = deepcopy(data)
        track_data.set_data_elements(scan_state=ScanStateEnum.TRACK)

        if self.do_delete_unsplit_data:
            data = None
            self.set_result(result=Result(location=ResultEnum.DATA, result=data, allow_overwrite=True))

        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.TRACK_DATA, result=track_data, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.SCAN_OBSERVATION_START,
                                      result=scan_observation_start,
                                      allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.SCAN_OBSERVATION_END,
                                      result=scan_observation_end,
                                      allow_overwrite=False))

        
        #receivers_list, flag_percent = flag_percent_recv(scan_data)
        #branch, commit = git_version_info()
        #current_datetime = datetime.datetime.now()
        #lines = ['...........................', 'Running ScanTrackSplitPlugin with '+f"MuSEEK version: {branch} ({commit})", 'Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"), 'The flag fraction for each receiver: '] + [f'{x}  {y}' for x, y in zip(receivers_list, flag_percent)]
        #flag_report_writer.write_to_report(lines)

        if self.do_store_context:
            context_file_name = 'scan_track_split_plugin.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=output_path)

    @staticmethod
    def _observation_start_end(data: TimeOrderedData) -> tuple[float, float]:
        """ Return first and last timestamp in `data` as a `tuple`. """
        return data.timestamps.get(time=0).squeeze, data.timestamps.get(time=-1).squeeze
