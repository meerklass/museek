import os

import numpy as np
from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_list import FlagList
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.visualiser import waterfall
from museek.util.tools import flag_percent_recv, git_version_info
import datetime


class RawdataFlaggerPlugin(AbstractPlugin):
    """ Plugin to flag raw data with values below a minimum """

    def __init__(self,
                 flag_lower_threshold: float,
                 do_store_context: bool):
        """
        Initialise
        :param flag_lower_threshold: lower threshold to flag the data, 
                                     it relates to raw correlator units without any normalisation
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        """
        super().__init__()
        self.data_element_factory = FlagElementFactory()
        self.flag_lower_threshold = flag_lower_threshold
        self.do_store_context = do_store_context

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.FLAG_NAME_LIST, variable='flag_name_list')]

    def run(self, data: TimeOrderedData, flag_report_writer: ReportWriter, output_path: str, flag_name_list:list):
        """
        Flag raw data with values below a minimum
        :param data: time ordered data of the entire block
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        :param flag_name_list: list of the name of existing flags
        """
        data.load_visibility_flags_weights(polars='auto')
        new_flag = np.zeros(data.visibility.shape, dtype=bool)
        new_flag[data.visibility.array < self.flag_lower_threshold] = True

        data.flags.add_flag(flag=FlagList.from_array(array=new_flag, element_factory=self.data_element_factory))
        flag_name_list.append('rawdata_low_value')
        self.set_result(result=Result(location=ResultEnum.DATA, result=data, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.FLAG_NAME_LIST, result=flag_name_list, allow_overwrite=True))

        if self.do_store_context:
            context_file_name = 'rawdata_flagger_plugin.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=output_path)

        receivers_list, flag_percent = flag_percent_recv(data)
        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = ['...........................', 'Running RawdataFlaggerPlugin with '+f"MuSEEK version: {branch} ({commit})", 'Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"), 'The flag fraction for each receiver: '] + [f'{x}  {y}' for x, y in zip(receivers_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

