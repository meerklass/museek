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

class KnownRfiPlugin(AbstractPlugin):
    """ Plugin to completely flag channels designated to known RFI. """

    def __init__(self,
                 gsm_900_uplink: tuple[float, float] | None,
                 gsm_900_downlink: tuple[float, float] | None,
                 gsm_1800_uplink: tuple[float, float] | None,
                 gps: tuple[float, float] | None,
                 extra_rfi: list[tuple[float, float]] | None = None):
        """
        Initialise the plugin
        :param gsm_900_uplink: optional lower and upper frequency [MHz] limits, usually `(890, 915)`
        :param gsm_900_downlink: optional lower and upper frequency [MHz] limits, usually `(925, 960)`
        :param gsm_1800_uplink: optional lower and upper frequency [MHz] limits, usually `(1710, 1785)`
        :param gps: optional lower and upper frequency [MHz] limits, usually `(1170, 1390)`
        :param extra_rfi: optional `list` of extra rfi frequency [MHz] limit tuples
        """
        super().__init__()
        rfi_list = [gsm_900_uplink,
                    gsm_900_downlink,
                    gsm_1800_uplink,
                    gps]
        if extra_rfi is not None:
            rfi_list.extend(extra_rfi)
        self.rfi_list = [rfi for rfi in rfi_list if rfi is not None]
        self.data_element_factory = FlagElementFactory()
        self.report_file_name = 'flag_report.md'

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')]

    def run(self, data: TimeOrderedData, flag_report_writer: ReportWriter, output_path: str):
        """
        Flag all channels defined by `self.rfi_list` and save the result to the context.
        :param data: time ordered data of the entire block
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        """
        mega = 1e6
        data.load_visibility_flags_weights(polars='auto')
        new_flag = np.zeros(data.shape, dtype=bool)
        for channel, frequency in enumerate(data.frequencies.squeeze):
            for rfi_tuple in self.rfi_list:
                if rfi_tuple[0] <= frequency / mega <= rfi_tuple[1]:
                    new_flag[:, channel, :] = True
                    continue
        data.flags.add_flag(flag=FlagList.from_array(array=new_flag, element_factory=self.data_element_factory))
        self.set_result(result=Result(location=ResultEnum.DATA, result=data, allow_overwrite=True))

        receivers_list, flag_percent = flag_percent_recv(data)
        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = ['...........................', 'Running KnownRfiPlugin with '+f"MuSEEK version: {branch} ({commit})", 'Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"), 'The flag fraction for each receiver: '] + [f'{x}  {y}' for x, y in zip(receivers_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        waterfall(data.visibility.get(recv=0),
                  data.flags.get(recv=0),
                  cmap='gist_ncar')
        plt.savefig(os.path.join(output_path, 'known_rfi_plugin_result_receiver_0.png'), dpi=1000)
        plt.close()
