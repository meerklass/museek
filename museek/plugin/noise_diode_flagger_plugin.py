import os

import numpy as np
from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.factory.data_element_factory import FlagElementFactory
from museek.noise_diode import NoiseDiode
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.visualiser import waterfall
from museek.util.tools import flag_percent_recv


class NoiseDiodeFlaggerPlugin(AbstractPlugin):
    """ Plugin to flag the noise diode firings. """

    def __init__(self):
        """ Initialise. """
        super().__init__()
        self.data_element_factory = FlagElementFactory()
        self.output_path = None
        self.report_file_name = 'flag_report.md'

    def set_requirements(self):
        """ Set the requirements `output_path` and the whole data. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')]

    def run(self, data: TimeOrderedData, flag_report_writer: ReportWriter, output_path: str):
        """
        Run the plugin, i.e. flag the noise diode firings
        :param data: containing the entire data
        :param flag_report_writer: report_writer of the flag
        :param output_path: path to store results
        """
        data.load_visibility_flags_weights()
        noise_diode = NoiseDiode(dump_period=data.dump_period, observation_log=data.obs_script_log)
        noise_diode_off_dumps = noise_diode.get_noise_diode_off_scan_dumps(timestamps=data.original_timestamps)
        new_mask = np.ones(data.shape, dtype=bool)
        for i in noise_diode_off_dumps:
            new_mask[i] = False
        data.flags.add_flag(flag=self.data_element_factory.create(array=new_mask))
        self.set_result(result=Result(location=ResultEnum.DATA, result=data, allow_overwrite=True))

        receivers_list, flag_percent = flag_percent_recv(data)
        lines = ['...........................', 'Running NoiseDiodeFlaggerPlugin...', 'The flag fraction for each receiver: '] + [f'{x}  {y}' for x, y in zip(receivers_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        waterfall(data.visibility.get(recv=0),
                  data.flags.get(recv=0),
                  cmap='gist_ncar')
        plt.savefig(os.path.join(output_path, 'noise_diode_flagger_result_receiver_0.png'), dpi=1000)
        plt.close()
