import os

from definitions import ROOT_DIR
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.util.report_writer import ReportWriter


class OutPlugin(AbstractPlugin):
    """ Plugin to set output paths. """

    def __init__(self, output_folder: str | None):
        """
        Initialise the plugin.
        :param output_folder: folder to store ouputs
        """
        super().__init__()
        self.output_folder = output_folder
        self.report_file_name = 'flag_report.md'
        if self.output_folder is None:
            self.output_folder = os.path.join(ROOT_DIR, 'results/')
        self.check_output_folder_exists()

    def set_requirements(self):
        """ Set the block name as a requirement. """
        self.requirements = [Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name')]

    def run(self, block_name: str):
        """ Store the `output_path` as a result. """
        output_path = os.path.join(self.output_folder, f'{block_name}/')
        os.makedirs(output_path, exist_ok=True)
        self.set_result(result=Result(location=ResultEnum.OUTPUT_PATH, result=output_path))

        flag_report_writer = ReportWriter(output_path=output_path,
                                     report_name=self.report_file_name,
                                     data_name=block_name,
                                     plugin_name=self.name)

        self.set_result(result=Result(location=ResultEnum.FLAG_REPORT_WRITER, result=flag_report_writer, allow_overwrite=True))

    def check_output_folder_exists(self):
        """ Raises a `ValueError` if `self.output_folder` does not exist. """
        if not os.path.exists(self.output_folder):
            raise ValueError(f'The output folder does not exists: {self.output_folder}')
