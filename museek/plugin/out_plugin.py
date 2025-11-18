import os

from museek.definitions import ROOT_DIR
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.util.report_writer import ReportWriter


class OutPlugin(AbstractPlugin):
    """Plugin to set output paths."""

    def __init__(self):
        """
        Initialise the plugin.
        """
        super().__init__()
        self.report_file_name = "flag_report.md"

    def set_requirements(self):
        """Set the block name as a requirement."""
        self.requirements = [
            Requirement(location=ResultEnum.BLOCK_NAME, variable="block_name"),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable="output_path"),
        ]

    def run(self, block_name: str, output_path: str):
        """Store the `flag_report_writer` as a result."""

        flag_report_writer = ReportWriter(
            output_path=output_path,
            report_name=self.report_file_name,
            data_name=block_name,
            plugin_name=self.name,
        )

        self.set_result(
            result=Result(
                location=ResultEnum.FLAG_REPORT_WRITER,
                result=flag_report_writer,
                allow_overwrite=True,
            )
        )

    def check_output_folder_exists(self):
        """Raises a `ValueError` if `self.output_folder` does not exist."""
        if not os.path.exists(self.output_folder):
            raise ValueError(f"The output folder does not exists: {self.output_folder}")
