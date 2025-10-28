import os
import sys
from io import StringIO
from typing import Any


class ReportWriter:
    """ Class to write analysis reports for perusal. """

    def __init__(self, output_path: str, report_name: str, data_name: str, plugin_name: str):
        """
        Create an empty file to contain the report.
        :param output_path: path to store the report
        :param report_name: name of stored report
        :data_name: name of analysed data
        :plugin_name: name of the plugin that creates the report
        """
        self.file_name = os.path.join(output_path, report_name)
        self._write_header(data_name=data_name, plugin_name=plugin_name)

    def print_to_report(self, anything: Any):
        """ Writes the `print(anything)` output to the report. """
        if isinstance(anything, list):
            to_print = ''
            for line_ in anything:
                to_print += f'{line_}\n'
        else:
            to_print = anything

        buffer = StringIO()
        # redirect stdout to buffer
        sys.stdout = buffer

        print(to_print)
        print_output = buffer.getvalue()

        # restore stdout to default for print()
        sys.stdout = sys.__stdout__

        self.write_to_report(lines=[f'```\n{print_output}\n```\n'])

    def write_plot_description_to_report(self, description: str, plot_name: str):
        """ Writes `description` to the report and embeds `plot_name`. """
        self.write_to_report(lines=[f'### {description}', f'![info]({plot_name})'])

    def write_to_report(self, lines: list[str]):
        """ Write `lines` to report. """
        with open(self.file_name, 'a') as report_file:
            report_file.writelines([line + '\n' for line in lines])

    def _write_header(self, plugin_name: str, data_name: str):
        """ Write the report header. """
        with open(self.file_name, 'w') as report_file:
            report_file.write(f'# {plugin_name}\n## Report of {data_name}\n')
