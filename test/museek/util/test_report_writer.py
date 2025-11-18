import os
import unittest
from unittest.mock import patch

from museek.util.report_writer import ReportWriter


class TestReportWriter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_path = "./test/museek/util/cache/"
        cls.report_name = "report.md"
        cls.plugin_name = "PluginName"
        if not os.path.exists(cls.output_path):
            os.makedirs(cls.output_path, exist_ok=True)

    def setUp(self):
        self.data_name = "data_name"
        self.report_writer = ReportWriter(
            output_path=self.output_path,
            report_name=self.report_name,
            data_name=self.data_name,
            plugin_name=self.plugin_name,
        )

    def test_ini(self):
        self.assertTrue(os.path.exists(self.report_writer.file_name))

    @patch.object(ReportWriter, "write_to_report")
    def test_write_plot_description_to_report(self, mock_write_to_report):
        self.report_writer.write_plot_description_to_report(
            description="mock_description", plot_name="mock_plot_name"
        )
        mock_write_to_report.assert_called_once_with(
            lines=["### mock_description", "![info](mock_plot_name)"]
        )

    @patch("museek.util.report_writer.open")
    def test_write_to_report(self, mock_open):
        self.report_writer.write_to_report(lines=["1", "2"])
        mock_open().__enter__().writelines.assert_called_once_with(["1\n", "2\n"])

    @patch("museek.util.report_writer.open")
    def test_write_header(self, mock_open):
        self.report_writer._write_header(
            data_name=self.data_name, plugin_name=self.plugin_name
        )
        mock_open().__enter__().write.assert_called_once()

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join(cls.output_path, cls.report_name))
        os.removedirs(cls.output_path)
