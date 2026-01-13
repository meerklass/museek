import unittest

from ivory.cli.main import _main


class TestDemoPlugins(unittest.TestCase):
    def test_demo_plugs(self):
        self.assertIsNone(
            _main(
                "--DemoPlotPlugin-do-show=False",
                "--DemoPlotPlugin-do-save=False",
                "museek.config.demo",
            )
        )
