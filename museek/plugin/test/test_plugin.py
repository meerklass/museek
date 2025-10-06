from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement


class TestPlugin(AbstractPlugin):
    """Test Plugin. Prints some text."""

    def __init__(self, testvar: int):
        super().__init__()
        print("Instance")
        self.testvar = testvar

    def set_requirements(self):
        self.requirements = []

    def run(self):
        print("Test!", self.testvar)
