from ivory.plugin.abstract_plugin import AbstractPlugin


class DummyPlugin(AbstractPlugin):
    """Dummy Plugin. Prints some text."""

    def __init__(self, testvar: int):
        super().__init__()
        print("Instance")
        self.testvar = testvar

    def set_requirements(self):
        self.requirements = []

    def run(self):
        print("Test!", self.testvar)
