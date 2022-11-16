import os
import pickle

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.struct import Struct

PLUGIN_ROOT = os.path.dirname(__file__)


class DemoStorePlugin(AbstractPlugin):
    """ For demonstration. Stores the context to hard disc. """

    def __init__(self, ctx: Struct):
        super().__init__(ctx=ctx)
        self.ctx = ctx
        self.output_file_name = os.path.join(PLUGIN_ROOT, f'../../../results/demo/{self.config.file_name}')

    def run(self):
        with open(self.output_file_name, "wb") as context_file:
            pickle.dump(self.ctx, context_file)

    def set_requirements(self):
        pass
