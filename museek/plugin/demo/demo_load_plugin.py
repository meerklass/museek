import os
from io import BytesIO

import requests
from PIL import Image

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.result import Result
from museek.enum.demo_enum import DemoEnum

PLUGIN_ROOT = os.path.dirname(__file__)


class DemoLoadPlugin(AbstractPlugin):
    """ For demonstration. Loads an image from the web. """

    def __init__(self,
                 url: str,
                 context_file_name: str):
        super().__init__()
        self.url = url
        self.context_file_name = context_file_name

    def run(self):
        response = requests.get(self.url)
        image = Image.open(BytesIO(response.content))

        self.set_result(result=Result(location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE,
                                      result=image,
                                      allow_overwrite=True))
        # setting the paths and names for context to disc storage
        context_storage_directory = os.path.join(PLUGIN_ROOT, f'../../../results/demo/')
        self.set_result(result=Result(location=DemoEnum.CONTEXT_STORAGE_DIRECTORY,
                                      result=context_storage_directory))
        self.set_result(result=Result(location=DemoEnum.CONTEXT_FILE_NAME,
                                      result=self.context_file_name))

    def set_requirements(self):
        pass
