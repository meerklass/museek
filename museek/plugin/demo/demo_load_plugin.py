import os
from io import BytesIO

import requests
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.result import Result
from PIL import Image

from museek.enums.demo_enum import DemoEnum


class DemoLoadPlugin(AbstractPlugin):
    """For demonstration. Loads an image from the web."""

    def __init__(self, url: str, context_file_name: str, context_folder: str):
        super().__init__()
        self.url = url
        self.context_file_name = context_file_name
        self.context_folder = context_folder

    def run(self, **kwargs):
        response = requests.get(self.url)
        image = Image.open(BytesIO(response.content))

        self.set_result(
            result=Result(
                location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE,
                result=image,
                allow_overwrite=True,
            )
        )
        # setting the paths and names for context to disc storage
        context_storage_directory = os.path.abspath(self.context_folder)
        self.set_result(
            result=Result(
                location=DemoEnum.CONTEXT_STORAGE_DIRECTORY,
                result=context_storage_directory,
            )
        )
        self.set_result(
            result=Result(
                location=DemoEnum.CONTEXT_FILE_NAME, result=self.context_file_name
            )
        )

    def set_requirements(self):
        pass
