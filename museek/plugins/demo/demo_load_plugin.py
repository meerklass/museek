from io import BytesIO

import requests
from PIL import Image

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.result import Result
from museek.enum.demo_enum import DemoEnum


class DemoLoadPlugin(AbstractPlugin):
    """ For demonstration. Loads an image from the web. """

    def run(self):
        response = requests.get(self.config.url)
        image = Image.open(BytesIO(response.content))

        self.set_result(result=Result(location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE,
                                      result=image,
                                      allow_overwrite=True))

    def set_requirements(self):
        pass
