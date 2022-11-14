from PIL import Image

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enum.demo_enum import DemoEnum


class DemoFlipPlugin(AbstractPlugin):
    """ For demonstration. Flips right and left in an image. """
    def set_requirements(self):
        self.requirements = [Requirement(location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE,
                                         variable='astronaut_image')]

    def run(self, astronaut_image: Image):
        if self.config.do_flip:
            astronaut_image = self._flip(image=astronaut_image)
        self.set_result(result=Result(location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE,
                                      result=astronaut_image))

    @staticmethod
    def _flip(image: Image) -> Image:
        return image.transpose(method=Image.FLIP_LEFT_RIGHT)
