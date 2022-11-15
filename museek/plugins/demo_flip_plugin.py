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
        if self.config.do_flip_right_left:
            print('Flipping right left...')
            astronaut_image = self._flip_right_left(image=astronaut_image)
        if self.config.do_flip_top_bottom:
            print('Flipping top bottom...')
            astronaut_image = self._flip_top_bottom(image=astronaut_image)
        self.set_result(result=Result(location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE_FLIPPED,
                                      result=astronaut_image))

    @staticmethod
    def _flip_right_left(image: Image) -> Image:
        return image.transpose(method=Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def _flip_top_bottom(image: Image) -> Image:
        return image.transpose(method=Image.FLIP_TOP_BOTTOM)
