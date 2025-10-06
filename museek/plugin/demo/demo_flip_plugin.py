from PIL import Image
from PIL.Image import Transpose

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.demo_enum import DemoEnum


class DemoFlipPlugin(AbstractPlugin):
    """For demonstration. Flips right and left in an image."""

    def __init__(self, do_flip_right_left: bool, do_flip_top_bottom: bool):
        super().__init__()

        self.do_flip_right_left = do_flip_right_left
        self.do_flip_top_bottom = do_flip_top_bottom

    def set_requirements(self):
        self.requirements = [
            Requirement(
                location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE,
                variable="astronaut_image",
            )
        ]

    def run(self, **kwargs):
        astronaut_image: Image.Image = kwargs["astronaut_image"]
        if self.do_flip_right_left:
            print("Flipping right left...")
            astronaut_image = self._flip_right_left(image=astronaut_image)
        if self.do_flip_top_bottom:
            print("Flipping top bottom...")
            astronaut_image = self._flip_top_bottom(image=astronaut_image)
        self.set_result(
            result=Result(
                location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE_FLIPPED,
                result=astronaut_image,
            )
        )

    @staticmethod
    def _flip_right_left(image: Image.Image) -> Image.Image:
        return image.transpose(method=Transpose.FLIP_LEFT_RIGHT)

    @staticmethod
    def _flip_top_bottom(image: Image.Image) -> Image.Image:
        return image.transpose(method=Transpose.FLIP_TOP_BOTTOM)
