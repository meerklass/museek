import itertools
from collections.abc import Iterator

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from matplotlib import pyplot as plt
from PIL import Image

from museek.enums.demo_enum import DemoEnum


class DemoPlotPlugin(AbstractPlugin):
    """For demonstration. Plots an image."""

    def __init__(self, do_show: bool, do_save: bool):
        super().__init__()
        self.do_show = do_show
        self.do_save = do_save

    def set_requirements(self):
        self.requirements = [
            Requirement(
                location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE,
                variable="astronaut_image",
            ),
            Requirement(
                location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE_FLIPPED,
                variable="astronaut_image_flipped",
            ),
            Requirement(
                location=DemoEnum.CONTEXT_FILE_NAME, variable="context_file_name"
            ),
            Requirement(
                location=DemoEnum.CONTEXT_STORAGE_DIRECTORY,
                variable="context_storage_directory",
            ),
        ]

    def run(self, **kwargs):
        astronaut_image: Image.Image = kwargs["astronaut_image"]
        astronaut_image_flipped: Image.Image = kwargs["astronaut_image_flipped"]
        context_storage_directory: str = kwargs["context_storage_directory"]
        context_file_name: str = kwargs["context_file_name"]

        plot_counter = itertools.count()
        self._plot(
            image=astronaut_image,
            count=plot_counter,
            context_directory=context_storage_directory,
        )
        self._plot(
            image=astronaut_image_flipped,
            count=plot_counter,
            context_directory=context_storage_directory,
        )
        self.store_context_to_disc(
            context_file_name=context_file_name,
            context_directory=context_storage_directory,
        )

    def _plot(self, image: Image.Image, count: Iterator[int], context_directory: str):
        plt.imshow(image)
        plt.axis("off")
        if self.do_show:
            plt.show()
        if self.do_save:
            plt.savefig(f"{context_directory}/demo_plot_plugin{next(count)}.png")
