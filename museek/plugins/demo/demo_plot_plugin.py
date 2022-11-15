import itertools
from typing import Iterable

from PIL import Image
from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.enum.demo_enum import DemoEnum


class DemoPlotPlugin(AbstractPlugin):
    """ For demonstration. Plots an image. """

    def set_requirements(self):
        self.requirements = [Requirement(location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE,
                                         variable='astronaut_image'),
                             Requirement(location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE_FLIPPED,
                                         variable='astronaut_image_flipped')]

    def run(self, astronaut_image: Image, astronaut_image_flipped: Image):
        plot_counter = itertools.count()
        self._plot(image=astronaut_image, count=plot_counter)
        self._plot(image=astronaut_image_flipped, count=plot_counter)

    def _plot(self, image: Image, count):
        plt.imshow(image)
        plt.axis('off')
        if self.config.do_show:
            plt.show()
        if self.config.do_save:
            plt.savefig(f'demo_plot_plugin{next(count)}.png')
