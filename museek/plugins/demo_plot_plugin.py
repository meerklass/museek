from PIL import Image
from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.enum.demo_enum import DemoEnum


class DemoPlotPlugin(AbstractPlugin):
    """ For demonstration. Plots an image. """
    def set_requirements(self):
        self.requirements = [Requirement(location=DemoEnum.ASTRONAUT_RIDING_HORSE_IN_SPACE,
                                         variable='astronaut_image')]

    def run(self, astronaut_image: Image):
        plt.imshow(astronaut_image)
        plt.axis('off')
        if self.config.do_show:
            plt.show()
        else:
            plt.savefig('demo_plot_plugin.png')

