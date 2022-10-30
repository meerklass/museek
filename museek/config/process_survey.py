from enum import Enum


class Pipeline(Enum):
    """ Settings relating to the pipeline. """
    plugins = [
        'museek.plugins.sanity_check_observation_plugin',
        'museek.plugins.sanity_check_dish_plugin'
    ]


class SanityCheckObservationPlugin(Enum):
    """ Sanity check of an observation with large amount of output plots and checks for scanning errors. """
    first_parameter = 1.0  # description


class SanityCheckDishPlugin(Enum):
    """ Sanity check of individual dishes and identification of defects. Many output plots. """
    second_parameter = 2.0  # description
