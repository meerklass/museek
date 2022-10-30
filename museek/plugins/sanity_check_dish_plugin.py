from ivy.plugin.abstract_plugin import AbstractPlugin

from museek.plugins.sanity_check_observation_plugin import SanityCheckObservationPlugin


class SanityCheckDishPlugin(AbstractPlugin):
    """
    DOC
    """

    def run(self):
        sanity_check_observation_results = self.output_of_plugin(SanityCheckObservationPlugin)

        print(f'successfully accessed own config: {self.config.second_parameter}.')
        second_result = self.config.second_parameter * 3
        self.save_to_context({'result': second_result})
        print(f'successfully accessed result of previous plugin: '
              f'{sanity_check_observation_results.first_result}')
