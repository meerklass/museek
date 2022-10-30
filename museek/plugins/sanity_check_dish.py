from ivy.plugin.abstract_plugin import AbstractPlugin

from museek.plugins.sanity_check_observation import Plugin as SanityCheckObservationPlugin


class Plugin(AbstractPlugin):
    """
    DOC
    """

    plugin_name = 'SanityCheckDish'

    def run(self):
        print(self)
        sanity_check_observation_results = self.output_of_plugin(SanityCheckObservationPlugin)

        print(self.config.second_parameter)
        second_result = self.config.second_parameter * 3
        self.save_to_context({'result': second_result})
        print(f'successfully accessed result of previous plugin: '
              f'{sanity_check_observation_results.first_result}')
