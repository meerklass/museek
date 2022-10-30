from ivy.plugin.abstract_plugin import AbstractPlugin


class Plugin(AbstractPlugin):
    """
    DOC
    """

    plugin_name = 'SanityCheckObservation'

    def run(self):
        print(self)
        result_dict = {
            'first_result': self.config.first_parameter * 2  # describe the result briefly
        }

        print(self.config.first_parameter)
        self.save_to_context(result_dict=result_dict)
