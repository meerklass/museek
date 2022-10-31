from ivory.plugin.abstract_plugin import AbstractPlugin


class SanityCheckObservationPlugin(AbstractPlugin):
    """
    DOC
    """

    def run(self):
        result_dict = {
            'first_result': self.config.first_parameter * 2  # describe the result briefly
        }

        print(f'successfully accessed own config: {self.config.first_parameter}')
        self.save_to_context(result_dict=result_dict)
