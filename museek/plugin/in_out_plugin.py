import os
from datetime import datetime

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.result import Result
from ivory.utils.struct import Struct
from museek.enum.plugin_enum import PluginEnum
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData

PLUGIN_ROOT = os.path.dirname(__file__)


class InOutPlugin(AbstractPlugin):
    """ Plugin to load data and to set output paths. """

    def __init__(self, ctx: Struct | None):
        super().__init__(ctx=ctx)
        self.output_folder = self.config.output_folder
        if self.output_folder is None:
            self.output_folder = os.path.join(PLUGIN_ROOT, '../../results/')
        self.check_output_folder_exists()

    def set_requirements(self):
        """ First plugin, no requirements. """
        pass

    def run(self):
        """ Loads the data as `TimeOrderedData` and sets it as a result. """
        receivers = [Receiver.from_string(receiver_string=receiver) for receiver in self.config.receiver_list]
        data = TimeOrderedData(token=self.config.token,
                               data_folder=self.config.data_folder,
                               block_name=self.config.block_name,
                               receivers=receivers,
                               force_load_from_correlator_data=self.config.force_load_from_correlator_data,
                               do_save_to_disc=self.config.do_save_visibility_to_disc)

        output_path = os.path.join(self.output_folder, f'{self.config.block_name}/')
        os.makedirs(output_path, exist_ok=True)

        timestamp = int(data.name.split('_')[0])
        observation_date = datetime.fromtimestamp(timestamp)

        self.set_result(result=Result(location=PluginEnum.DATA, result=data))
        self.set_result(result=Result(location=PluginEnum.RECEIVERS, result=receivers))
        self.set_result(result=Result(location=PluginEnum.OUTPUT_PATH, result=output_path))
        self.set_result(result=Result(location=PluginEnum.OBSERVATION_DATE, result=observation_date))

    def check_output_folder_exists(self):
        """ Raises a `ValueError` if `self.output_folder` does not exist. """
        if not os.path.exists(self.output_folder):
            raise ValueError(f'The output folder does not exists: {self.output_folder}')
