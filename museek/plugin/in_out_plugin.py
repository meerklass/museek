import os
from copy import deepcopy
from datetime import datetime

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.result import Result
from ivory.utils.struct import Struct
from museek.enum.result_enum import ResultEnum
from museek.enum.scan_state_enum import ScanStateEnum
from museek.noise_diode_data import NoiseDiodeData
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
        self._do_use_noise_diode = self.config.do_use_noise_diode

    def set_requirements(self):
        """ First plugin, no requirements. """
        pass

    def run(self):
        """ Loads the data as `TimeOrderedData` and sets it as a result. """
        receivers = None
        if self.config.receiver_list is not None:
            receivers = [Receiver.from_string(receiver_string=receiver) for receiver in self.config.receiver_list]
        if self._do_use_noise_diode:
            data_class = NoiseDiodeData
        else:
            data_class = TimeOrderedData
        all_data = data_class(
            token=self.config.token,
            data_folder=self.config.data_folder,
            block_name=self.config.block_name,
            receivers=receivers,
            force_load_from_correlator_data=self.config.force_load_from_correlator_data,
            do_create_cache=self.config.do_save_visibility_to_disc,
        )
        scan_data = deepcopy(all_data)
        scan_data.set_data_elements(scan_state=ScanStateEnum.SCAN)

        output_path = os.path.join(self.output_folder, f'{self.config.block_name}/')
        os.makedirs(output_path, exist_ok=True)

        # observation data from file name
        observation_date = datetime.fromtimestamp(int(all_data.name.split('_')[0]))

        self.set_result(result=Result(location=ResultEnum.DATA, result=all_data))
        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data))
        self.set_result(result=Result(location=ResultEnum.RECEIVERS, result=receivers))
        self.set_result(result=Result(location=ResultEnum.OUTPUT_PATH, result=output_path))
        self.set_result(result=Result(location=ResultEnum.OBSERVATION_DATE, result=observation_date))

    def check_output_folder_exists(self):
        """ Raises a `ValueError` if `self.output_folder` does not exist. """
        if not os.path.exists(self.output_folder):
            raise ValueError(f'The output folder does not exists: {self.output_folder}')
