import json
import os
from typing import Callable

from definitions import MEGA
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enum.result_enum import ResultEnum
from museek.model.bandpass_model import BandpassModel
from museek.time_ordered_data import TimeOrderedData


class StandingWaveFitScanPlugin(AbstractPlugin):
    """
    Experimental plugin to fit a standing wave model to the scanning data.
    The model assumes that the dish panel gaps are responsible for a sum of sinusoidal standing waves.
    """

    def __init__(self,
                 target_channels: range | list[int] | None,
                 do_store_parameters: bool = False):
        """
        Initialise
        :param target_channels: optional `list` or `range` of channel indices to be examined, if `None`, all are used
        :param do_store_parameters: whether to store the standing wave fit parameters to a file
        """
        super().__init__()
        self.target_channels = target_channels
        self.plot_name = 'standing_wave_fit_scan_plugin'
        self.do_store_parameters = do_store_parameters

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, scan_data: TimeOrderedData, output_path: str):
        """
        Run the plugin, i.e. fit the bandpass model on scanning data and store the results
        :param scan_data: the time ordered data containing the observation's scanning part
        :param output_path: path to store results
        """

        parameters_dict_name = f'parameters_dict_frequency_' \
                               f'{scan_data.frequencies.get(freq=self.target_channels[0]).squeeze / MEGA:.0f}_to_' \
                               f'{scan_data.frequencies.get(freq=self.target_channels[-1]).squeeze / MEGA:.0f}' \
                               f'_MHz.json'

        scan_data.load_visibility_flags_weights()

        start_dump_index = 0
        end_dump_index = 124  # 124 is the first swing back and forth
        calibrator_label = 'first_scan_dumps'

        epsilon_function_dict = {}  # type: dict[dict[[Callable]]]
        legendre_function_dict = {}  # type: dict[dict[[Callable]]]
        parameters_dict = {}  # type: dict[dict[dict[float]]]
        times = range(start_dump_index, end_dump_index)

        for i_receiver, receiver in enumerate(scan_data.receivers):
            if receiver.name != 'm008v':
                continue
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            if receiver.name not in epsilon_function_dict:
                epsilon_function_dict[receiver.name] = {}  # type: dict[Callable]
                legendre_function_dict[receiver.name] = {}  # type: dict[Callable]
                parameters_dict[receiver.name] = {}  # type: dict[dict[float]]

            frequencies = scan_data.frequencies.get(freq=self.target_channels)
            bandpass_model = BandpassModel(
                plot_name=self.plot_name,
                standing_wave_displacements=[14.7, 13.4, 16.2, 17.9, 12.4, 19.6, 11.7, 5.8],
                legendre_degree=1,
            )
            flags = scan_data.flags.get(time=times,
                                        freq=self.target_channels,
                                        recv=i_receiver)
            bandpass_estimator = scan_data.visibility.get(time=times,
                                                          freq=self.target_channels,
                                                          recv=i_receiver).mean(axis=0, flags=flags)
            bandpass_model.fit(frequencies,
                               estimator=bandpass_estimator,
                               receiver_path=receiver_path,
                               calibrator_label=calibrator_label)
            epsilon_function_dict[receiver.name][calibrator_label] = bandpass_model.epsilon_function
            legendre_function_dict[receiver.name][calibrator_label] = bandpass_model.legendre_function
            parameters_dict[receiver.name][calibrator_label] = bandpass_model.parameters_dictionary

        if self.do_store_parameters:
            with open(os.path.join(output_path, parameters_dict_name), 'w') as f:
                json.dump(parameters_dict, f)

        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_EPSILON_FUNCTION_DICT,
                                      result=epsilon_function_dict,
                                      allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_LEGENDRE_FUNCTION_DICT,
                                      result=legendre_function_dict,
                                      allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_CHANNELS,
                                      result=self.target_channels,
                                      allow_overwrite=False))
