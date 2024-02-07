import json
import os
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from definitions import MEGA
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.model.bandpass_model import BandpassModel
from museek.time_ordered_data import TimeOrderedData


class StandingWaveFitScanPlugin(AbstractPlugin):
    """
    Experimental plugin to fit a standing wave model to the scanning data.
    The model assumes that the dish panel gaps are responsible for a sum of sinusoidal standing waves.
    """

    def __init__(self,
                 target_channels: range | list[int] | None,
                 footprint_ra_dec: tuple[tuple[float, float], tuple[float, float]] | None,
                 do_store_parameters: bool = False):
        """
        Initialise
        :param target_channels: optional `list` or `range` of channel indices to be examined, if `None`, all are used
        :param footprint_ra_dec: optional tuple of min and max right ascension (first) and declination (second)
                                 defining a rectangle. data points outside are used for standing wave calibration
                                 if `None`, the first few data points are used for calibration
        :param do_store_parameters: whether to store the standing wave fit parameters to a file
        """
        super().__init__()
        self.target_channels = target_channels
        self.plot_name = 'standing_wave_fit_scan_plugin'
        self.footprint_ra_dec = footprint_ra_dec
        self.do_store_parameters = do_store_parameters
        self.first_scan_dumps_label = 'first_scan_dumps'
        self.off_cut_label = 'off_cut'
        if self.footprint_ra_dec is None:
            self.calibrator_label = self.first_scan_dumps_label
        else:
            self.calibrator_label = self.off_cut_label

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
        epsilon_function_dict = {}  # type: dict[dict[[Callable]]]
        legendre_function_dict = {}  # type: dict[dict[[Callable]]]
        parameters_dict = {}  # type: dict[dict[dict[float]]]

        for i_receiver, receiver in enumerate(scan_data.receivers):
            # if receiver.name != 'm008v' and receiver.name != 'm008h':
                # continue
            print(f'Working on {receiver.name}...')
            i_antenna = receiver.antenna_index(receivers=scan_data.receivers)
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            times = self.calibrator_times(data=scan_data, i_antenna=i_antenna)
            self.plot_times(data=scan_data, times=times, i_antenna=i_antenna, output_path=receiver_path)
            if receiver.name not in epsilon_function_dict:
                epsilon_function_dict[receiver.name] = {}  # type: dict[Callable]
                legendre_function_dict[receiver.name] = {}  # type: dict[Callable]
                parameters_dict[receiver.name] = {}  # type: dict[dict[float]]

            frequencies = scan_data.frequencies.get(freq=self.target_channels)
            bandpass_model = BandpassModel(
                plot_name=self.plot_name,
                # standing_wave_displacements=[14.7, 13.4, 16.2, 17.9, 12.4, 19.6, 11.7, 5.8],
                # standing_wave_displacements=[0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                # standing_wave_displacements=[1],
                standing_wave_displacements=[0.5, 5, 6, 9, 16.2, 17.9],
                legendre_degree=6,
                # polyphase_parameters=(6, 64, 1.0003)
                polyphase_parameters=(6, 64, 1.0)
            )
            flags = scan_data.flags.get(time=times,
                                        freq=self.target_channels,
                                        recv=i_receiver)
            if flags.combine().squeeze.all():
                print('Everything flagged... - continue')
                continue
            bandpass_estimator = scan_data.visibility.get(time=times,
                                                          freq=self.target_channels,
                                                          recv=i_receiver).mean(axis=0, flags=flags)
            bandpass_estimator /= bandpass_estimator.max(axis=1).squeeze
            fit_args = dict(frequencies=frequencies,
                            estimator=bandpass_estimator,
                            receiver_path=receiver_path,
                            calibrator_label=self.calibrator_label)
            bandpass_model.fit(**fit_args)
            print('no double fit')
            # try:
            #     bandpass_model.double_fit(n_double=2, **fit_args)
            # except RuntimeError:
            #     print('warning: fit did not converge?')
            epsilon_function_dict[receiver.name][self.calibrator_label] = bandpass_model.epsilon_function
            legendre_function_dict[receiver.name][self.calibrator_label] = bandpass_model.legendre_function
            parameters_dict[receiver.name][self.calibrator_label] = bandpass_model.parameters_dictionary

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
        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_CALIBRATOR_LABEL,
                                      result=self.calibrator_label,
                                      allow_overwrite=False))

    def calibrator_times(self, data: TimeOrderedData, i_antenna: int) -> range | np.ndarray:
        """ Return the calibration time dump indices for antenna `i_antenna` in `data` as `range` or `np.ndarray`. """
        if self.calibrator_label == self.first_scan_dumps_label:
            return self.first_scan_dumps()
        elif self.calibrator_label == self.off_cut_label:
            return self.off_cut_dumps(data=data, i_antenna=i_antenna)
        else:
            raise NotImplementedError(f'No calibration implemented with label {self.calibrator_label}.'
                                      f'Available: {self.first_scan_dumps_label} and {self.off_cut_label}')

    @staticmethod
    def first_scan_dumps() -> range:
        """ Return the first few scan dump indices as `range`. """
        start_dump_index = 0
        # end_dump_index = 124  # 124 is the first swing back and forth
        end_dump_index = 2700  # this can't be too beg to make it work with all blocks
        return range(start_dump_index, end_dump_index)

    def off_cut_dumps(self, data: TimeOrderedData, i_antenna: int) -> range | np.ndarray:
        """
        Return the scan dump indices of antenna `i_antenna` in `data` that lie outside a defined rectangle in ra-dec.
        """
        coordinates = (data.right_ascension.get(recv=i_antenna), data.declination.get(recv=i_antenna))
        conditions = [((coordinate.squeeze < footprint[0]) | (footprint[1] < coordinate.squeeze))
                      for footprint, coordinate in zip(self.footprint_ra_dec, coordinates)]
        return np.where(conditions[0] | conditions[1])[0]

    @staticmethod
    def plot_times(data: TimeOrderedData, i_antenna: int, times: range | np.ndarray, output_path: str):
        """ Plot the time dumps used for calibration on a coordinate grid. """
        right_ascension = data.right_ascension.get(recv=i_antenna).squeeze
        declination = data.declination.get(recv=i_antenna).squeeze
        colors = ['black' if i not in times else 'red' for i in range(len(right_ascension))]
        plt.scatter(right_ascension, declination, c=colors, s=5)
        plt.xlabel('right ascension [deg]')
        plt.ylabel('declination [deg]')
        plt.savefig(os.path.join(output_path, 'part_of_footprint_for_standing_wave_calibration.png'))
