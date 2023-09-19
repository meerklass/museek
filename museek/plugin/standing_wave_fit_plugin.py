import json
import os
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from definitions import MEGA
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enum.result_enum import ResultEnum
from museek.model.bandpass_model import BandpassModel
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData
from museek.util.track_pointing_iterator import TrackPointingIterator
from museek.data_element import DataElement


class StandingWaveFitPlugin(AbstractPlugin):
    """
    Experimental plugin to fit a standing wave model to the data.
    The model assumes that the dish panel gaps are responsible for a sum of sinusoidal standing waves.
    """

    def __init__(self,
                 target_channels: range | list[int] | None,
                 pointing_labels: list[str],
                 do_store_parameters: bool = False):
        """
        Initialise
        :param target_channels: optional `list` or `range` of channel indices to be examined, if `None`, all are used
        :param pointing_labels: strings to label the different calibrator pointings, usually something like "on_centre"
                                and "off_centre_up" etc.
        :param do_store_parameters: whether to store the standing wave fit parameters to a file
        """
        super().__init__()
        self.target_channels = target_channels
        self.pointing_labels = pointing_labels
        self.plot_name = 'standing_wave_fit_plugin'
        self.do_store_parameters = do_store_parameters

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, track_data: TimeOrderedData, output_path: str):
        """
        Run the plugin, i.e. fit the bandpass model and store the results
        :param track_data: the time ordered data containing the observation's tracking part
        :param output_path: path to store results
        """

        parameters_dict_name = f'parameters_dict_frequency_' \
                               f'{track_data.frequencies.get(freq=self.target_channels[0]).squeeze / MEGA:.0f}_to_' \
                               f'{track_data.frequencies.get(freq=self.target_channels[-1]).squeeze / MEGA:.0f}' \
                               f'_MHz.json'

        track_data.load_visibility_flags_weights()

        parameters_dict = {}  # type: dict[dict[dict[float]]]
        epsilon_function_dict = {}  # type: dict[dict[[Callable]]]

        for i_receiver, receiver in enumerate(track_data.receivers):
            print(f'Working on {receiver}...')
            track_pointing_iterator = TrackPointingIterator(track_data=track_data,
                                                            receiver=receiver,
                                                            receiver_index=i_receiver)
            receiver_path = self.add_to_dicts_and_receiver_path(receiver=receiver,
                                                                parameters_dict=parameters_dict,
                                                                epsilon_function_dict=epsilon_function_dict,
                                                                output_path=output_path)

            for before_or_after, times, times_list, pointing_centres in track_pointing_iterator.iterate():
                bandpasses_std_dict, bandpasses_dict, track_times_dict = self.get_bandpasses_std_dicts(
                    track_data=track_data,
                    times_list=times_list,
                    times=times,
                    pointing_labels=self.pointing_labels,
                    i_receiver=i_receiver
                )
                # labels = ['on centre 1',
                #           'on centre 2',
                #           'on centre 3',
                #           'on centre 4',
                #           'on centre 5']
                labels = self.pointing_labels
                bandpass_estimator = np.sum([bandpasses_dict[key_] for key_ in labels]) / len(labels)
                frequencies = track_data.frequencies.get(freq=self.target_channels)
                bandpass_model = BandpassModel(
                    plot_name=self.plot_name,
                    standing_wave_displacements=[14.7, 13.4, 16.2, 17.9, 12.4, 19.6, 11.7, 5.8],
                    legendre_degree=1,
                )
                bandpass_model.fit(frequencies,
                                   estimator=bandpass_estimator,
                                   receiver_path=receiver_path,
                                   calibrator_label=before_or_after)
                chi_square_dd = np.append(times_list[0], times_list[2])
                chi_square_dd = np.append(chi_square_dd, times_list[4])
                chi_square_dumps = list(np.asarray(times)[chi_square_dd])  # 0 2 and 4 are on centre
                chi_square = self.chi_square(data=track_data, i_receiver=i_receiver, time_dumps=chi_square_dumps, target_channels=self.target_channels, bandpass_model=bandpass_model)
                print(f'chi square = {chi_square}')
                self.plot_corrected_track_bandpasses(bandpasses_dict=bandpasses_dict,
                                                     epsilon=bandpass_model.epsilon,
                                                     frequencies=frequencies,
                                                     before_or_after=before_or_after,
                                                     receiver_path=receiver_path)
                parameters_dict[receiver.name][before_or_after] = bandpass_model.parameters_dictionary
                epsilon_function_dict[receiver.name][before_or_after] = bandpass_model.epsilon_function

        if self.do_store_parameters:
            with open(os.path.join(output_path, parameters_dict_name), 'w') as f:
                json.dump(parameters_dict, f)

        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_EPSILON_FUNCTION_DICT,
                                      result=epsilon_function_dict,
                                      allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_CHANNELS,
                                      result=self.target_channels,
                                      allow_overwrite=False))

    @staticmethod
    def chi_square(data: TimeOrderedData, i_receiver: int, time_dumps, target_channels, bandpass_model: BandpassModel) -> float:
        n_time_dumps = len(time_dumps)
        degrees_of_freedom = n_time_dumps - 18

        # visibility = data.visibility.get(freq=target_channels, recv=i_receiver, time=time_dumps)
        # flags = data.flags.get(freq=target_channels, recv=i_receiver, time=time_dumps)
        # flags.remove_flag(index=-1)
        frequencies = data.frequencies.get(freq=target_channels)
        model_bandpass = bandpass_model.model_bandpass(frequencies)

        chi_square = []
        visibility = data.visibility.get(freq=target_channels, recv=i_receiver, time=time_dumps)
        flags = data.flags.get(freq=target_channels, recv=i_receiver, time=time_dumps)
        flags.remove_flag(index=-1)    
        standard_deviation = visibility.standard_deviation(axis=0, flags=flags)  # shape (1, n_freq, 1)
        for time_dump in time_dumps:
            visibility = data.visibility.get(freq=target_channels, recv=i_receiver, time=time_dump)
            flags = data.flags.get(freq=target_channels, recv=i_receiver, time=time_dump)
            flags.remove_flag(index=-1)    
            if all(flags.combine(1).squeeze):
                continue
            model = model_bandpass / np.mean(model_bandpass) * np.mean(visibility.squeeze)
            chi_square.append( np.square(visibility.squeeze - model) / np.square(standard_deviation.squeeze))
        empt = chi_square
        # for array in chi_square:
            # if isinstance(array, np.ndarray) and array.dtype == np.float64 and isinstance(np.mean(array),np.float64) :
                # empt.append(np.mean(array))
        empt = np.sum(empt, axis=0) / degrees_of_freedom
        empt = np.mean(empt)
        chi_square_reduced = empt
        # chi_square_reduced_mean = chi_square_reduced.mean(axis=1).squeeze
        return chi_square_reduced



    def get_bandpasses_std_dicts(self,
                                 track_data: TimeOrderedData,
                                 times_list: list[np.ndarray],
                                 times: range,
                                 pointing_labels: list[str],
                                 i_receiver: int) -> tuple:
        bandpasses_std_dict = dict()
        bandpasses_dict = dict()
        track_times_dict = dict()
        for i_label, pointing_times in enumerate(times_list):
            label = pointing_labels[i_label]
            track_times = list(np.asarray(times)[pointing_times])
            track_times_dict[label] = track_times
            target_visibility = track_data.visibility.get(recv=i_receiver,
                                                          freq=self.target_channels,
                                                          time=track_times)
            flags = None
            bandpass_std_pointing = target_visibility.standard_deviation(axis=0, flags=flags)
            bandpass_pointing = target_visibility.mean(axis=0, flags=flags)
            bandpasses_std_dict[label] = bandpass_std_pointing
            bandpasses_dict[label] = bandpass_pointing

        return bandpasses_std_dict, bandpasses_dict, track_times_dict

    def plot_corrected_track_bandpasses(self,
                                        bandpasses_dict,
                                        epsilon,
                                        frequencies,
                                        before_or_after,
                                        receiver_path):
        plt.figure()
        for key, bandpass in bandpasses_dict.items():
            plt.plot(frequencies.squeeze / MEGA, bandpass.squeeze / (epsilon + 1), label=key)
        plt.legend()
        plt.xlabel('frequency [MHz]')
        plt.ylabel('intensity')
        plt.savefig(os.path.join(receiver_path, f'{self.plot_name}_no_wiggle_pointings_{before_or_after}.png'))
        plt.close()

    @staticmethod
    def add_to_dicts_and_receiver_path(
            receiver: Receiver,
            parameters_dict: dict,
            epsilon_function_dict: dict,
            output_path: str
    ) -> str:
        """ Create directories if not existing and return default results path. """
        if receiver.name not in parameters_dict:
            parameters_dict[receiver.name] = {}  # type: dict[dict[float]]
        if receiver.name not in epsilon_function_dict:
            epsilon_function_dict[receiver.name] = {}  # type: dict[Callable]
        if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
            os.makedirs(receiver_path)
        return receiver_path
