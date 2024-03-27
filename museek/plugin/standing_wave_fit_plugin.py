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
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData
from museek.util.track_pointing_iterator import TrackPointingIterator


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
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.SCAN_OBSERVATION_START, variable='scan_start'),
                             Requirement(location=ResultEnum.SCAN_OBSERVATION_END, variable='scan_end')]

    def run(self,
            track_data: TimeOrderedData,
            output_path: str,
            scan_start: float,
            scan_end: float):
        """
        Run the plugin, i.e. fit the bandpass model and store the results
        :param track_data: the time ordered data containing the observation's tracking part
        :param output_path: path to store results
        :param scan_start: 
        :param scan_end:
        """

        parameters_dict_name = f'parameters_dict_frequency_' \
                               f'{track_data.frequencies.get(freq=self.target_channels[0]).squeeze / MEGA:.0f}_to_' \
                               f'{track_data.frequencies.get(freq=self.target_channels[-1]).squeeze / MEGA:.0f}' \
                               f'_MHz.json'

        track_data.load_visibility_flags_weights()

        parameters_dict = {}  # type: dict[dict[dict[float]]]
        epsilon_function_dict = {}  # type: dict[dict[[Callable]]]
        legendre_function_dict = {}  # type: dict[dict[[Callable]]]

        for i_receiver, receiver in enumerate(track_data.receivers):
            # if receiver.name != 'm008v':
            #     continue
            print(f'Working on {receiver}...')

            receiver_path = self.add_to_dicts_and_receiver_path(receiver=receiver,
                                                                parameters_dict=parameters_dict,
                                                                epsilon_function_dict=epsilon_function_dict,
                                                                legendre_function_dict=legendre_function_dict,
                                                                output_path=output_path)
            track_pointing_iterator = TrackPointingIterator(track_data=track_data,
                                                            receiver=receiver,
                                                            plot_dir=receiver_path,
                                                            scan_start=scan_start,
                                                            scan_end=scan_end)

            for before_or_after, times, times_list, pointing_centres in track_pointing_iterator.iterate():
                print(f'{before_or_after}...')
                if times_list is None:
                    print('calibrator not found?... continue')
                    continue
                _, bandpasses_dict, _ = self.get_bandpasses_std_dicts(
                    track_data=track_data,
                    times_list=times_list,
                    times=times,
                    pointing_labels=self.pointing_labels,
                    i_receiver=i_receiver
                )
                frequencies = track_data.frequencies.get(freq=self.target_channels)
                # for label in self.pointing_labels:
                for label, bandpass_estimator in bandpasses_dict.items():
                    print(f'found {label}...')
                    bandpass_estimator = bandpasses_dict[label]

                    bandpass_model = BandpassModel(
                        plot_name=self.plot_name,
                        standing_wave_displacements=[14.7, 13.4, 16.2, 17.9, 12.4, 19.6, 11.7, 5.8],
                        legendre_degree=1,
                        polyphase_parameters=(6, 64, 1.0003)
                    )
                    flags = track_data.flags.get(time=times,
                                                 freq=self.target_channels,
                                                 recv=i_receiver)
                    if flags.combine().squeeze.all():
                        print('Everything flagged... - continue')
                        continue
                    dict_key = f'{before_or_after}_{label}'
                    fit_args = dict(frequencies=frequencies,
                                    estimator=bandpass_estimator,
                                    receiver_path=receiver_path,
                                    calibrator_label=dict_key)
                    try:
                        bandpass_model.fit(**fit_args)
                    except RuntimeError:
                        print('RuntimeError: continue...')
                        continue
                    parameters_dict[receiver.name][dict_key] = bandpass_model.parameters_dictionary
                    epsilon_function_dict[receiver.name][dict_key] = bandpass_model.epsilon_function
                    legendre_function_dict[receiver.name][dict_key] = bandpass_model.legendre_function

                    if self.do_store_parameters:
                        np.savez(
                            os.path.join(
                                receiver_path,
                                f'standing_wave_epsilon_and_frequencies_track_{dict_key}'
                            ),
                            epsilon=bandpass_model.epsilon,
                            frequencies=frequencies.squeeze/MEGA
                        )
                    self.plot_corrected_track_bandpasses(bandpass=bandpass_estimator,
                                                         epsilon=bandpass_model.epsilon,
                                                         frequencies=frequencies,
                                                         before_or_after=dict_key,
                                                         receiver_path=receiver_path)

        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_EPSILON_FUNCTION_DICT,
                                      result=epsilon_function_dict,
                                      allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_LEGENDRE_FUNCTION_DICT,
                                      result=legendre_function_dict,
                                      allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_CHANNELS,
                                      result=self.target_channels,
                                      allow_overwrite=False))

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
                                        bandpass,
                                        epsilon,
                                        frequencies,
                                        before_or_after,
                                        receiver_path):
        plt.figure()
        plt.plot(frequencies.squeeze / MEGA, bandpass.squeeze / (epsilon + 1), label=before_or_after)
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
            legendre_function_dict: dict,
            output_path: str
    ) -> str:
        """ Create directories if not existing and return default results path. """
        if receiver.name not in parameters_dict:
            parameters_dict[receiver.name] = {}  # type: dict[dict[float]]
        if receiver.name not in epsilon_function_dict:
            epsilon_function_dict[receiver.name] = {}  # type: dict[Callable]
        if receiver.name not in legendre_function_dict:
            legendre_function_dict[receiver.name] = {}  # type: dict[Callable]
        if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
            os.makedirs(receiver_path)
        return receiver_path
