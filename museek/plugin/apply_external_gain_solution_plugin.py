import os
import pickle

import numpy as np

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData


class ApplyExternalGainSolutionPlugin(AbstractPlugin):
    """ Plugin to load and apply an external gain solution. """

    def __init__(self, gain_file_path: str):
        """ Initialise with the path to the folder containing the gain file `gain_file_path`. """
        super().__init__()
        self.gain_file_path = gain_file_path

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
                             Requirement(location=ResultEnum.TRACK_DATA, variable='track_data')]

    def run(self, block_name, track_data: TimeOrderedData):
        """
        Run the plugin. The file containing the gain data is assumed to have `katcali` specific name and structure.
        Unavailable gain solutions are set to 1.
        :param block_name: name of the observation block
        :param track_data: `TimeOrderedDatata` containing the tracking part of the observation
        """
        track_data.load_visibility_flags_weights()
        gain_solution_array = np.ones(track_data.shape)
        gain_solution_mask_array = np.ones(track_data.shape)
        for i_receiver, receiver in enumerate(track_data.receivers):
            try:
                full_gain_file_path = os.path.join(self.gain_file_path,
                                                   f'{block_name}/{block_name}_{receiver.name}_level2_data')
                gain_file = pickle.load(open(full_gain_file_path, 'rb'), encoding='latin-1')
                receiver_gain_solution = gain_file['gain_map'].data
                receiver_gain_solution[receiver_gain_solution == 0] = 1.0  # we cannot divide by zero
                gain_solution_array[:, :, i_receiver] = receiver_gain_solution
                gain_solution_mask_array[:, :, i_receiver] = gain_file['gain_map'].mask
            except FileNotFoundError:
                print(f'No gain solution file for {block_name} {receiver.name} found, solution set to 1.')
                continue
        track_data.set_gain_solution(gain_solution_array=gain_solution_array,
                                     gain_solution_mask_array=gain_solution_mask_array)
        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=track_data))
