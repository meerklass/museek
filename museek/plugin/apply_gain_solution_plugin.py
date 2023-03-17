import os
import pickle

import numpy as np

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.visualiser import waterfall
from matplotlib import pyplot as plt


class ApplyGainSolutionPlugin(AbstractPlugin):

    def __init__(self, gain_file_path: str):
        super().__init__()
        self.gain_file_path = gain_file_path

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
                             Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, block_name, scan_data: TimeOrderedData, output_path: str):
        scan_data.load_visibility_flags_weights()
        gain_solution_array = np.ones(scan_data.shape)
        gain_solution_mask_array = np.ones(scan_data.shape)
        for i_receiver, receiver in enumerate(scan_data.receivers):
            try:
                full_gain_file_path = os.path.join(self.gain_file_path,
                                                   f'{block_name}/{block_name}_{receiver.name}_level2_data')
                gain_file = pickle.load(open(full_gain_file_path, 'rb'), encoding='latin-1')
                receiver_gain_solution = gain_file['gain_map'].data
                receiver_gain_solution[receiver_gain_solution==0] = 1.0
                gain_solution_array[:,:,i_receiver] = receiver_gain_solution
                gain_solution_mask_array[:,:,i_receiver] = gain_file['gain_map'].mask
            except FileNotFoundError:
                print(f'No gain solution file for {block_name} {receiver.name} found, solution set to 1.')
                continue
        scan_data.set_gain_solution(gain_solution_array=gain_solution_array,
                                    gain_solution_mask_array=gain_solution_mask_array)
        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data))

        corrected_visibility = scan_data.visibility.get(recv=0) \
                               * scan_data.gain_solution.get(recv=0)
        print(corrected_visibility.shape)

        waterfall(corrected_visibility, flags=scan_data.flags.get(recv=0), flag_threshold=1)
        plt.show()