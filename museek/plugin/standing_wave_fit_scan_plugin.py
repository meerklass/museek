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
from museek.time_ordered_data import DataElementFactory, TimeOrderedData
from museek.util.swings import Swings


class StandingWaveFitScanPlugin(AbstractPlugin):
    """
    Experimental plugin to correct standing waves from data using time-averaged visibilities.
    """

    def __init__(self):
        """
        Initialise
        """
        super().__init__()
        self.plot_name = 'standing_wave_scan_plugin'
        self.off_cut_label = 'off_cut'
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

        scan_data.load_visibility_flags_weights()

        bandpass_estimator_before_list = []
        bandpass_estimator_after_list = []

        for i_receiver, receiver in enumerate(scan_data.receivers):
            print(f'Working on {receiver.name}')
            i_antenna = receiver.antenna_index(receivers=scan_data.receivers)
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            print(f'Output goes to {receiver_path}...')
            times_before, times_after = self.calibrator_times(data=scan_data, i_antenna=i_antenna)
            self.plot_times(data=scan_data, times=[times_before, times_after], i_antenna=i_antenna, output_path=receiver_path)

            flags = scan_data.flags.get(time=times_before, recv=i_receiver)
            bandpass_estimator_before = scan_data.visibility.get(time=times_before,
                                                                 recv=i_receiver).mean(axis=0, flags=flags)
            bandpass_estimator_before_list.append(bandpass_estimator_before.squeeze)
            flags = scan_data.flags.get(time=times_after, recv=i_receiver)
            bandpass_estimator_after = scan_data.visibility.get(time=times_after,
                                                                recv=i_receiver).mean(axis=0, flags=flags)
            bandpass_estimator_after_list.append(bandpass_estimator_after.squeeze)
            
        bandpass_estimator_after_array = np.asarray(bandpass_estimator_after_list).T[np.newaxis]
        bandpass_estimator_before_array = np.asarray(bandpass_estimator_before_list).T[np.newaxis]
        bandpass_estimator_after = DataElementFactory().create(array=bandpass_estimator_after_array)
        bandpass_estimator_before = DataElementFactory().create(array=bandpass_estimator_before_array)

        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_BANDPASS_ESTIMATOR,
                                      result=[bandpass_estimator_before, bandpass_estimator_after],
                                      allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_CALIBRATOR_LABEL,
                                      result=self.calibrator_label,
                                      allow_overwrite=False))

    def calibrator_times(self, data: TimeOrderedData, i_antenna: int) -> list[range] | list[np.ndarray]:
        """
        Return the calibration time dump indices for antenna `i_antenna` in `data` as `list` of
        `range` or `np.ndarray`.
        """
        if self.calibrator_label == self.off_cut_label:
            return self.off_cut_dumps(data=data, i_antenna=i_antenna)
        else:
            raise NotImplementedError(f'No calibration implemented with label {self.calibrator_label}.'
                                      f'Available: {self.off_cut_label}')

    @staticmethod
    def off_cut_dumps(data: TimeOrderedData, i_antenna: int) -> list[range] | list[np.ndarray]:
        """
        Return the off-cut time dump indices at begginning and end of scan of antenna `i_antenna` in `data` as `list`
        of `range` or `np.ndarray`.
        """
        turnaround_dumps = Swings.swing_turnaround_dumps(azimuth=data.azimuth.get(recv=i_antenna))
        lower_offcut_threshold = (data.right_ascension.get(time=turnaround_dumps[0]).squeeze
                                  + data.right_ascension.get(time=turnaround_dumps[1]).squeeze) / 2
        upper_offcut_threshold = (data.right_ascension.get(time=-1).squeeze
                                  + data.right_ascension.get(time=turnaround_dumps[-1]).squeeze) / 2
        is_offcut_start = data.right_ascension.squeeze < lower_offcut_threshold
        is_offcut_end = data.right_ascension.squeeze > upper_offcut_threshold

        return [np.where(is_offcut_start)[0], np.where(is_offcut_end)[0]]

    @staticmethod
    def plot_times(data: TimeOrderedData, i_antenna: int, times: list[range] | list[np.ndarray], output_path: str):
        """ Plot the time dumps used for calibration on a coordinate grid. """
        right_ascension = data.right_ascension.get(recv=i_antenna).squeeze
        declination = data.declination.get(recv=i_antenna).squeeze
        colors = ['black' if i not in times[0] else 'red' for i in range(len(right_ascension))]
        colors = [c if i not in times[1] else 'blue' for i, c in enumerate(colors)]
        plt.scatter(right_ascension, declination, c=colors, s=5)
        plt.xlabel('right ascension [deg]')
        plt.ylabel('declination [deg]')
        plt.savefig(os.path.join(output_path, 'part_of_footprint_for_standing_wave_calibration.png'))
