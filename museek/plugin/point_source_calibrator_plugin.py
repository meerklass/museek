import os

import numpy as np
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from matplotlib import pyplot as plt

from museek.enums.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.track_pointing_iterator import TrackPointingIterator


class PointSourceCalibratorPlugin(AbstractPlugin):
    """ Incomplete plugin to calibrate from point sources. For later completion. """

    def set_requirements(self):
        """ Define the plugin requirements """
        self.requirements = [Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.SCAN_OBSERVATION_START, variable='scan_start'),
                             Requirement(location=ResultEnum.SCAN_OBSERVATION_END, variable='scan_end')]

    def run(self,
            track_data: TimeOrderedData,
            scan_start: float,
            scan_end: float,
            output_path: str):
        """
        Skeleton of the point source calibrator plugin `run` method. Loops through the calibrator pointing and
        plots the pointing centers on a coordinate grid.
        :param track_data: tracking part of the time ordered data
        :param scan_start: time dump [s] of scan observation start
        :param scan_end: time dump [s] of scan observation end
        :param output_path: path to store plot
        """
        for receiver in track_data.receivers:
            print(f'Working on {receiver.name}...')
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            track_pointing_iterator = TrackPointingIterator(track_data=track_data,
                                                            receiver=receiver,
                                                            plot_dir=receiver_path,
                                                            scan_start=scan_start,
                                                            scan_end=scan_end)
            i_antenna = track_data.antenna_index_of_receiver(receiver=receiver)
            for before_or_after, times, pointing_times_list, pointing_centres in track_pointing_iterator.iterate():
                plot_name = os.path.join(receiver_path, f'point_source_calibrator_plugin_{before_or_after}.png')
                print(f'Saving plot {plot_name}.')
                colors = ['red', 'blue', 'green', 'orange', 'yellow', 'black', 'purple']
                for centre, pointing_times, color in zip(pointing_centres, pointing_times_list, colors):
                    track_times = list(np.asarray(times)[pointing_times])
                    plt.scatter(centre[0], centre[1], marker='x', s=100, c=color)
                    right_ascension = track_data.right_ascension.get(time=track_times, recv=i_antenna).squeeze
                    declination = track_data.declination.get(time=track_times, recv=i_antenna).squeeze
                    plt.scatter(right_ascension, declination, c=color)
                plt.savefig(plot_name)
                plt.close()
