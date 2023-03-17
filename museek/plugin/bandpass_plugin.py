import os

import numpy as np
from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData


class BandpassPlugin(AbstractPlugin):
    """ Example plugin to help later development """

    def __init__(self,
                 target_channels: range | list[int],
                 centre_coord: tuple[float, float],
                 pointing_tolerance: float):
        """
        Initialise
        :param target_channels: `list` or `range` of channel indices to be examined
        :param centre_coord: calibrator coordinate in degrees right ascension and declination
        :param pointing_tolerance: deviations up to this tolerance from the pointing are accepted
        """
        super().__init__()
        self.target_channels = target_channels
        self.centre_coord = centre_coord
        self.pointing_tolerance = pointing_tolerance

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, track_data: TimeOrderedData, output_path: str):
        """
        Split the tracking data in the centre observation parts before and after scanning
        and plot their frequency dependence.
        """

        mega = 1e6
        track_data.load_visibility_flags_weights()
        timestamps = track_data.timestamps.squeeze
        track_1_end_index = 340
        track_2_start_index = 345
        times_1 = range(track_1_end_index)
        times_2 = range(track_2_start_index, len(timestamps))

        for i_receiver, receiver in enumerate(track_data.receivers):
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)

            bandpasses = {}
            for before_or_after, times in zip(['before_scan', 'after_scan'], [times_1, times_2]):
                right_ascension = track_data.right_ascension.get(recv=i_receiver // 2,
                                                                 time=times).squeeze
                declination = track_data.declination.get(recv=i_receiver // 2,
                                                         time=times).squeeze

                centre_times = np.where(
                    (abs(right_ascension - self.centre_coord[0]) < self.pointing_tolerance)
                    & (abs(declination - self.centre_coord[1]) < self.pointing_tolerance)
                )[0]
                all_times = np.where(
                    (abs(right_ascension - self.centre_coord[0]) < 5)
                    & (abs(declination - self.centre_coord[1]) < 5)
                )[0]

                plt.scatter(right_ascension[all_times], declination[all_times], color='black')
                plt.scatter(right_ascension[centre_times], declination[centre_times], color='red')
                plt.savefig(os.path.join(receiver_path, f'track_pointing_{before_or_after}.png'))
                plt.close()

                track_centre_times = list(np.asarray(times)[centre_times])
                target_visibility_centre = track_data.visibility.get(recv=i_receiver,
                                                                     freq=self.target_channels,
                                                                     time=track_centre_times)
                bandpass_centre = target_visibility_centre.mean(axis=0).squeeze
                bandpasses[before_or_after] = bandpass_centre

            bandpass_before_scan = bandpasses['before_scan']
            bandpass_after_scan = bandpasses['after_scan']

            plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / mega, bandpass_before_scan[1:],
                     label='before scan')
            plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / mega, bandpass_after_scan[1:],
                     label='after scan')
            plt.xlabel('frequency [MHz]')
            plt.ylabel('mean bandpass')
            plt.legend()
            plt.savefig(os.path.join(receiver_path, 'bandpass_during_tracking.png'))
