import os

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from matplotlib import pyplot as plt

import numpy as np


class BandpassPlugin(AbstractPlugin):
    def __init__(self,
                 target_channels: range | list[int],
                 zebra_channels: range | list[int]):
        """
        Initialise
        :param target_channels: `list` or `range` of channel indices affected by the emission from the vanwyksvlei tower
        """
        super().__init__()
        self.target_channels = target_channels
        self.zebra_channels = zebra_channels

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, track_data: TimeOrderedData, output_path: str):
        context_file_name = 'bandpass_plugin.pickle'
        # self.store_context_to_disc(context_file_name=context_file_name,
        #                            context_directory=output_path)
        self.store_context_to_disc(context_file_name=None,
                                   context_directory=None)

        # hack to deal with context created on ilifu:
        output_path = '/home/amadeus/git/museek/results/1638898468/'

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

            # plt.plot(timestamps[times_1], track_data.visibility.get(recv=i_receiver,
            #                                                         freq=self.target_channels[0],
            #                                                         time=times_1).squeeze)
            # plt.show()
            #
            # plt.plot(timestamps[times_2], track_data.visibility.get(recv=i_receiver,
            #                                                         freq=self.target_channels[0],
            #                                                         time=times_2).squeeze)
            # plt.show()

            bandpasses = {}
            mean_az_el = {}

            for before_or_after, times in zip(['before_scan', 'after_scan'], [times_1, times_2]):
                # # big waterfall
                # extent = [timestamps[times][0],
                #           timestamps[times][-1],
                #           track_data.frequencies.get(freq=self.zebra_channels[-1]).squeeze / mega,
                #           track_data.frequencies.get(freq=self.zebra_channels[0]).squeeze / mega]
                # image = plt.imshow(track_data.visibility.get(recv=i_receiver,
                #                                              freq=self.zebra_channels,
                #                                              time=times).squeeze.T,
                #                    aspect='auto',
                #                    extent=extent,
                #                    cmap='gist_ncar',
                #                    norm='log')
                # plt.colorbar(image)
                # plt.xlabel('timestamp [s]')
                # plt.ylabel('frequency [MHz]')
                # plt.title('"zebra" channels')
                # plt.show()

                right_ascension = track_data.right_ascension.get(recv=i_receiver // 2,
                                                                 time=times).squeeze
                declination = track_data.declination.get(recv=i_receiver // 2,
                                                         time=times).squeeze
                azimuth = track_data.azimuth.get(recv=i_receiver // 2,
                                                 time=times).squeeze
                elevation = track_data.elevation.get(recv=i_receiver // 2,
                                                     time=times).squeeze

                mean_azimuth = track_data.azimuth.get(recv=i_receiver // 2,
                                                      time=times).mean(axis=0).squeeze
                mean_elevation = track_data.elevation.get(recv=i_receiver // 2,
                                                          time=times).mean(axis=0).squeeze

                mean_az_el[before_or_after] = (mean_azimuth, mean_elevation)

                centre_coord = (79.95, -45.78)
                up_coord = (79.963, -45.283)
                right_coord = (80.680, -45.778)
                # center_coord = ((np.min(right_ascension) + np.max(right_ascension)) / 2,
                #                 (np.min(declination) + np.max(declination)) / 2)
                tolerance = .1
                centre_times = np.where(
                    (abs(right_ascension - centre_coord[0]) < tolerance)
                    & (abs(declination - centre_coord[1]) < tolerance)
                )[0]
                up_times = np.where(
                    (abs(right_ascension - up_coord[0]) < tolerance)
                    & (abs(declination - up_coord[1]) < tolerance)
                )[0]
                right_times = np.where(
                    (abs(right_ascension - right_coord[0]) < tolerance)
                    & (abs(declination - right_coord[1]) < tolerance)
                )[0]

                plt.scatter(right_ascension, declination, color='black')
                plt.scatter(right_ascension[centre_times], declination[centre_times], color='red')
                plt.scatter(right_ascension[up_times], declination[up_times], color='blue')
                plt.scatter(right_ascension[right_times], declination[right_times], color='orange')
                plt.savefig(os.path.join(receiver_path, f'track_pointing_{before_or_after}.png'))
                plt.close()

                plt.scatter(azimuth, elevation, color='black')
                plt.scatter(azimuth[centre_times], elevation[centre_times], color='red')
                plt.scatter(azimuth[up_times], elevation[up_times], color='blue')
                plt.scatter(azimuth[right_times], elevation[right_times], color='orange')
                plt.savefig(os.path.join(receiver_path, f'track_pointing_az_el_{before_or_after}.png'))
                plt.close()

                # track_times = [time_ for time_ in times if time_-times[0] in center_times]
                track_centre_times = list(np.asarray(times)[centre_times])
                track_up_times = list(np.asarray(times)[up_times])
                track_right_times = list(np.asarray(times)[right_times])
                track_centre_timestamps = track_data.timestamps.get(time=track_centre_times).squeeze
                track_up_timestamps = track_data.timestamps.get(time=track_up_times).squeeze
                track_right_timestamps = track_data.timestamps.get(time=track_right_times).squeeze

                plt.figure(figsize=(6, 12))
                plt.subplot(2, 1, 1)
                extent = [track_centre_timestamps[0],
                          track_centre_timestamps[-1],
                          track_data.frequencies.get(freq=self.target_channels[-1]).squeeze / mega,
                          track_data.frequencies.get(freq=self.target_channels[0]).squeeze / mega]
                image = plt.imshow(track_data.visibility.get(recv=i_receiver,
                                                             freq=self.target_channels,
                                                             time=track_centre_times).squeeze.T,
                                   aspect='auto',
                                   extent=extent,
                                   cmap='gist_ncar',
                                   norm='log')
                plt.colorbar(image)
                plt.xlabel('timestamp [s]')
                plt.ylabel('frequency [MHz]')
                plt.title('"clean" channels')

                plt.subplot(2, 1, 2)
                extent = [track_centre_timestamps[0],
                          track_centre_timestamps[-1],
                          track_data.frequencies.get(freq=self.zebra_channels[-1]).squeeze / mega,
                          track_data.frequencies.get(freq=self.zebra_channels[0]).squeeze / mega]
                image = plt.imshow(track_data.visibility.get(recv=i_receiver,
                                                             freq=self.zebra_channels,
                                                             time=track_centre_times).squeeze.T,
                                   aspect='auto',
                                   extent=extent,
                                   cmap='gist_ncar',
                                   norm='log')
                plt.colorbar(image)
                plt.xlabel('timestamp [s]')
                plt.ylabel('frequency [MHz]')
                plt.title('"zebra" channels')

                plt.savefig(os.path.join(receiver_path, f'track_waterfall_clean_and_zebra_{before_or_after}.png'))
                plt.close()

                target_visibility_centre = track_data.visibility.get(recv=i_receiver,
                                                              freq=self.target_channels,
                                                              time=track_centre_times)
                target_visibility_right = track_data.visibility.get(recv=i_receiver,
                                                              freq=self.target_channels,
                                                              time=track_right_times)
                target_visibility_up = track_data.visibility.get(recv=i_receiver,
                                                              freq=self.target_channels,
                                                              time=track_up_times)
                # bandpass_centre = target_visibility_centre.mean(axis=0).squeeze
                bandpass_up_minus_right = target_visibility_up.mean(axis=0).squeeze - target_visibility_right.mean(axis=0).squeeze
                # bandpasses[before_or_after] = bandpass_centre
                bandpasses[before_or_after] = bandpass_up_minus_right

            bandpass_tower_off = bandpasses['before_scan']
            bandpass_tower_on = bandpasses['after_scan']
            plt.figure(figsize=(6, 12))
            plt.subplot(2, 1, 1)
            plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / mega, bandpass_tower_off[1:],
                     label='tower off')
            plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / mega, bandpass_tower_on[1:],
                     label='tower on')
            plt.xlabel('frequency [MHz]')
            plt.ylabel('mean bandpass')
            plt.legend()

            bandpass_diff = (bandpass_tower_on - bandpass_tower_off) / bandpass_tower_off * 100
            plt.subplot(2, 1, 2)
            plt.plot(track_data.frequencies.get(freq=self.target_channels).squeeze[1:] / mega, bandpass_diff[1:])
            plt.xlabel('frequency [MHz]')
            plt.ylabel('bandpass (tower on - tower off) / tower off %')
            plt.savefig(os.path.join(receiver_path, 'bandpass_during_tracking.png'))
            plt.close()
