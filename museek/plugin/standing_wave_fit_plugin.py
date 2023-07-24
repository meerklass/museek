import json
import os
from typing import Callable

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy.polynomial import legendre

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.data_element import DataElement
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.clustering import Clustering

MEGA = 1e6


class StandingWaveFitPlugin(AbstractPlugin):
    """ """

    def __init__(self, target_channels: range | list[int] | None, do_store_parameters: bool = False):
        """
        Initialise
        :param target_channels: optional `list` or `range` of channel indices to be examined, if `None`, all are used
        :param do_store_parameters: whether to store the standing wave fit parameters to a file
        """
        super().__init__()
        self.target_channels = target_channels
        self.plot_name = 'panel_gap_reflection_model'
        self.do_store_parameters = do_store_parameters

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, track_data: TimeOrderedData, output_path: str):
        """
        DOCUMENTATION
        :param track_data: the time ordered data containing the observation's tracking part
        :param output_path: path to store results
        """

        parameters_dict_name = f'parameters_dict_frequency_' \
                               f'{track_data.frequencies.get(freq=self.target_channels[0]).squeeze / MEGA:.0f}_to_' \
                               f'{track_data.frequencies.get(freq=self.target_channels[-1]).squeeze / MEGA:.0f}' \
                               f'_MHz.json'
        pointing_labels = ['on centre 1',
                           'off centre top',
                           'on centre 2',
                           'off centre right',
                           'on centre 3',
                           'off centre down',
                           'on centre 4',
                           'off centre left']

        track_data.load_visibility_flags_weights()
        parameters_dict = {}  # type: dict[dict[dict[float]]]
        target_dumps_list = Clustering().ordered_dumps_of_coherent_clusters(features=track_data.timestamps.squeeze,
                                                                            n_clusters=2)

        epsilon_function_dict = {}  # type: dict[dict[[Callable]]]

        for i_receiver, receiver in enumerate(track_data.receivers):
            print(f'Working on {receiver}...')
            if receiver.name not in parameters_dict:
                parameters_dict[receiver.name] = {}  # type: dict[dict[float]]
            if receiver.name not in epsilon_function_dict:
                epsilon_function_dict[receiver.name] = {}  # type: dict[Callable]
            try:
                if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                    os.makedirs(receiver_path)

                for before_or_after, times in zip(['before_scan', 'after_scan'], target_dumps_list):
                    print(f'Working on {before_or_after}...')
                    right_ascension = track_data.right_ascension.get(recv=i_receiver // 2,
                                                                     time=times).squeeze
                    declination = track_data.declination.get(recv=i_receiver // 2,
                                                             time=times).squeeze
                    timestamps = track_data.timestamps.get(time=times).squeeze
                    times_list, pointing_centres = Clustering().split_pointings(
                        coordinate_1=right_ascension,
                        coordinate_2=declination,
                        timestamps=timestamps,
                        n_pointings=5,
                        n_centre_observations=4,
                        distance_threshold=5.,
                    )
                    bandpasses_dict, track_times_dict = self.get_bandpasses_dicts(
                        track_data=track_data,
                        times_list=times_list,
                        times=times,
                        pointing_labels=pointing_labels,
                        i_receiver=i_receiver
                    )
                    flags = None
                    # mean_bandpass = track_data.visibility.get(time=times,
                    #                                           freq=self.target_channels,
                    #                                           recv=i_receiver).mean(axis=0, flags=flags)
                    # mean_bandpass = (bandpasses_dict['on centre 1'] + bandpasses_dict['off centre top']) / 2
                    mean_bandpass = bandpasses_dict['on centre 1'] + bandpasses_dict['on centre 2'] \
                                    + bandpasses_dict['on centre 3'] + bandpasses_dict['on centre 4']

                    frequencies = track_data.frequencies.get(freq=self.target_channels)
                    epsilon, parameters, variances, epsilon_function = self.epsilon(frequencies,
                                                                                    bandpass=mean_bandpass,
                                                                                    receiver_path=receiver_path,
                                                                                    before_or_after=before_or_after)
                    parameters_dict[receiver.name][before_or_after] = parameters
                    epsilon_function_dict[receiver.name][before_or_after] = epsilon_function

            except ValueError:
                # print(f'Receiver {receiver.name} failed to process. Continue...')
                raise

        if self.do_store_parameters:
            with open(os.path.join(output_path, parameters_dict_name), 'w') as f:
                json.dump(parameters_dict, f)

        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_EPSILON_FUNCTION_DICT,
                                      result=epsilon_function_dict,
                                      allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.STANDING_WAVE_CHANNELS,
                                      result=self.target_channels,
                                      allow_overwrite=False))

    def epsilon(self,
                frequencies: DataElement,
                bandpass: DataElement,
                receiver_path: str,
                before_or_after: str,
                do_plot: bool = True):
        """ DOC """
        legendre_degree = 1
        displacements = [14.7, 13.4, 16.2, 17.9, 12.4, 19.6, 11.7, 5.8]
        wavelengths = [d * 2 for d in displacements]
        target_frequencies = frequencies.squeeze / MEGA
        target_bandpass = bandpass.squeeze

        starting_legendre_coefficients = [x for x in legendre.legfit(
            target_frequencies,
            target_bandpass,
            legendre_degree
        )]
        starting_coefficients = starting_legendre_coefficients + [0.1 * (i % 2)
                                                                  for i in range(len(wavelengths) * 2)]

        def bandpass_model_wrapper(f, *coefficients):
            sinus_coefficient_list = self._sinus_coefficient_list(
                coefficients=coefficients,
                n_legendre_coefficients=len(starting_legendre_coefficients),
                wavelengths=wavelengths
            )
            return self.bandpass_model(frequencies=f,
                                       legendre_coefficients=coefficients[:len(starting_legendre_coefficients)],
                                       sinus_coefficient_list=sinus_coefficient_list)

        lower_bounds = np.asarray([-np.inf for _ in starting_legendre_coefficients] + [-np.pi, 0] * len(wavelengths))
        upper_bounds = np.asarray([np.inf for _ in starting_legendre_coefficients] + [np.pi, 1] * len(wavelengths))
        bounds = (lower_bounds, upper_bounds)

        curve_fit = scipy.optimize.curve_fit(bandpass_model_wrapper,
                                             target_frequencies,
                                             target_bandpass,
                                             p0=starting_coefficients,
                                             bounds=bounds)

        model_bandpass = bandpass_model_wrapper(target_frequencies, *curve_fit[0])
        smooth_bandpass = legendre.legval(target_frequencies, curve_fit[0][:len(starting_legendre_coefficients)])
        epsilon = model_bandpass / smooth_bandpass - 1

        parameters_dict = self.parameters_to_dict(curve_fit[0],
                                                  n_legendre_coefficients=len(starting_legendre_coefficients),
                                                  wavelengths=wavelengths)
        variances_dict = self.parameters_to_dict(np.diag(curve_fit[1]),
                                                 n_legendre_coefficients=len(starting_legendre_coefficients),
                                                 wavelengths=wavelengths)

        if do_plot:
            self.plot(frequencies=target_frequencies,
                      bandpass=target_bandpass,
                      model_bandpass=model_bandpass,
                      smooth_bandpass=smooth_bandpass,
                      epsilon=epsilon,
                      receiver_path=receiver_path,
                      before_or_after=before_or_after)

        def epsilon_function(f):
            return bandpass_model_wrapper(f.squeeze / MEGA, *curve_fit[0]) / smooth_bandpass - 1

        return epsilon, parameters_dict, variances_dict, epsilon_function

    @staticmethod
    def _sinus_coefficient_list(coefficients, n_legendre_coefficients, wavelengths):
        sinus_coefficient_list = [
            (coefficients[n_legendre_coefficients + 2 * i],
             coefficients[n_legendre_coefficients + 2 * i + 1],
             w)
            for i, w in enumerate(wavelengths)
        ]
        return sinus_coefficient_list

    @staticmethod
    def parameters_to_dict(parameters, n_legendre_coefficients, wavelengths):
        """ DOC """
        parameter_dict = {f'l_{i}': l for i, l in enumerate(parameters[:n_legendre_coefficients])}
        for i, w in enumerate(wavelengths):
            parameter_dict[f'wavelength_{w}_phase'] = parameters[n_legendre_coefficients + 2 * i]
            parameter_dict[f'wavelength_{w}_amplitude'] = parameters[n_legendre_coefficients + 2 * i + 1]
        return parameter_dict

    def bandpass_model(self,
                       frequencies,
                       legendre_coefficients,
                       sinus_coefficient_list: list[tuple[float, float, float]]):
        """ DOC """
        legendre_fit = legendre.legval(frequencies, legendre_coefficients)
        sinusoidal_fits = [self.sinusoidal_fit(frequencies, sinus_coefficients)
                           for sinus_coefficients in sinus_coefficient_list]
        return legendre_fit * (1 + sum(sinusoidal_fits))

    @staticmethod
    def sinusoidal_fit(frequencies, coefficients: tuple[float, float, float]):
        """ DOC """
        speed_of_light = 3e8  # m/s
        phase, amplitude, wavelength = coefficients
        return amplitude * np.sin(phase + frequencies * 2 * np.pi / (speed_of_light / wavelength / MEGA))

    def plot(self,
             frequencies: np.ndarray,
             bandpass: np.ndarray,
             model_bandpass: np.ndarray,
             smooth_bandpass: np.ndarray,
             epsilon: np.ndarray,
             receiver_path: str,
             before_or_after: str):
        """ DOC """
        plt.figure(figsize=(16, 24))

        plt.subplot(4, 1, 1)
        plt.plot(frequencies, bandpass, label='bandpass mean')
        plt.plot(frequencies, model_bandpass, ls=':', color='black', label='model')
        plt.plot(frequencies, smooth_bandpass, label='smooth model')
        plt.xlabel('frequency [MHz]')
        plt.ylabel('intensity')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(frequencies, epsilon, ls=':', color='black', label='epsilon wiggle model')
        plt.plot(frequencies, bandpass / smooth_bandpass - 1, label='epsilon wiggle data')
        plt.xlabel('frequency [MHz]')
        plt.legend()

        plt.subplot(4, 1, 3)
        residual = (bandpass - model_bandpass) / bandpass * 100
        plt.plot(frequencies, residual, color='black')
        plt.xlabel('frequency [MHz]')
        plt.ylabel('residual [%]')

        plt.subplot(4, 1, 4)
        plt.hist(residual, bins=50, color='black')
        plt.xlabel('residual [%]')
        plt.ylabel('histogram')
        plt.savefig(os.path.join(receiver_path, f'{self.plot_name}_{before_or_after}.png'))
        plt.close()

    def get_bandpasses_dicts(self, track_data, times_list, times, pointing_labels, i_receiver):
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

            bandpass_pointing = target_visibility.mean(axis=0, flags=flags)
            bandpasses_dict[label] = bandpass_pointing

        return bandpasses_dict, track_times_dict
