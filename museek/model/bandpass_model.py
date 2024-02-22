import os
from typing import Callable, Optional
import emcee

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy.polynomial import legendre

from definitions import MEGA, SPEED_OF_LIGHT
from museek.data_element import DataElement


class BandpassModel:
    """ Class to model the bandpass shape including standing waves. """

    def __init__(self,
                 standing_wave_displacements: list[float],
                 legendre_degree: int,
                 polyphase_parameters: tuple[int, int, float],
                 plot_name: str | None = None):
        """
        Initialise
        :param standing_wave_displacements: list of `float` standing wave displacement distances [m]
        :param legendre_degree: degree of the legendre polynomial
        :param polyphase_parameters: `tuple` of first dip dump index and dip separation (usually 64)
                                      and neighbors/dip ratio
        :param plot_name: optional plot name, not created if `None`
        """
        self.plot_name = plot_name
        self.standing_wave_displacements = standing_wave_displacements
        self.wavelengths = [d * 2 for d in self.standing_wave_displacements]
        self.n_wave = len(standing_wave_displacements)
        self.legendre_degree = legendre_degree
        self.polyphase_parameters = polyphase_parameters

        # function to encapsulate the wiggly structure of the bandpass
        self.epsilon_function = None  # type: Optional[Callable]
        # function to encapsulate the flat, or legendre-like structure of the bandpass
        self.legendre_function = None  # type: Optional[Callable]
        self.variances_dictionary = None  # type: Optional[dict]
        self.parameters_dictionary = None  # type: Optional[dict]
        self.epsilon = None  # type: Optional[np.ndarray]

        self._curve_fit: list | None = None

    def _reset_wavelengths(self, wavelengths: list[float] | np.ndarray):
        """
        Set `self.n_wave`, `self.wavelengths` and `self.standing_wave_displacements` according to input `wavelengths`.
        """
        self.wavelengths = wavelengths
        self.standing_wave_displacements = [w / 2 for w in self.wavelengths]
        self.n_wave = len(self.standing_wave_displacements)

    def fit(self,
            frequencies: DataElement,
            estimator: DataElement,
            receiver_path: str,
            calibrator_label: str,
            starting_coefficients: list[float] | None = None):
        """
        Fit the standing wave model to a single receiver.
        :param frequencies: the frequencies as a `DataElement`
        :param estimator: the bandpass itself or an estimator thereof as `DataElement`
        :param receiver_path: path to store results specific to this receiver
        :param calibrator_label: usually 'before_scan' or 'after_scan' or similar
        :param starting_coefficients: optional `list` of `float` starting coefficients
        """
        target_frequencies = frequencies.squeeze / MEGA
        target_estimator = estimator.squeeze

        legendre_start_fit = legendre.legfit(
            target_frequencies,
            target_estimator,
            self.legendre_degree
        )
        n_legendre_coeff = len(legendre_start_fit)
        total_n_wavelengths = len(self.wavelengths)

        if starting_coefficients is None:
            # sinus_start_amplitude = 0.001  # recommended value
            sinus_start_amplitude = 0.1  # original
            # sinus_start_amplitude = 1.  # experimental
            starting_legendre_coefficients = [x for x in legendre_start_fit]
            # starting_coefficients = starting_legendre_coefficients + [sinus_start_amplitude * (i % 2)
            #                                                           for i in range(total_n_wavelengths * 2)]
            starting_coefficients = starting_legendre_coefficients
            for i in range(total_n_wavelengths):
                starting_coefficients += [0, sinus_start_amplitude, self.wavelengths[i]]

        def bandpass_model_wrapper(f: np.ndarray, *parameters) -> np.ndarray:
            """ Wrap the bandpass model for the scipy fit. """
            sinus_coefficient_list = self._sinus_parameter_list_free_wavelengths(
                parameters=parameters,
                n_legendre_coefficients=n_legendre_coeff,
            )
            return self._bandpass_model(frequencies=f,
                                        legendre_coefficients=parameters[:n_legendre_coeff],
                                        sinus_parameters_list=sinus_coefficient_list)

        # the amplitude limits seem to depend on case
        # sinus_upper_amplitude = 0.01  # recommended value
        # sinus_upper_amplitude = 0.1
        sinus_upper_amplitude = 100  # original value, works best, but not physical
        # lower_bounds = np.asarray([-np.inf for _ in range(n_legendre_coeff)] + [-np.pi, 0, 0.2] * self.n_wave)
        lower_bounds = np.asarray([-np.inf for _ in range(n_legendre_coeff)] + [-np.pi, 0, 0.5] * self.n_wave)
        upper_bounds = np.asarray(
            [np.inf for _ in range(n_legendre_coeff)] + [np.pi, sinus_upper_amplitude, 50] * self.n_wave
        )
        bounds = (lower_bounds, upper_bounds)

        curve_fit = self._fit(executable=bandpass_model_wrapper,
                              on_x_axis=target_frequencies,
                              estimator=target_estimator,
                              p0=starting_coefficients,
                              bounds=bounds)

        model_bandpass = bandpass_model_wrapper(target_frequencies, *curve_fit[0])
        legendre_bandpass = legendre.legval(target_frequencies, curve_fit[0][:n_legendre_coeff])
        epsilon = model_bandpass / legendre_bandpass - 1

        parameters_dict = self._parameters_to_dictionary(curve_fit[0], n_legendre_coefficients=n_legendre_coeff)
        variances_dict = self._parameters_to_dictionary(np.diag(curve_fit[1]), n_legendre_coefficients=n_legendre_coeff)

        if self.plot_name is not None:
            self._plot(frequencies=target_frequencies,
                       bandpass=target_estimator,
                       model_bandpass=model_bandpass,
                       smooth_bandpass=legendre_bandpass,
                       epsilon=epsilon,
                       receiver_path=receiver_path,
                       before_or_after=calibrator_label,
                       n_legendre_coeff=n_legendre_coeff,
                       parameters=curve_fit[0])

        def legendre_function(f: DataElement) -> np.ndarray:
            """ Return the legendre contribution to the bandpass as a function of frequency. """
            return legendre.legval(f.squeeze / MEGA, curve_fit[0][:n_legendre_coeff])

        def epsilon_function(f: DataElement) -> np.ndarray:
            """ Return the sinusoidal contribution to the bandpass as a function of frequency. """
            return bandpass_model_wrapper(f.squeeze / MEGA, *curve_fit[0]) / legendre_function(f) - 1

        self.epsilon_function = epsilon_function
        self.legendre_function = legendre_function
        self.variances_dictionary = variances_dict
        self.parameters_dictionary = parameters_dict
        self.epsilon = epsilon
        self._curve_fit = curve_fit

    @staticmethod
    def _fit(executable, on_x_axis, estimator, p0, bounds):
        return scipy.optimize.curve_fit(executable,
                                        on_x_axis,
                                        estimator,
                                        p0=p0,
                                        bounds=bounds,
                                        method='dogbox')
        
    @staticmethod
    def _fit_mcmc(executable, on_x_axis, estimator, p0, bounds):
        nwalkers=50
        ndim=len(p0)
        log_prob=1
        ivar = 1
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
        sampler.run_mcmc(p0, 10000)
        return None

    def double_fit(self,
                   n_double: int,
                   frequencies: DataElement,
                   estimator: DataElement,
                   receiver_path: str,
                   calibrator_label: str):
        """
        Fit the standing wave model including `n_double` double reflections to a single receiver.
        :param n_double: number of high-amplitude wavelengths to consider for double reflections
        :param frequencies: the frequencies as a `DataElement`
        :param estimator: the bandpass itself or an estimator thereof as `DataElement`
        :param receiver_path: path to store results specific to this receiver
        :param calibrator_label: usually 'before_scan' or 'after_scan' or similar
        """
        if self.epsilon_function is None:
            raise ValueError('Method `fit` must be run beforehand.')
        maximum_wavelengths = self._maximum_wavelengths(n=n_double)
        double_wavelengths = self._double_wavelengths(wavelengths_to_double=maximum_wavelengths)
        starting_coefficients = list(self._curve_fit[0]) + [0.001 * (i % 2)
                                                            for i in range(len(double_wavelengths) * 2)]
        wavelengths = np.append(self.wavelengths, double_wavelengths)
        self._reset_wavelengths(wavelengths=wavelengths)
        self.fit(frequencies=frequencies,
                 estimator=estimator,
                 receiver_path=receiver_path,
                 calibrator_label=calibrator_label,
                 starting_coefficients=starting_coefficients)

    def _bandpass_model(
            self,
            frequencies: np.ndarray,
            legendre_coefficients: np.ndarray | tuple,
            sinus_parameters_list: list[tuple[float, float, float]],
    ) -> np.ndarray:
        """
        Bandpass model based on a discrete set of standing waves
        :param frequencies: frequencies as array
        :param legendre_coefficients: the coefficients of the legendre polynomial
        :param sinus_parameters_list: `list` of the standing wave parameters as `tuple`s
        :return: the bandpass model as a `numpy` array
        """
        legendre_ = legendre.legval(frequencies, legendre_coefficients)
        sinusoidal = [self._sinus(frequencies, sinus_parameters) for sinus_parameters in sinus_parameters_list]
        polyphase_dips = self._polyphase_dips(frequencies=frequencies)
        return polyphase_dips * legendre_ * (1 + sum(sinusoidal))

    def _polyphase_dips(self, frequencies: np.ndarray) -> np.ndarray:
        result = np.ones_like(frequencies)
        for i in range(len(result)):
            if (i - self.polyphase_parameters[0]) % self.polyphase_parameters[1] == 0:
                result[i] = 1 / self.polyphase_parameters[2]
        return result

    @staticmethod
    def _sinus(frequencies: np.ndarray, parameters: tuple[float, float, float]) -> np.ndarray:
        """ Simple wrapper of `np.sin` to work on `frequencies` with `parameters`. """
        phase, amplitude, wavelength = parameters
        return amplitude * np.sin(phase + frequencies * 2 * np.pi / (SPEED_OF_LIGHT / wavelength / MEGA))

    def _sinus_parameter_list(
            self,
            parameters: list[float] | np.ndarray | tuple,
            n_legendre_coefficients: int,
    ) -> list[tuple[float, float, float]]:
        """
        Returns the elements in `parameters` that belong to the sinus function
        by ignoring the first `n_legendre_coefficients` entries.
        """
        sinus_parameter_list = [
            (parameters[n_legendre_coefficients + 2 * i],
             parameters[n_legendre_coefficients + 2 * i + 1],
             w)
            for i, w in enumerate(self.wavelengths)
        ]
        return sinus_parameter_list

    def _sinus_parameter_list_free_wavelengths(
            self,
            parameters: list[float] | np.ndarray | tuple,
            n_legendre_coefficients: int,
    ) -> list[tuple[float, float, float]]:
        """
        Returns the elements in `parameters` that belong to the sinus function
        by ignoring the first `n_legendre_coefficients` entries.
        """
        sinus_parameter_list = [
            (parameters[n_legendre_coefficients + 3 * i],
             parameters[n_legendre_coefficients + 3 * i + 1],
             parameters[n_legendre_coefficients + 3 * i + 2])
            for i, _ in enumerate(self.wavelengths)
        ]
        return sinus_parameter_list

    def _parameters_to_dictionary(
            self,
            parameters: list[float] | np.ndarray,
            n_legendre_coefficients: int,
    ) -> dict[str, float]:
        """
        Turn `parameters` into a dictionary and return the result.
        :param parameters: bandpass model parameters
        :param n_legendre_coefficients: number of legendre coefficients
        :return: parameters dictionary
        """
        self._check_parameters(parameters=parameters,
                               n_legendre_coefficients=n_legendre_coefficients)
        parameter_dict = {f'l_{i}': l for i, l in enumerate(parameters[:n_legendre_coefficients])}
        for i, w in enumerate(self.wavelengths):
            parameter_dict[f'wavelength_{w}_phase'] = parameters[n_legendre_coefficients + 2 * i]
            parameter_dict[f'wavelength_{w}_amplitude'] = parameters[n_legendre_coefficients + 2 * i + 1]
        return parameter_dict

    def _check_parameters(
            self,
            parameters: list[float] | np.ndarray,
            n_legendre_coefficients: int,
    ):
        """
        :raise ValueError: if `len(parameters)` does not match with `self.n_wave` and `n_legendre_coefficients`
        """
        if (must_be_equal := 3 * self.n_wave + n_legendre_coefficients) != len(parameters):
            raise ValueError(f'The number of parameters must be equal to the number of legendre coefficients + twice'
                             f'the number of wavelengths, got {len(parameters)} and {must_be_equal}.')

    def _all_amplitudes_all_wavelengths(self) -> tuple[list[float], list[float]]:
        """ Return a `tuple` of all amplitudes and all corresponding wavelengths. """
        all_wavelengths = []  # type: list[float]
        all_amplitudes = []  # type: list[float]
        for key_, value_ in self.parameters_dictionary.items():
            if 'wavelength' in key_ and '_amplitude' in key_:
                wavelength = float(key_.split('_')[1])
                all_wavelengths.append(wavelength)
                all_amplitudes.append(value_)
        return all_amplitudes, all_wavelengths

    def _maximum_wavelengths(self, n: int) -> list[float]:
        """ Return all wavelengths sorted for amplitudes. """
        all_amplitudes, all_wavelengths = self._all_amplitudes_all_wavelengths()
        sort_args = np.argsort(all_amplitudes)[::-1]
        return np.asarray(all_wavelengths)[sort_args[:n]]

    @staticmethod
    def _double_wavelengths(wavelengths_to_double: list[float]) -> np.ndarray:
        """
        Return a unique `np.ndarray` of all possible additive combinations of the elements in `wavelengths_to_double`.
        Example: `[1, 2]` will become `[1+1, 2+1, 2+2] = [2, 3, 4]`
        """
        top_wavelength_combinations = np.array(np.meshgrid(wavelengths_to_double,
                                                           wavelengths_to_double)).T.reshape(-1, 2)
        wavelengths_to_double = np.unique(np.sum(top_wavelength_combinations, axis=1))
        return wavelengths_to_double

    def _plot(self,
              frequencies: np.ndarray,
              bandpass: np.ndarray,
              model_bandpass: np.ndarray,
              smooth_bandpass: np.ndarray,
              epsilon: np.ndarray,
              receiver_path: str,
              before_or_after: str,
              n_legendre_coeff: int,
              parameters):
        """
        Plot.
        :param frequencies: as array
        :param bandpass: the true bandpass as array
        :param model_bandpass: the model bandpass as array
        :param smooth_bandpass: the true bandpass after wiggle removal as array
        :param epsilon: the wiggles extracted
        :param receiver_path: path to store the plot
        :param before_or_after: calibrator label needed for plot name
        :param n_legendre_coeff: number of legendre polynomials
        :param parameters: bandpass model parameters
        """
        plt.figure(figsize=(16, 24))

        plt.subplot(6, 1, 1)
        plt.plot(frequencies, bandpass, label='bandpass mean')
        plt.plot(frequencies, model_bandpass, ls=':', color='black', label='model')
        plt.xlabel('frequency [MHz]')
        plt.ylabel('bandpass mean')
        plt.legend()

        plt.subplot(6, 1, 2)
        plt.plot(frequencies, bandpass / smooth_bandpass - 1, label='epsilon wiggle data')
        plt.plot(frequencies, epsilon, ls=':', color='black', label='epsilon wiggle model')
        plt.xlabel('frequency [MHz]')
        plt.legend()

        plt.subplot(6, 1, 4)
        residual = (bandpass - model_bandpass) / bandpass * 100
        plt.plot(frequencies, residual, color='black')
        plt.xlabel('frequency [MHz]')
        plt.ylabel('residual [%]')

        plt.subplot(6, 1, 5)
        plt.hist(residual, bins=50, color='black')
        plt.xlabel('residual [%]')
        plt.ylabel('histogram')

        plt.subplot(6, 1, 3)
        sinus_parameters_list = self._sinus_parameter_list_free_wavelengths(
            parameters=parameters,
            n_legendre_coefficients=n_legendre_coeff,
        )
        sinusoidal = [self._sinus(frequencies, sinus_parameters) for sinus_parameters in sinus_parameters_list]
        for sinusoid, p in zip(sinusoidal, sinus_parameters_list):
            plt.plot(frequencies, sinusoid, label=f'ph{p[0]:.2f} amp{p[1]:.2f} wv{p[2] / 2:.1f}')
        sinusoid_sum = np.sum(sinusoidal, axis=0)
        plt.plot(frequencies, sinusoid_sum, lw=2, color='black', label='sum')
        plt.legend()
        plt.xlabel('frequency [MHz]')

        plt.subplot(6, 1, 6)
        fft = np.fft.fft(residual)
        fftfreq = np.fft.fftfreq(len(residual), d=0.2)
        fft = fft[fftfreq >= 0]
        fftfreq = fftfreq[fftfreq >= 0]
        on_x_axis = fftfreq / MEGA * SPEED_OF_LIGHT / 2  # [m]
        on_y_axis = abs(fft) / max(abs(fft))
        plt.semilogy(on_x_axis, on_y_axis, label='residual')

        # epsilon_data = bandpass / smooth_bandpass - 1
        epsilon_data = bandpass
        fft = np.fft.fft(epsilon_data)
        fftfreq = np.fft.fftfreq(len(epsilon_data), d=0.2)
        fft = fft[fftfreq >= 0]
        on_y_axis = abs(fft) / max(abs(fft))
        plt.semilogy(on_x_axis, on_y_axis, label='bandpass data')
        plt.axvline(25, color='black')
        for p in self.standing_wave_displacements:
            plt.axvline(p, color='grey', ls='-')
        for p in sinus_parameters_list:
            plt.axvline(p[2]/2, color='black', ls=':')

        plt.ylabel('fft abs normalised')
        plt.xlabel('[m]')
        plt.xlim((0, 30))
        plt.legend()

        plt.savefig(os.path.join(receiver_path, f'{self.plot_name}_{before_or_after}.png'), dpi=300)
        plt.close()
