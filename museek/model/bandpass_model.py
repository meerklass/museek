import os
from typing import Callable, Optional
import emcee

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy.polynomial import legendre

from definitions import MEGA, SPEED_OF_LIGHT
from museek.data_element import DataElement
import corner


class BandpassModel:
    """ Class to model the bandpass shape including standing waves. """

    def __init__(self,
                 standing_wave_displacements: list[float],
                 polyphase_parameters: tuple[int, int, float],
                 plot_name: str | None = None):
        """
        Initialise
        :param standing_wave_displacements: list of `float` standing wave displacement distances [m]
        :param polyphase_parameters: `tuple` of first dip dump index and dip separation (usually 64)
                                      and neighbors/dip ratio
        :param plot_name: optional plot name, not created if `None`
        """
        self.plot_name = plot_name
        self.standing_wave_displacements = standing_wave_displacements
        self.wavelengths = [d * 2 for d in self.standing_wave_displacements]
        self.n_wave = len(standing_wave_displacements)
        self.polyphase_parameters = polyphase_parameters

        # function to encapsulate the wiggly structure of the bandpass
        self.bandpass_model_result = None  # type: Optional[Callable]
        # function to encapsulate the flat, or legendre-like structure of the bandpass
        self.variances_dictionary = None  # type: Optional[dict]
        self.parameters_dictionary = None  # type: Optional[dict]
        self.epsilon = None  # type: Optional[np.ndarray]

        self._curve_fit: list | None = None

        self.bandpass_model_wrapper: Callable | None = None

        sinus_upper_amplitude = 100  # original value, works best, but not physical
        wavelength_error = 0.1  # meters
        wavelength_lower_bounds = [w - wavelength_error for w in self.wavelengths]
        wavelength_upper_bounds = [w + wavelength_error for w in self.wavelengths]

        sinus_lower_bounds = []
        sinus_upper_bounds = []
        for upper, lower in zip(wavelength_upper_bounds, wavelength_lower_bounds):
            sinus_lower_bounds += [-np.pi, 0, lower]
            sinus_upper_bounds += [np.pi, sinus_upper_amplitude, upper]

        self.lower_bounds = np.asarray(sinus_lower_bounds)
        self.upper_bounds = np.asarray(sinus_upper_bounds)
        self.priors = [(l, u) for l, u in zip(self.lower_bounds, self.upper_bounds)]

        parameter_names = []
        for wave in self.standing_wave_displacements:
            parameter_names += ['phase', 'amp', f'wl {wave} m']
        self.parameter_names = parameter_names

        self.legendre_coefficients = None

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
            estimator_error: DataElement,
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
        target_estimator_error = estimator_error.squeeze
        total_n_wavelengths = len(self.wavelengths)

        if starting_coefficients is None:
            sinus_start_amplitude = 0.001  # recommended value
            starting_coefficients = []
            for i in range(total_n_wavelengths):
                starting_coefficients += [0, sinus_start_amplitude, self.wavelengths[i]]

        def bandpass_model_wrapper(f: np.ndarray, *parameters) -> np.ndarray:
            """ Wrap the bandpass model for the scipy fit. """
            if isinstance(parameters, tuple):
                parameters = parameters[0]
            sinus_coefficient_list = self._sinus_parameter_list_free_wavelengths(
                parameters=parameters,
            )
            return self._bandpass_model(frequencies=f,
                                        sinus_parameters_list=sinus_coefficient_list)

        self.bandpass_model_wrapper = bandpass_model_wrapper

        # bounds = (self.lower_bounds, self.upper_bounds)
        # curve_fit = self._fit(executable=bandpass_model_wrapper,
        #                       frequencies=target_frequencies,
        #                       estimator=target_estimator,
        #                       p0=starting_coefficients,
        #                       bounds=bounds)
        # best_fit = curve_fit[0]
        # variances = curve_fit[1]

            sampler = self._fit_mcmc(frequencies=target_frequencies,
                                    estimator=target_estimator,
                                    estimator_error=target_estimator_error,
                                    p0=starting_coefficients)
            samples = sampler.get_chain()
            ndim = len(starting_coefficients)
            thin = 1
            # discard = 8000
            discard = 40000
            # discard = 50
            _, axes = plt.subplots(ndim, figsize=(10, 50), sharex=True)
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                # ax.set_xlim(0, len(samples))
                ax.set_ylabel(self.parameter_names[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
                ax.axvline(discard)

            axes[-1].set_xlabel("step number")
            plt.savefig(os.path.join(receiver_path, f'{self.plot_name}_mcmc_walkers.png'))
            plt.close()
            
            flat_samples = sampler.get_chain(discard=discard, flat=True, thin=thin)
            log_probability = sampler.get_log_prob(discard=discard, flat=True, thin=thin)

            np.savez(os.path.join(receiver_path, 'samples.npz'),
                     flat_samples=flat_samples,
                    log_probability=log_probability)




        # histograms_bins = [np.histogram(flat, bins=100) for flat in flat_samples.T]
        # half_bins = [(hb[1][1:] + hb[1][:-1])/2 for hb in histograms_bins]
        # best_fit_h = [half_bins[i][np.argmax(histograms_bins[i][0])] for i in range(len(histograms_bins))]

        best_fit_index = np.argmax(log_probability)
        best_fit = flat_samples[best_fit_index]

        variances = np.std(flat_samples, axis=0)**2
        curve_fit = (best_fit, variances)

        fig = corner.corner(flat_samples, labels=self.parameter_names, truths=best_fit)
        plt.savefig(os.path.join(receiver_path, f'{self.plot_name}_mcmc_corner.png'))
        plt.close()

        plt.plot(target_frequencies, target_estimator)
        for i in range(0, flat_samples.shape[0], 100000):
            best_fit = flat_samples[i]
            model_bandpass = bandpass_model_wrapper(target_frequencies, best_fit)
            plt.plot(target_frequencies, model_bandpass, ls = ':')
        plt.savefig(os.path.join(receiver_path, f'test.png'))
        plt.close()


        model_bandpass = bandpass_model_wrapper(target_frequencies, best_fit)
        epsilon = model_bandpass - 1

        if self.plot_name is not None:
            self._plot(frequencies=target_frequencies,
                       bandpass=target_estimator,
                       model_bandpass=model_bandpass,
                       epsilon=epsilon,
                       receiver_path=receiver_path,
                       before_or_after=calibrator_label,
                       parameters=best_fit)

        def bandpass_model_result(f: DataElement) -> np.ndarray:
            """ Return the sinusoidal contribution to the bandpass as a function of frequency. """
            return bandpass_model_wrapper(f.squeeze / MEGA, best_fit)

        self.bandpass_model_result = bandpass_model_result
        self.epsilon = epsilon
        self._curve_fit = curve_fit

    @staticmethod
    def _fit(executable, frequencies, estimator, p0, bounds):
        return scipy.optimize.curve_fit(executable,
                                        frequencies,
                                        estimator,
                                        p0=p0,
                                        bounds=bounds,
                                        method='dogbox')

    def _fit_mcmc(self, frequencies, estimator, estimator_error, p0):
        nwalkers = 100
        ndim = len(p0)
        n_steps = 100000
        pos = p0 + 1e-3 * abs(np.random.randn(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        self.log_probability,
                                        args=(frequencies, estimator, estimator_error))
        sampler.run_mcmc(pos, n_steps, progress=True)
        return sampler

    def log_likelihood(self, theta, frequencies, estimator, estimator_error):
        if self.bandpass_model_wrapper is None:
            raise ValueError('`self.bandpass_model_wrapper` must not be `None` at this stage...')
        model = self.bandpass_model_wrapper(frequencies, theta)
        sigma2 = estimator_error**2
        return -0.5 * np.sum((estimator - model) ** 2 / sigma2)

    def log_prior(self, theta):
        for t, p in zip(theta, self.priors):
            if p[0] > t or p[1] < t:  # lower prior bigger/equal to param or upper prior smaller/equal
                return -np.inf
        return 0

    def log_probability(self, theta, frequencies, estimator, estimator_error):
        log_prior = self.log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf
        return log_prior + self.log_likelihood(theta, frequencies, estimator, estimator_error)

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
            sinus_parameters_list: list[tuple[float, float, float]],
    ) -> np.ndarray:
        """
        Bandpass model based on a discrete set of standing waves
        :param frequencies: frequencies as array
        :param legendre_coefficients: the coefficients of the legendre polynomial
        :param sinus_parameters_list: `list` of the standing wave parameters as `tuple`s
        :return: the bandpass model as a `numpy` array
        """
        sinusoidal = [self._sinus(frequencies, sinus_parameters) for sinus_parameters in sinus_parameters_list]
        polyphase_dips = self._polyphase_dips(frequencies=frequencies)
        return polyphase_dips * (1 + sum(sinusoidal))

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
    ) -> list[tuple[float, float, float]]:
        """
        Returns the elements in `parameters` that belong to the sinus function
        by ignoring the first `n_legendre_coefficients` entries.
        """
        sinus_parameter_list = [
            (parameters[3 * i],
             parameters[3 * i + 1],
             parameters[3 * i + 2])
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
              epsilon: np.ndarray,
              receiver_path: str,
              before_or_after: str,
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
        plt.figure(figsize=(16, 30))

        plt.subplot(7, 1, 1)
        plt.plot(frequencies, bandpass, label='bandpass mean')
        plt.plot(frequencies, model_bandpass, ls=':', color='black', label='model')
        plt.xlabel('frequency [MHz]')
        plt.ylabel('bandpass mean')
        plt.legend()

        plt.subplot(7, 1, 2)
        plt.plot(frequencies, epsilon, ls=':', color='black', label='epsilon wiggle model')
        plt.xlabel('frequency [MHz]')
        plt.legend()

        plt.subplot(7, 1, 4)
        residual = (bandpass - model_bandpass) / bandpass * 100
        plt.plot(frequencies, residual, color='black')
        plt.xlabel('frequency [MHz]')
        plt.ylabel('residual [%]')

        plt.subplot(7, 1, 5)
        plt.hist(residual, bins=50, color='black')
        plt.xlabel('residual [%]')
        plt.ylabel('histogram')

        plt.subplot(7, 1, 3)
        sinus_parameters_list = self._sinus_parameter_list_free_wavelengths(
            parameters=parameters,
        )
        sinusoidal = [self._sinus(frequencies, sinus_parameters) for sinus_parameters in sinus_parameters_list]
        for sinusoid, p in zip(sinusoidal, sinus_parameters_list):
            plt.plot(frequencies, sinusoid, label=f'ph{p[0]:.2f} amp{p[1]:.2f} wv{p[2] / 2:.1f}')
        sinusoid_sum = np.sum(sinusoidal, axis=0)
        plt.plot(frequencies, sinusoid_sum, lw=2, color='black', label='sum')
        plt.legend()
        plt.xlabel('frequency [MHz]')

        plt.subplot(7, 1, 6)
        for_fft_1 = (residual/np.amax(residual) - 1) * np.hanning(len(residual))

        epsilon_data = bandpass - 1  # TODO: one minus one too many??
        for_fft_2 = (epsilon_data / np.amax(epsilon_data) - 1) * np.hanning(len(epsilon_data))

        plt.plot(frequencies, for_fft_1, label='residual * hanning')
        plt.plot(frequencies, for_fft_2, label='epsilon_data * hanning')
        plt.xlabel('frequency [MHz]')
        plt.legend()

        plt.subplot(7, 1, 7)
        fft = np.fft.fft(for_fft_1)
        fftfreq = np.fft.fftfreq(len(residual), d=0.2)  # TODO: double check
        fft = fft[fftfreq >= 0]
        fftfreq = fftfreq[fftfreq >= 0]
        on_x_axis = fftfreq / MEGA * SPEED_OF_LIGHT / 2  # [m]  # TODO: isn't the MEGA the wrong way??
        on_y_axis = abs(fft) / max(abs(fft))
        plt.semilogy(on_x_axis, on_y_axis, label='residual')

        fft = np.fft.fft(for_fft_2)
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
        plt.xlim((0, 60))
        plt.legend()
        plt.savefig(os.path.join(receiver_path, f'{self.plot_name}_{before_or_after}.png'), dpi=300)
        plt.close()
