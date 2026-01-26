import os

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy.polynomial import legendre

from museek.data_element import DataElement
from museek.definitions import MEGA, SPEED_OF_LIGHT


class BandpassModel:
    """Class to model the bandpass shape including standing waves."""

    def __init__(
        self,
        standing_wave_displacements: list[float],
        legendre_degree: int,
        plot_name: str | None = None,
    ):
        """
        Initialise
        :param standing_wave_displacements: list of `float` standing wave displacement distances [m]
        :param legendre_degree: degree of the legendre polynomial
        :param plot_name: optional plot name, not created if `None`
        """
        self.plot_name = plot_name
        self.standing_wave_displacements = standing_wave_displacements
        self.wavelengths = [d * 2 for d in self.standing_wave_displacements]
        self.n_wave = len(standing_wave_displacements)
        self.legendre_degree = legendre_degree

        # function to encapsulate the wiggly structure of the bandpass
        self.epsilon_function = None  # type: Optional[Callable]
        # function to encapsulate the flat, or legendre-like structure of the bandpass
        self.legendre_function = None  # type: Optional[Callable]
        self.variances_dictionary = None  # type: Optional[dict]
        self.parameters_dictionary = None  # type: Optional[dict]
        self.epsilon = None  # type: Optional[np.ndarray]

    def fit(
        self,
        frequencies: DataElement,
        estimator: DataElement,
        receiver_path: str,
        calibrator_label: str,
    ):
        """
        Fit the standing wave model to a single receiver.
        :param frequencies: the frequencies as a `DataElement`
        :param estimator: the bandpass itself or an estimator thereof as `DataElement`
        :param receiver_path: path to store results specific to this receiver
        :param calibrator_label: usually 'before_scan' or 'after_scan' or similar
        """
        target_frequencies = frequencies.squeeze / MEGA
        target_estimator = estimator.squeeze

        starting_legendre_coefficients = [
            x
            for x in legendre.legfit(
                target_frequencies, target_estimator, self.legendre_degree
            )
        ]
        starting_coefficients = starting_legendre_coefficients + [
            0.1 * (i % 2) for i in range(len(self.wavelengths) * 2)
        ]

        def bandpass_model_wrapper(f: np.ndarray, *parameters) -> np.ndarray:
            """Wrap the bandpass model for the scipy fit."""
            sinus_coefficient_list = self._sinus_parameter_list(
                parameters=parameters,
                n_legendre_coefficients=len(starting_legendre_coefficients),
            )
            return self._bandpass_model(
                frequencies=f,
                legendre_coefficients=parameters[: len(starting_legendre_coefficients)],
                sinus_parameters_list=sinus_coefficient_list,
            )

        lower_bounds = np.asarray(
            [-np.inf for _ in starting_legendre_coefficients]
            + [-np.pi, 0] * self.n_wave
        )
        upper_bounds = np.asarray(
            [np.inf for _ in starting_legendre_coefficients] + [np.pi, 1] * self.n_wave
        )
        bounds = (lower_bounds, upper_bounds)

        curve_fit = scipy.optimize.curve_fit(
            bandpass_model_wrapper,
            target_frequencies,
            target_estimator,
            p0=starting_coefficients,
            bounds=bounds,
        )

        model_bandpass = bandpass_model_wrapper(target_frequencies, *curve_fit[0])
        legendre_bandpass = legendre.legval(
            target_frequencies, curve_fit[0][: len(starting_legendre_coefficients)]
        )
        epsilon = model_bandpass / legendre_bandpass - 1

        parameters_dict = self._parameters_to_dictionary(
            curve_fit[0], n_legendre_coefficients=len(starting_legendre_coefficients)
        )
        variances_dict = self._parameters_to_dictionary(
            np.diag(curve_fit[1]),
            n_legendre_coefficients=len(starting_legendre_coefficients),
        )

        if self.plot_name is not None:
            self._plot(
                frequencies=target_frequencies,
                bandpass=target_estimator,
                model_bandpass=model_bandpass,
                smooth_bandpass=legendre_bandpass,
                epsilon=epsilon,
                receiver_path=receiver_path,
                before_or_after=calibrator_label,
            )

        def legendre_function(f: DataElement) -> np.ndarray:
            """Return the legendre contribution to the bandpass as a function of frequency."""
            return legendre.legval(
                f.squeeze / MEGA, curve_fit[0][: len(starting_legendre_coefficients)]
            )

        def epsilon_function(f: DataElement) -> np.ndarray:
            """Return the sinusoidal contribution to the bandpass as a function of frequency."""
            return (
                bandpass_model_wrapper(f.squeeze / MEGA, *curve_fit[0])
                / legendre_function(f)
                - 1
            )

        self.epsilon_function = epsilon_function
        self.legendre_function = legendre_function
        self.variances_dictionary = variances_dict
        self.parameters_dictionary = parameters_dict
        self.epsilon = epsilon

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
        sinusoidal = [
            self._sinus(frequencies, sinus_parameters)
            for sinus_parameters in sinus_parameters_list
        ]
        return legendre_ * (1 + sum(sinusoidal))

    @staticmethod
    def _sinus(
        frequencies: np.ndarray, parameters: tuple[float, float, float]
    ) -> np.ndarray:
        """Simple wrapper of `np.sin` to work on `frequencies` with `parameters`."""
        phase, amplitude, wavelength = parameters
        return amplitude * np.sin(
            phase + frequencies * 2 * np.pi / (SPEED_OF_LIGHT / wavelength / MEGA)
        )

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
            (
                parameters[n_legendre_coefficients + 2 * i],
                parameters[n_legendre_coefficients + 2 * i + 1],
                w,
            )
            for i, w in enumerate(self.wavelengths)
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
        self._check_parameters(
            parameters=parameters, n_legendre_coefficients=n_legendre_coefficients
        )
        parameter_dict = {
            f"l_{i}": leg_coeff
            for i, leg_coeff in enumerate(parameters[:n_legendre_coefficients])
        }
        for i, w in enumerate(self.wavelengths):
            parameter_dict[f"wavelength_{w}_phase"] = parameters[
                n_legendre_coefficients + 2 * i
            ]
            parameter_dict[f"wavelength_{w}_amplitude"] = parameters[
                n_legendre_coefficients + 2 * i + 1
            ]
        return parameter_dict

    def _check_parameters(
        self,
        parameters: list[float] | np.ndarray,
        n_legendre_coefficients: int,
    ):
        """
        :raise ValueError: if `len(parameters)` does not match with `self.n_wave` and `n_legendre_coefficients`
        """
        if (must_be_equal := 2 * self.n_wave + n_legendre_coefficients) != len(
            parameters
        ):
            raise ValueError(
                f"The number of parameters must be equal to the number of legendre coefficients + twice"
                f"the number of wavelengths, got {len(parameters)} and {must_be_equal}."
            )

    def _plot(
        self,
        frequencies: np.ndarray,
        bandpass: np.ndarray,
        model_bandpass: np.ndarray,
        smooth_bandpass: np.ndarray,
        epsilon: np.ndarray,
        receiver_path: str,
        before_or_after: str,
    ):
        """
        Plot.
        :param frequencies: as array
        :param bandpass: the true bandpass as array
        :param model_bandpass: the model bandpass as array
        :param smooth_bandpass: the true bandpass after wiggle removal as array
        :param epsilon: the wiggles extracted
        :param receiver_path: path to store the plot
        :param before_or_after: calibrator label needed for plot name
        """
        plt.figure(figsize=(16, 24))

        plt.subplot(4, 1, 1)
        plt.plot(frequencies, bandpass, label="bandpass mean")
        plt.plot(frequencies, model_bandpass, ls=":", color="black", label="model")
        plt.plot(frequencies, smooth_bandpass, label="smooth model")
        plt.xlabel("frequency [MHz]")
        plt.ylabel("bandpass mean")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(
            frequencies, epsilon, ls=":", color="black", label="epsilon wiggle model"
        )
        plt.plot(
            frequencies, bandpass / smooth_bandpass - 1, label="epsilon wiggle data"
        )
        plt.xlabel("frequency [MHz]")
        plt.legend()

        plt.subplot(4, 1, 3)
        residual = (bandpass - model_bandpass) / bandpass * 100
        plt.plot(frequencies, residual, color="black")
        plt.xlabel("frequency [MHz]")
        plt.ylabel("residual [%]")

        plt.subplot(4, 1, 4)
        plt.hist(residual, bins=50, color="black")
        plt.xlabel("residual [%]")
        plt.ylabel("histogram")
        plt.savefig(
            os.path.join(receiver_path, f"{self.plot_name}_{before_or_after}.png")
        )
        plt.close()
