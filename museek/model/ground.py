import os
from typing import Tuple

import numpy as np
import pandas
from scikits import fitting
from scikits.fitting import Spline2DGridFit

from definitions import ROOT_DIR
from museek.data_element import DataElement
from museek.factory.data_element_factory import DataElementFactory
from museek.model.abstract_model import AbstractModel
from museek.receiver import Receiver

MEGA = 1e6


class Ground(AbstractModel):
    """ Class to model the temperature of the ground spill. """

    def __init__(self, data_element_factory: DataElementFactory, file_name: str = 'MK_L_Tspill_AsBuilt_atm_mask.dat'):
        """
        Initialise with a `data_element_factory` and `file_name`.
        :param data_element_factory: to create `DataElement`s
        :param file_name: file name for the ground spill in `data`
        """
        super().__init__(data_element_factory=data_element_factory)
        self._file_name = file_name

    def temperature(self, receiver: Receiver, elevation: DataElement, frequency: DataElement):
        """
        Returns the temperature model as a `DataElement`.
        :param receiver: the polarisation is relevant
        :param elevation: must only contain one non-empty dimension, the time axis
        :param frequency: must only contain one non-empty dimension, the frequency axis
        :return: a `DataElement` with shape `(n_dump, n_freq, 1)` containing the temperature model
        """
        fit_h, fit_v = self.polarisation_temperature_fits()
        fit_dict = {
            'h': fit_h,
            'v': fit_v
        }
        # this will evaluate the fit on the grid defined by `elevation` and `frequency`
        array = fit_dict[receiver.polarisation]((elevation.squeeze, frequency.squeeze / MEGA))
        return self.data_element_factory.create(array=array[:, :, np.newaxis])

    def polarisation_temperature_fits(self) -> Tuple[Spline2DGridFit, Spline2DGridFit]:
        """ Return a `tuple` of fit functions for the h and v polarisation ground spill temperature. """
        h_data, v_data, elevation, frequencies = self._load_data()
        temperature_h = fitting.Spline2DGridFit(degree=(3, 3))
        temperature_h.fit((elevation, frequencies), h_data)

        temperature_v = fitting.Spline2DGridFit(degree=(3, 3))
        temperature_v.fit((elevation, frequencies), v_data)

        return temperature_h, temperature_v

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the data text file in `self._file_name` and extracts and returns its
        contents assuming a very specific format.
        :return: `tuple` of the polarisation temperature data in h and v polarisations and the datapoint
                 elevation and frequency values
        """
        file_name = os.path.join(ROOT_DIR, 'data', self._file_name)
        data_frame = pandas.read_csv(file_name,
                                     sep=" ",
                                     header=0,
                                     engine='python',
                                     comment='#',
                                     skipinitialspace=True)
        elevation = 90. - data_frame['0'].to_numpy()
        string_frequencies = [key for key in data_frame.columns if key != '0' and not key.endswith('.1')]
        frequencies = np.asarray([float(frequency) for frequency in string_frequencies])
        h_data = np.asarray([data_frame[key].to_numpy() for key in string_frequencies]).T
        v_data = np.asarray([data_frame[key + '.1'].to_numpy() for key in string_frequencies]).T
        return h_data, v_data, elevation, frequencies
