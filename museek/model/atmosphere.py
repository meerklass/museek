import numpy as np
from katpoint import Antenna

from museek.data_element import DataElement
from museek.factory.data_element_factory import DataElementFactory
from museek.model.abstract_model import AbstractModel
from museek.model.functions import atmospheric_opacity
from museek.receiver import Receiver


class Atmosphere(AbstractModel):
    """ Class to model the atmosphere temperature. """

    def __init__(self,
                 data_element_factory: DataElementFactory,
                 temperature: DataElement,
                 pressure: DataElement,
                 humidity: DataElement,
                 antennas: list[Antenna],
                 receivers: list[Receiver]):
        """
        Initialise
        :param data_element_factory:
        :param temperature: time ordered temperature data in degree celsius
        :param pressure: time ordered pressure data in hPa
        :param humidity: time ordered relative humidity data
        :param antennas: `list` of all `Antenna`s
        :param receivers: `list of all `Receiver`s
        """
        super().__init__(data_element_factory=data_element_factory)
        self.ground_temperature = temperature
        self.pressure = pressure
        self.humidity = humidity
        self.receivers = receivers
        self.heights = self.get_height_of_antennas(antennas=antennas)

    def temperature(self, receiver: Receiver, elevation: DataElement, frequency: DataElement):
        """
        Returns the temperature model as a `DataElement`.
        :param receiver: the height above sea level of the antenna is relevant
        :param elevation: must only contain one non-empty dimension, the time axis
        :param frequency: must only contain one non-empty dimension, the frequency axis
        :return: a `DataElement` with shape `(n_dump, n_freq, 1)` containing the temperature model
        """
        n_dump = elevation.shape[0]
        n_freq = frequency.shape[1]
        n_recv = 1
        height = self.heights[receiver.antenna_index(receivers=self.receivers)]
        to_shape = (n_dump, n_freq, n_recv)
        atmosphere_temperature = self.data_element_factory.create(
            array=1.12 * (273.15 + self.ground_temperature.fill(to_shape=to_shape).array) - 50.0
        )  # TODO: confirm this equation
        air_relative_humidity = self.humidity / 100.  # this is a percentage in katdal

        atmospheric_opacity_ = atmospheric_opacity(
            temperature=self.ground_temperature.fill(to_shape=to_shape).array,
            relative_humidity=air_relative_humidity.fill(to_shape=to_shape).array,
            pressure=self.pressure.fill(to_shape=to_shape).array,
            height=height,
            frequency=frequency.fill(to_shape=to_shape).array
        )
        atmosphere_array = atmosphere_temperature.fill(to_shape=to_shape) * (
                1 - np.exp(-atmospheric_opacity_ / np.sin(np.radians(elevation.fill(to_shape=to_shape).array)))
        )

        return self.data_element_factory.create(array=atmosphere_array)

    @staticmethod
    def get_height_of_antennas(antennas: list[Antenna]) -> list[float]:
        """ Helper method to return the height above sea level in meters of the `Antenna`s in `antennas`. """
        return [antenna.observer.elevation for antenna in antennas]
