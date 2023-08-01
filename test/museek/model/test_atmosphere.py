import unittest

import numpy as np
from katpoint import Antenna

from museek.data_element import DataElement
from museek.factory.data_element_factory import DataElementFactory
from museek.model.atmosphere import Atmosphere
from museek.receiver import Receiver


class TestAtmosphere(unittest.TestCase):
    def setUp(self):
        n_dump = 30  # number of time dumps
        mock_temperature_array = np.ones((n_dump, 1, 1)) * 20  # typical temperature
        mock_pressure_array = np.ones((n_dump, 1, 1)) * 1000  # typical pressure
        mock_humidity_array = np.ones((n_dump, 1, 1)) * 25  # typical humidity
        mock_temperature = DataElementFactory().create(array=mock_temperature_array)
        mock_pressure = DataElementFactory().create(array=mock_pressure_array)
        mock_humidity = DataElementFactory().create(array=mock_humidity_array)
        mock_antenna_0 = Antenna(name='m000',
                                 latitude=0,
                                 longitude=0,
                                 altitude=1000)
        mock_antenna_1 = Antenna(name='m001',
                                 latitude=0,
                                 longitude=0,
                                 altitude=1001)
        mock_receiver_1 = Receiver.from_string('m000h')
        mock_receiver_2 = Receiver.from_string('m000v')
        mock_receiver_3 = Receiver.from_string('m001h')
        mock_receiver_4 = Receiver.from_string('m001v')

        self.mock_antennas = [mock_antenna_0, mock_antenna_1]
        self.mock_receivers = [mock_receiver_1, mock_receiver_2, mock_receiver_3, mock_receiver_4]
        self.atmosphere = Atmosphere(data_element_factory=DataElementFactory(),
                                     temperature=mock_temperature,
                                     pressure=mock_pressure,
                                     humidity=mock_humidity,
                                     antennas=self.mock_antennas,
                                     receivers=self.mock_receivers)

        mock_elevation_array = np.linspace(52, 57, n_dump)[:, np.newaxis, np.newaxis]
        mock_frequencies_array = np.linspace(856000000, 1711791016, 20)[np.newaxis, :, np.newaxis]
        self.mock_elevation = DataElementFactory().create(array=mock_elevation_array)
        self.mock_frequencies = DataElementFactory().create(array=mock_frequencies_array)

    def test_temperature(self):
        receiver = Receiver.from_string('m000h')

        temperature = self.atmosphere.temperature(receiver=receiver,
                                                  elevation=self.mock_elevation,
                                                  frequencies=self.mock_frequencies)
        self.assertTupleEqual((30, 20, 1), temperature.shape)
        self.assertIsInstance(temperature, DataElement)

    def test_temperature_expect_receivers_different(self):
        temperatures = np.asarray([self.atmosphere.temperature(receiver=receiver,
                                                               elevation=self.mock_elevation,
                                                               frequencies=self.mock_frequencies)
                                   for receiver in self.mock_receivers])
        for i, temp in enumerate(temperatures):
            is_equal = temperatures == temp
            if i in [0, 1]:
                np.testing.assert_array_equal((True, True, False, False), is_equal)
            else:
                np.testing.assert_array_equal((False, False, True, True), is_equal)

    def test_get_height_of_antennas(self):
        heights = self.atmosphere.get_height_of_antennas(antennas=self.mock_antennas)
        self.assertListEqual([1000, 1001], heights)
