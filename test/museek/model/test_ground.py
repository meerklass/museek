import unittest

import numpy as np
from scikits.fitting import Spline2DGridFit

from museek.data_element import DataElement
from museek.factory.data_element_factory import DataElementFactory
from museek.model.ground import Ground
from museek.receiver import Receiver


class TestGround(unittest.TestCase):

    def setUp(self):
        self.ground = Ground(data_element_factory=DataElementFactory())

        mock_elevation_array = np.linspace(52, 57, 10)[:, np.newaxis, np.newaxis]
        mock_frequency_array = np.linspace(856000000, 1711791016, 20)[np.newaxis, :, np.newaxis]
        self.mock_elevation = DataElementFactory().create(array=mock_elevation_array)
        self.mock_frequency = DataElementFactory().create(array=mock_frequency_array)

    def test_temperature_when_polarisation_h(self):
        receiver = Receiver.from_string('m000h')

        temperature = self.ground.temperature(receiver=receiver,
                                              elevation=self.mock_elevation,
                                              frequency=self.mock_frequency)
        self.assertTupleEqual((10, 20, 1), temperature.shape)
        self.assertIsInstance(temperature, DataElement)
        self.assertLess(temperature.squeeze.max(), 4)
        self.assertGreater(temperature.squeeze.min(), 1)

    def test_temperature_when_polarisation_v(self):
        receiver = Receiver.from_string('m000v')

        temperature = self.ground.temperature(receiver=receiver,
                                              elevation=self.mock_elevation,
                                              frequency=self.mock_frequency)
        self.assertTupleEqual((10, 20, 1), temperature.shape)
        self.assertIsInstance(temperature, DataElement)
        self.assertLess(temperature.squeeze.max(), 4)
        self.assertGreater(temperature.squeeze.min(), 1)

    def test_polarisation_temperature_fits(self):
        h_fit, v_fit = self.ground.polarisation_temperature_fits()
        self.assertIsInstance(h_fit, Spline2DGridFit)
        self.assertIsInstance(v_fit, Spline2DGridFit)

    def test_load_data(self):
        h_data, v_data, elevation, frequencies = self.ground._load_data()
        self.assertLessEqual(max(elevation), 90)
        self.assertGreater(min(elevation), 0)
        self.assertGreaterEqual(min(frequencies), 900)
        self.assertLessEqual(max(frequencies), 1700)
        self.assertIsInstance(h_data, np.ndarray)
        self.assertIsInstance(v_data, np.ndarray)


if __name__ == '__main__':
    unittest.main()
