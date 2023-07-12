import unittest
from unittest.mock import Mock

from definitions import GIGA
from museek.model.functions import atmospheric_opacity


class TestFunctions(unittest.TestCase):

    def test_atmospheric_opacity_when_frequency_too_high_expect_raise(self):
        self.assertRaises(ValueError,
                          atmospheric_opacity,
                          temperature=Mock(),
                          relative_humidity=Mock(),
                          height=Mock(),
                          pressure=Mock(),
                          frequency=56 * GIGA)

    def test_atmospheric_opacity(self):  # TODO: add meaningful test here (compare to katcali?)
        pass
