import unittest

from museek.receiver import Receiver, Polarisation


class TestReceiver(unittest.TestCase):

    def test_str(self):
        receiver = Receiver(antenna_number=0, polarisation=Polarisation.v)
        self.assertEqual(str(receiver), 'm000v')

    def test_name(self):
        receiver = Receiver(antenna_number=0, polarisation=Polarisation.v)
        self.assertEqual(receiver.name, 'm000v')

    def test_from_string(self):
        receiver = Receiver.from_string(receiver_string='m000v')
        self.assertEqual(receiver.polarisation, Polarisation.v.name)
        self.assertEqual(receiver.antenna_number, 0)
        self.assertEqual(receiver.antenna_name, 'm000')

    def test_from_string_when_invalid_polarisation_expect_raise(self):
        self.assertRaises(ValueError, Receiver.from_string, receiver_string='m000r')

    def test_from_string_when_invalid_antenna_index_expect_raise(self):
        self.assertRaises(ValueError, Receiver.from_string, receiver_string='m1v')

    def test_from_string_when_invalid_receiver_string_expect_raise(self):
        self.assertRaises(ValueError, Receiver.from_string, receiver_string='r001v')


if __name__ == '__main__':
    unittest.main()
