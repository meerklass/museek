import unittest
from unittest.mock import Mock, patch

from museek.receiver import Polarisation, Receiver


class TestReceiver(unittest.TestCase):
    def test_str(self):
        receiver = Receiver(antenna_number=0, polarisation=Polarisation.v)
        self.assertEqual(str(receiver), "m000v")

    def test_eq_when_equal(self):
        self.assertTrue(Receiver.from_string("m000h") == Receiver.from_string("m000h"))

    def test_eq_when_not_equal(self):
        self.assertFalse(Receiver.from_string("m000h") == Receiver.from_string("m000v"))

    def test_name(self):
        receiver = Receiver(antenna_number=0, polarisation=Polarisation.v)
        self.assertEqual(receiver.name, "m000v")

    def test_from_string(self):
        receiver = Receiver.from_string(receiver_string="m000v")
        self.assertEqual(receiver.polarisation, Polarisation.v.name)
        self.assertEqual(receiver.antenna_number, 0)
        self.assertEqual(receiver.antenna_name, "m000")

    def test_from_string_when_invalid_polarisation_expect_raise(self):
        self.assertRaises(ValueError, Receiver.from_string, receiver_string="m000r")

    def test_from_string_when_invalid_antenna_index_expect_raise(self):
        self.assertRaises(ValueError, Receiver.from_string, receiver_string="m1v")

    def test_from_string_when_invalid_receiver_string_expect_raise(self):
        self.assertRaises(ValueError, Receiver.from_string, receiver_string="r001v")

    @patch.object(Receiver, "receivers_to_antennas")
    def test_antenna_index(self, mock_receivers_to_antennas):
        receiver = Receiver.from_string(receiver_string="m000h")
        mock_receivers = Mock()
        self.assertEqual(
            receiver.antenna_index(receivers=mock_receivers),
            mock_receivers_to_antennas.return_value.index.return_value,
        )
        mock_receivers_to_antennas.assert_called_once_with(receivers=mock_receivers)

    def test_receivers_to_antennas(self):
        expect = ["m000", "m001"]
        self.assertListEqual(
            Receiver.receivers_to_antennas(
                receivers=[
                    Receiver.from_string(receiver_string="m000h"),
                    Receiver.from_string(receiver_string="m000v"),
                    Receiver.from_string(receiver_string="m001h"),
                    Receiver.from_string(receiver_string="m001v"),
                ]
            ),
            expect,
        )


if __name__ == "__main__":
    unittest.main()
