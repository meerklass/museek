import unittest

from museek.enums.scan_state_enum import ScanStateEnum
from museek.factory.data_element_factory import ScanElementFactory


class TestScanStateEnum(unittest.TestCase):
    def test_factory(self):
        enum = ScanStateEnum.SCAN
        factory = enum.factory(scan_dumps=[0], component=ScanElementFactory)
        self.assertIsInstance(factory, ScanElementFactory)

    def test_scan_name(self):
        enum = ScanStateEnum.SCAN
        scan_name = enum.scan_name
        self.assertEqual('scan', scan_name)

    def test_get_enum_when_scan(self):
        self.assertEqual(ScanStateEnum.SCAN, ScanStateEnum.get_enum(enum_string='scan'))
