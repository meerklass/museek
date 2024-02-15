import unittest
from museek.util.time_analysis import TimeAnalysis
from datetime import datetime


class TestTimeAnalysis(unittest.TestCase):
    def test_time_difference_to_sunset_sunrise(self):
        time_analysis = TimeAnalysis(latitude=0, longitude=0)
        obs_start = datetime.fromtimestamp(0)
        obs_end = datetime.fromtimestamp(3600)  # 1 hour later
        start, end, end_diff, start_diff = time_analysis.time_difference_to_sunset_sunrise(obs_start=obs_start, 
                                                                                           obs_end=obs_end)

        self.assertEqual(datetime(1969, 12, 31, 18, 7, 1, 543037), start)
        self.assertEqual(datetime(1970, 1, 1, 5, 59, 31, 249269), end)
        self.assertAlmostEqual(10771, end_diff, 0)
        self.assertAlmostEqual(28378, start_diff, 0)


if __name__ == '__main__':
    unittest.main()
