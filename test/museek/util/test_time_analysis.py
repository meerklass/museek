import unittest
from datetime import datetime

from museek.util.time_analysis import TimeAnalysis


class TestTimeAnalysis(unittest.TestCase):
    def test_time_difference_to_sunset_sunrise(self):
        time_analysis = TimeAnalysis(latitude=0, longitude=0)
        obs_start = datetime.fromtimestamp(0)
        obs_end = datetime.fromtimestamp(3600)  # 1 hour later
        start, end, end_diff, start_diff = (
            time_analysis.time_difference_to_sunset_sunrise(
                obs_start=obs_start, obs_end=obs_end
            )
        )

        expected_start = datetime(1969, 12, 31, 18, 7, 1, 543037)
        expected_end = datetime(1970, 1, 1, 5, 59, 31, 249269)
        # Allow small differences due to platform/ephemeris microsecond rounding
        self.assertAlmostEqual((start - expected_start).total_seconds(), 0, places=3)
        self.assertAlmostEqual((end - expected_end).total_seconds(), 0, places=3)
        self.assertAlmostEqual(10771, end_diff, 0)
        self.assertAlmostEqual(28378, start_diff, 0)


if __name__ == "__main__":
    unittest.main()
