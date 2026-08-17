import unittest
from datetime import datetime, timezone

from museek.util.time_analysis import TimeAnalysis


class TestTimeAnalysis(unittest.TestCase):
    def test_time_difference_to_sunset_sunrise(self):
        time_analysis = TimeAnalysis(latitude=0, longitude=0)
        # Use UTC timestamps to make the test deterministic across runner timezones
        obs_start = datetime.fromtimestamp(0, timezone.utc)
        obs_end = datetime.fromtimestamp(3600, timezone.utc)  # 1 hour later
        start, end, end_diff, start_diff = (
            time_analysis.time_difference_to_sunset_sunrise(
                obs_start=obs_start, obs_end=obs_end
            )
        )

        # Expected values for UTC inputs
        expected_start = datetime(1969, 12, 31, 18, 7, 1, 542154)
        expected_end = datetime(1970, 1, 1, 5, 59, 31, 248386)
        # Allow small differences due to platform/ephemeris microsecond rounding
        self.assertAlmostEqual((start - expected_start).total_seconds(), 0, delta=1e-3)
        self.assertAlmostEqual((end - expected_end).total_seconds(), 0, delta=1e-3)
        # Compare to nearest second to avoid fractional-second ephemeris differences
        self.assertAlmostEqual(17971, end_diff, 0)
        self.assertAlmostEqual(21178, start_diff, 0)


if __name__ == "__main__":
    unittest.main()
