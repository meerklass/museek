import unittest

from museek.antenna_sanity.from_log import FromLog


class TestFromLog(unittest.TestCase):
    def test_straggler_list(self):
        straggler_mock_list = ["m006", "m031", "m056", "m060"]
        obs_mock_log = [
            "2023-07-11 17:46:29.817Z INFO     start of scan reached",
            "2023-07-11 17:46:29.817Z INFO     performing scan",
            "2023-07-11 17:49:25.329Z INFO     Waiting for sensor 'scan_status' == 'after' had 3 straggler(s): ['m006', 'm031', 'm056']",
            "2023-07-11 17:49:25.331Z INFO     scan complete",
            "2023-07-11 17:52:29.366Z INFO     performing scan",
            "2023-07-11 17:55:24.860Z INFO     Waiting for sensor 'scan_status' == 'after' had 3 straggler(s): ['m006', 'm056', 'm060']",
            "2023-07-11 17:55:24.861Z INFO     scan complete",
            "2023-07-11 17:55:24.866Z INFO     Scan completed - 38 scan lines",
            "2023-07-11 17:55:24.868Z INFO     Initialising Track target 3C273_u0.8 for 70.0 sec",
            "2023-07-11 17:55:24.869Z INFO     New compound scan: 'track'",
            "2023-07-11 17:55:24.870Z INFO     Initiating 70-second track on target '3C273_u0.8'",
            "2023-07-11 17:55:27.536Z INFO     slewing to target",
            "2023-07-11 17:55:38.385Z INFO     Waiting for sensor 'lock' == True had 3 straggler(s): ['m016', 'm056', 'm060']",
            "2023-07-11 17:55:38.386Z INFO     target reached",
            "2023-07-11 17:55:38.390Z INFO     tracking target",
            "2023-07-11 17:56:48.461Z INFO     target tracked for 70.0 seconds",
            "2023-07-11 17:56:48.463Z INFO     Initialising Track target 3C273 for 70.0 sec",
            "2023-07-11 17:56:48.465Z INFO     New compound scan: 'track'",
            "2023-07-11 17:56:48.465Z INFO     Initiating 70-second track on target '3C273'",
            "2023-07-11 17:56:51.260Z INFO     slewing to target",
            "2023-07-11 17:56:58.350Z INFO     Waiting for sensor 'lock' == True had 1 straggler(s): ['m012']",
            "2023-07-11 17:56:58.351Z INFO     target reached",
            "2023-07-11 17:56:58.356Z INFO     tracking target",
        ]
        straggler_list = FromLog(obs_script_log=obs_mock_log).straggler_list()
        self.assertEqual(straggler_mock_list, straggler_list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
