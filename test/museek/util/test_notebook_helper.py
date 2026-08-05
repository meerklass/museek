import unittest

import numpy as np
import xarray as xr

from museek.util import notebook_helper


class TestTrimFrequencies(unittest.TestCase):
    def setUp(self):
        self.freq_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    def test_trim_frequencies(self):
        trimmed = notebook_helper.trim_frequencies(
            self.freq_array, frequency_range=(150.0, 450.0)
        )
        np.testing.assert_array_equal(np.array([200.0, 300.0, 400.0]), trimmed)

    def test_trim_frequencies_return_slice(self):
        trimmed, freq_slice = notebook_helper.trim_frequencies(
            self.freq_array, frequency_range=(150.0, 450.0), return_slice=True
        )
        np.testing.assert_array_equal(np.array([200.0, 300.0, 400.0]), trimmed)
        self.assertEqual(slice(1, 4), freq_slice)

    def test_trim_frequencies_below_range_raises(self):
        with self.assertRaises(ValueError):
            notebook_helper.trim_frequencies(
                self.freq_array, frequency_range=(50.0, 450.0)
            )

    def test_trim_frequencies_above_range_raises(self):
        with self.assertRaises(ValueError):
            notebook_helper.trim_frequencies(
                self.freq_array, frequency_range=(150.0, 600.0)
            )

    def test_trim_frequencies_empty_result_raises(self):
        with self.assertRaises(ValueError):
            notebook_helper.trim_frequencies(
                self.freq_array, frequency_range=(210.0, 290.0)
            )


class TestCircularMeanDeg(unittest.TestCase):
    def test_circular_mean_deg_no_wrap(self):
        mean = notebook_helper.circular_mean_deg(np.array([10.0, 20.0, 30.0]))
        self.assertAlmostEqual(20.0, mean)

    def test_circular_mean_deg_wraps_at_360(self):
        mean = notebook_helper.circular_mean_deg(
            np.array([353.0, 355.0, 359.0, 1.0, 3.0])
        )
        # The naive arithmetic mean would be ~214.2 (badly wrong); the circular mean
        # should land near 358 (i.e. close to 0/360), not near 180.
        self.assertLess(min(abs(mean - 358.2), abs(mean - 358.2 + 360)), 1.0)


class TestWrapToNearest(unittest.TestCase):
    def test_wrap_to_nearest_no_shift_needed(self):
        wrapped = notebook_helper.wrap_to_nearest(
            np.array([10.0, 20.0]), reference=15.0
        )
        np.testing.assert_allclose(np.array([10.0, 20.0]), wrapped)

    def test_wrap_to_nearest_shifts_across_branch(self):
        wrapped = notebook_helper.wrap_to_nearest(np.array([358.0]), reference=2.0)
        np.testing.assert_allclose(np.array([-2.0]), wrapped)

    def test_wrap_to_nearest_array(self):
        wrapped = notebook_helper.wrap_to_nearest(
            np.array([353.0, 355.0, 359.0, 1.0, 3.0]), reference=0.0
        )
        np.testing.assert_allclose(np.array([-7.0, -5.0, -1.0, 1.0, 3.0]), wrapped)


class TestReduceFlags(unittest.TestCase):
    def setUp(self):
        self.flag_da = xr.DataArray(
            np.array([[True, False], [False, False], [True, True]]),
            dims=("timestamps", "feeds"),
            coords={"timestamps": [0, 1, 2], "feeds": ["h", "v"]},
        )

    def test_reduce_flags_or(self):
        reduced = notebook_helper.reduce_flags(
            self.flag_da, output_dims=("timestamps",), operator="or"
        )
        self.assertEqual(("timestamps",), reduced.dims)
        np.testing.assert_array_equal(np.array([True, False, True]), reduced.values)

    def test_reduce_flags_and(self):
        reduced = notebook_helper.reduce_flags(
            self.flag_da, output_dims=("timestamps",), operator="and"
        )
        np.testing.assert_array_equal(np.array([False, False, True]), reduced.values)

    def test_reduce_flags_xor(self):
        reduced = notebook_helper.reduce_flags(
            self.flag_da, output_dims=("timestamps",), operator="xor"
        )
        np.testing.assert_array_equal(np.array([True, False, False]), reduced.values)

    def test_reduce_flags_multiple_dims(self):
        flag_da = xr.DataArray(
            np.array(
                [
                    [[True, False], [False, False]],
                    [[False, False], [False, True]],
                ]
            ),
            dims=("timestamps", "frequencies", "feeds"),
            coords={
                "timestamps": [0, 1],
                "frequencies": [100.0, 200.0],
                "feeds": ["h", "v"],
            },
        )
        reduced = notebook_helper.reduce_flags(
            flag_da, output_dims=("timestamps",), operator="or"
        )
        self.assertEqual(("timestamps",), reduced.dims)
        np.testing.assert_array_equal(np.array([True, True]), reduced.values)

    def test_reduce_flags_invalid_output_dims_raises(self):
        with self.assertRaises(ValueError):
            notebook_helper.reduce_flags(
                self.flag_da, output_dims=("bogus",), operator="or"
            )

    def test_reduce_flags_unreachable_output_dims_raises(self):
        with self.assertRaises(ValueError):
            notebook_helper.reduce_flags(
                self.flag_da, output_dims=("frequencies",), operator="or"
            )


class TestCombineFlags(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset(
            {
                "flag_a": (("timestamps",), np.array([True, False, False])),
                "flag_b": (("timestamps",), np.array([False, False, True])),
            },
            coords={"timestamps": [0, 1, 2]},
        )

    def test_combine_flags_single_string_returns_unchanged(self):
        combined = notebook_helper.combine_flags(self.ds, "flag_a")
        np.testing.assert_array_equal(self.ds["flag_a"].values, combined.values)

    def test_combine_flags_single_dataarray_returns_unchanged(self):
        flag = self.ds["flag_b"]
        combined = notebook_helper.combine_flags(self.ds, flag)
        self.assertIs(flag, combined)

    def test_combine_flags_or(self):
        combined = notebook_helper.combine_flags(
            self.ds, ["flag_a", "flag_b"], combine="or"
        )
        np.testing.assert_array_equal(np.array([True, False, True]), combined.values)

    def test_combine_flags_and(self):
        combined = notebook_helper.combine_flags(
            self.ds, ["flag_a", "flag_b"], combine="and"
        )
        np.testing.assert_array_equal(np.array([False, False, False]), combined.values)

    def test_combine_flags_mixed_string_and_dataarray(self):
        combined = notebook_helper.combine_flags(
            self.ds, ["flag_a", self.ds["flag_b"]], combine="or"
        )
        np.testing.assert_array_equal(np.array([True, False, True]), combined.values)


class TestSelectAndFlag(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset(
            {
                "vis": (
                    ("timestamps", "frequencies"),
                    np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                ),
                "flag": (
                    ("timestamps", "frequencies"),
                    np.array([[False, False], [True, False], [False, True]]),
                ),
                "ra": (("timestamps",), np.array([10.0, 20.0, 30.0])),
            },
            coords={"timestamps": [0, 1, 2], "frequencies": [100.0, 200.0]},
        )

    def test_select_and_flag_no_flags(self):
        result = notebook_helper.select_and_flag(self.ds, "vis")
        np.testing.assert_array_equal(self.ds["vis"].values, result.values)

    def test_select_and_flag_masks_matching_dims(self):
        result = notebook_helper.select_and_flag(self.ds, "vis", flags="flag")
        expected = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan]])
        np.testing.assert_array_equal(expected, result.values)

    def test_select_and_flag_reduces_extra_flag_dims(self):
        # "flag" carries a "frequencies" dim that "ra" doesn't have; it should be
        # OR-reduced away before masking.
        result = notebook_helper.select_and_flag(self.ds, "ra", flags="flag")
        expected = np.array([10.0, np.nan, np.nan])
        np.testing.assert_array_equal(expected, result.values)

    def test_select_and_flag_dropna(self):
        result = notebook_helper.select_and_flag(
            self.ds, "ra", flags="flag", dropna_dim="timestamps"
        )
        np.testing.assert_array_equal(np.array([10.0]), result.values)

    def test_select_and_flag_sel(self):
        result = notebook_helper.select_and_flag(
            self.ds, "vis", sel={"frequencies": 100.0}
        )
        self.assertNotIn("frequencies", result.dims)
        np.testing.assert_array_equal(np.array([1.0, 3.0, 5.0]), result.values)

    def test_select_and_flag_isel(self):
        result = notebook_helper.select_and_flag(
            self.ds, "vis", isel={"timestamps": slice(0, 2)}
        )
        np.testing.assert_array_equal(np.array([[1.0, 2.0], [3.0, 4.0]]), result.values)

    def test_select_and_flag_nearest(self):
        result = notebook_helper.select_and_flag(
            self.ds, "vis", nearest={"frequencies": 180.0}
        )
        self.assertNotIn("frequencies", result.dims)
        np.testing.assert_array_equal(np.array([2.0, 4.0, 6.0]), result.values)


if __name__ == "__main__":
    unittest.main()
