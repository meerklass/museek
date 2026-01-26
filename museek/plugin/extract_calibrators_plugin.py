import sys

import matplotlib.pyplot as plt
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result

from museek.enums.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.calibrator_finder import find_calibrators


class ExtractCalibratorsPlugin(AbstractPlugin):
    """Plugin to extract and validate single-dish calibrator scans using simple calibrator finding functions."""

    def __init__(
        self,
        n_calibrator_observations: int,
        calibrator_names: list[str],
        n_pointings: int,
        max_gap_seconds: float = 40.0,
        min_duration_seconds: float = 20.0,
    ):
        """
        Initialize with calibrator finding parameters.

        :param n_calibrator_observations: Number of calibrator observations (typically 2: before/after scan)
        :param calibrator_names: Expected calibrator names (must match n_calibrator_observations length)
        :param n_pointings: Exact number of scans required for each calibrator
        :param max_gap_seconds: Maximum allowed time gap between calibrator track scans in seconds
        :param min_duration_seconds: Minimum scan duration in seconds to be considered valid
        """
        super().__init__()
        self.n_calibrator_observations = n_calibrator_observations
        self.calibrator_observation_labels = ["before_scan", "after_scan"]
        self.calibrator_names = calibrator_names
        self.n_pointings = n_pointings
        self.max_gap_seconds = max_gap_seconds
        self.min_duration_seconds = min_duration_seconds

        # Validate n_calibrator_observations value
        if self.n_calibrator_observations not in [1, 2]:
            raise ValueError(
                f"n_calibrator_observations must be 1 or 2, got {self.n_calibrator_observations}"
            )

        # Validate calibrator_names length
        if (
            self.calibrator_names
            and len(self.calibrator_names) != self.n_calibrator_observations
        ):
            raise ValueError(
                f"calibrator_names length ({len(self.calibrator_names)}) must match "
                f"n_calibrator_observations ({self.n_calibrator_observations})"
            )

    def set_requirements(self):
        """Define the plugin requirements"""
        self.requirements = [
            Requirement(location=ResultEnum.TRACK_DATA, variable="track_data"),
            Requirement(
                location=ResultEnum.SCAN_OBSERVATION_START, variable="scan_start"
            ),
            Requirement(location=ResultEnum.SCAN_OBSERVATION_END, variable="scan_end"),
        ]

    def run(self, track_data: TimeOrderedData, scan_start: float, scan_end: float):
        """
        Find and validate single-dish calibrator scans.
        :param track_data: tracking part of the time ordered data
        :param scan_start: time dump [s] of scan observation start
        :param scan_end: time dump [s] of scan observation end
        """
        # Find single dish calibrators in the track data
        calibrator_results = find_calibrators(
            track_data=track_data,
            scan_start=scan_start,
            scan_end=scan_end,
            calibrator_names=self.calibrator_names,
            min_duration_seconds=self.min_duration_seconds,
            max_gap_seconds=self.max_gap_seconds,
        )

        # Validate results based on user expectations
        validation_success, validated_periods = self._validate_and_report_results(
            calibrator_results
        )

        # Exit if validation failed
        if not validation_success:
            print("Calibration validation failed. Terminating pipeline.")
            sys.exit(1)

        # Store results for downstream plugins
        validated_dump_indices = {}
        for period in validated_periods:
            dump_indices, scan_count, total_duration = calibrator_results[period]
            validated_dump_indices[period] = dump_indices

        self.set_result(
            result=Result(
                location=ResultEnum.CALIBRATOR_VALIDATED_PERIODS,
                result=validated_periods,
                allow_overwrite=False,
            )
        )
        self.set_result(
            result=Result(
                location=ResultEnum.CALIBRATOR_DUMP_INDICES,
                result=validated_dump_indices,
                allow_overwrite=False,
            )
        )

        # Plot RA, Dec positions and elevation vs time
        self._plot_calibrator_positions(
            track_data, validated_periods, calibrator_results
        )
        self._plot_elevation_vs_time(track_data, validated_periods, calibrator_results)

    def _validate_and_report_results(self, calibrator_results):
        """Validate found calibrators against user expectations and report results.

        Returns:
            tuple: (validation_success: bool, validated_periods: list)
        """
        validated_periods = []

        # Process each period: report findings and validate
        for period, result in calibrator_results.items():
            # Get correct calibrator name for this period
            if period == "before_scan":
                calibrator_name = self.calibrator_names[0]
            else:  # after_scan
                calibrator_name = self.calibrator_names[-1]

            if result is not None:
                dump_indices, scan_count, total_duration = result
                print(
                    f'{period}: Found {scan_count} consecutive "{calibrator_name}" tracks, '
                    f"total duration: {total_duration:.1f}s"
                )

                # Validate scan count for this period
                if scan_count == self.n_pointings:
                    validated_periods.append(period)
                else:
                    print(
                        f"{period}: INVALID - Found {scan_count} tracks, need exactly {self.n_pointings}"
                    )
            else:
                print(f'{period}: No valid "{calibrator_name}" tracks found')

        # Validate overall results based on n_calibrator_observations
        validation_success = False
        if self.n_calibrator_observations == 1:
            if len(validated_periods) == 0:
                print("ERROR: No valid calibrator periods found")
            elif len(validated_periods) == 2:
                print(
                    f"ERROR: Found valid calibrators in both periods ({', '.join(validated_periods)}), expected exactly one"
                )
            elif len(validated_periods) == 1:
                period = validated_periods[0]
                dump_indices, scan_count, total_duration = calibrator_results[period]
                print(
                    f'SUCCESS: Calibrator "{self.calibrator_names[0]}" validated in {period} period'
                )
                print(
                    f"SUCCESS: {period} calibrator validated with {len(dump_indices)} dumps"
                )
                validation_success = True

        elif self.n_calibrator_observations == 2:
            if len(validated_periods) != 2:
                print(
                    f"ERROR: Found {len(validated_periods)} valid periods, expected exactly 2"
                )
            else:
                for period in validated_periods:
                    # Get correct calibrator name for this period
                    if period == "before_scan":
                        calibrator_name = self.calibrator_names[0]
                    else:  # after_scan
                        calibrator_name = self.calibrator_names[-1]

                    dump_indices, scan_count, total_duration = calibrator_results[
                        period
                    ]
                    print(
                        f'SUCCESS: {period} calibrator "{calibrator_name}" validated with {len(dump_indices)} dumps'
                    )
                validation_success = True

        return validation_success, validated_periods

    def _plot_calibrator_positions(
        self, track_data, validated_periods, calibrator_results
    ):
        """Plot RA, Dec positions for validated calibrator tracks (first receiver only)."""
        # Use first receiver only
        first_receiver = track_data.receivers[0]
        antenna_index = first_receiver.antenna_index(receivers=track_data.receivers)

        # Get RA, Dec data for first receiver
        ra_data = track_data.right_ascension.get(recv=antenna_index)
        dec_data = track_data.declination.get(recv=antenna_index)

        plt.figure(figsize=(10, 8))

        # Plot for each validated period
        colors = ["blue", "red"]  # Different colors for before_scan vs after_scan
        for i, period in enumerate(validated_periods):
            dump_indices, scan_count, total_duration = calibrator_results[period]

            # Get boolean mask for absolute dump indices
            select = track_data.dump_mask(dump_indices)

            # Extract RA, Dec using boolean indexing
            ra_values = ra_data.squeeze[select]
            dec_values = dec_data.squeeze[select]

            # Create scatter plot (data is already in degrees)
            plt.scatter(
                ra_values,
                dec_values,
                c=colors[i % len(colors)],
                alpha=0.6,
                s=20,
                label=f"{period} ({scan_count} tracks, {total_duration:.1f}s)",
            )

        plt.xlabel("Right Ascension (degrees)")
        plt.ylabel("Declination (degrees)")
        if len(set(self.calibrator_names)) == 1:
            # Same calibrator for all periods
            plt.title(f"Calibrator Track Positions - {self.calibrator_names[0]}")
            plot_filename = (
                f"calibrator_positions_{self.calibrator_names[0].lower()}.png"
            )
        else:
            # Different calibrators
            calibrator_list = "_".join([name.lower() for name in self.calibrator_names])
            plt.title(
                f"Calibrator Track Positions - {' & '.join(self.calibrator_names)}"
            )
            plot_filename = f"calibrator_positions_{calibrator_list}.png"

        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"Calibrator position plot saved to: {plot_filename}")
        plt.close()

    def _plot_elevation_vs_time(
        self, track_data, validated_periods, calibrator_results
    ):
        """Plot elevation vs time for validated calibrator tracks (first receiver only)."""
        # Use first receiver only
        first_receiver = track_data.receivers[0]
        antenna_index = first_receiver.antenna_index(receivers=track_data.receivers)

        # Get elevation and timestamp data for first receiver
        elevation_data = track_data.elevation.get(recv=antenna_index)
        timestamp_data = track_data.original_timestamps.get(recv=antenna_index)

        plt.figure(figsize=(12, 6))

        # Plot for each validated period
        colors = ["blue", "red"]
        for i, period in enumerate(validated_periods):
            dump_indices, scan_count, total_duration = calibrator_results[period]

            # Get boolean mask for absolute dump indices
            select = track_data.dump_mask(dump_indices)

            # Extract elevation and timestamps using boolean indexing
            elevation_values = elevation_data.squeeze[select]
            timestamp_values = timestamp_data.squeeze[select]

            # Convert timestamps to relative time in minutes from first timestamp
            time_minutes = (timestamp_values - timestamp_values[0]) / 60.0

            # Create line plot (elevation data is already in degrees)
            plt.plot(
                time_minutes,
                elevation_values,
                color=colors[i % len(colors)],
                linewidth=2,
                marker="o",
                markersize=3,
                label=f"{period} ({scan_count} tracks, {total_duration:.1f}s)",
            )

        plt.xlabel("Time (minutes from start)")
        plt.ylabel("Elevation (degrees)")
        if len(set(self.calibrator_names)) == 1:
            # Same calibrator for all periods
            plt.title(f"Calibrator Elevation vs Time - {self.calibrator_names[0]}")
            plot_filename = (
                f"calibrator_elevation_{self.calibrator_names[0].lower()}.png"
            )
        else:
            # Different calibrators
            calibrator_list = "_".join([name.lower() for name in self.calibrator_names])
            plt.title(
                f"Calibrator Elevation vs Time - {' & '.join(self.calibrator_names)}"
            )
            plot_filename = f"calibrator_elevation_{calibrator_list}.png"

        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"Calibrator elevation plot saved to: {plot_filename}")
        plt.close()
