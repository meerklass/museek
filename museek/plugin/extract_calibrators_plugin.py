from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result

from museek.enums.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.calibrator_finder import find_calibrators, detect_calibrator_names
import matplotlib.pyplot as plt
import numpy as np
import sys


class ExtractCalibratorsPlugin(AbstractPlugin):
    """ Plugin to extract and validate single-dish calibrator scans using simple calibrator finding functions. """
    
    def __init__(self,
                 n_pointings: int,
                 n_calibrator_observations: int = 2,
                 calibrator_names: list[str] | None = None,
                 max_gap_seconds: float = 40.0,
                 min_duration_seconds: float = 20.0,
                 verbose: int = 0):
        """
        Initialize with calibrator finding parameters.

        :param n_pointings: Exact number of scans required for each calibrator
        :param n_calibrator_observations: Number of calibrator observations (typically 2: before/after scan)
        :param calibrator_names: Calibrator names [before, after]. If None, auto-detected from track data.
        :param max_gap_seconds: Maximum allowed time gap between calibrator track scans in seconds
        :param min_duration_seconds: Minimum scan duration in seconds to be considered valid
        """
        super().__init__()
        self.n_calibrator_observations = n_calibrator_observations
        self.calibrator_observation_labels = ['before_scan', 'after_scan']
        self.calibrator_names = calibrator_names
        self.n_pointings = n_pointings
        self.max_gap_seconds = max_gap_seconds
        self.min_duration_seconds = min_duration_seconds
        self.verbose = verbose

        # Validate n_calibrator_observations value
        if self.n_calibrator_observations not in [1, 2]:
            raise ValueError(f"n_calibrator_observations must be 1 or 2, got {self.n_calibrator_observations}")

        # calibrator_names maps to periods as: [0] -> before_scan, [-1] -> after_scan.
        # Use 2 entries for different calibrators, or 1 entry if the same source is used for both.
        if self.calibrator_names is not None and (len(self.calibrator_names) == 0 or len(self.calibrator_names) > 2):
            raise ValueError(f"calibrator_names must have 1 or 2 entries, got {len(self.calibrator_names)}")

    def set_requirements(self):
        """ Define the plugin requirements """
        self.requirements = [Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.SCAN_OBSERVATION_START, variable='scan_start'),
                             Requirement(location=ResultEnum.SCAN_OBSERVATION_END, variable='scan_end')]

    def run(self,
            track_data: TimeOrderedData,
            scan_start: float,
            scan_end: float):
        """
        Find and validate single-dish calibrator scans.
        :param track_data: tracking part of the time ordered data
        :param scan_start: time dump [s] of scan observation start
        :param scan_end: time dump [s] of scan observation end
        """
        # Auto-detect calibrator names if not provided
        if self.calibrator_names is None:
            self.calibrator_names = detect_calibrator_names(
                track_data=track_data,
                scan_start=scan_start,
                scan_end=scan_end,
            )
            if not self.calibrator_names:
                print('ERROR: No known calibrators detected in track data.')
                sys.exit(1)
            print(f'Auto-detected calibrators: {self.calibrator_names}')

        # Find single dish calibrators in the track data
        calibrator_results = find_calibrators(
            track_data=track_data,
            scan_start=scan_start,
            scan_end=scan_end,
            calibrator_names=self.calibrator_names,
            min_duration_seconds=self.min_duration_seconds,
            max_gap_seconds=self.max_gap_seconds
        )
        
        # Validate results based on user expectations
        validation_success, validated_periods = self._validate_and_report_results(calibrator_results)
        
        # Exit if validation failed
        if not validation_success:
            print(f'Calibration validation failed. Terminating pipeline.')
            sys.exit(1)
        
        # Store results for downstream plugins
        validated_dump_indices = {}
        calibrator_names_for_periods = {}

        for period in validated_periods:
            dump_indices, scan_count, total_duration = calibrator_results[period]
            validated_dump_indices[period] = dump_indices

            # Store calibrator name for this period
            if period == 'before_scan':
                calibrator_names_for_periods[period] = self.calibrator_names[0]
            else:  # after_scan
                calibrator_names_for_periods[period] = self.calibrator_names[-1]

        self.set_result(result=Result(location=ResultEnum.CALIBRATOR_VALIDATED_PERIODS,
                                    result=validated_periods, allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.CALIBRATOR_DUMP_INDICES,
                                    result=validated_dump_indices, allow_overwrite=False))
        self.set_result(result=Result(location=ResultEnum.CALIBRATOR_NAMES,
                                    result=calibrator_names_for_periods, allow_overwrite=False))
        
        if self.verbose:
            self._plot_calibrator_positions(track_data, validated_periods, calibrator_results)
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
            if period == 'before_scan':
                calibrator_name = self.calibrator_names[0]
            else:  # after_scan
                calibrator_name = self.calibrator_names[-1]
            
            if result is not None:
                dump_indices, scan_count, total_duration = result
                print(f'{period}: Found {scan_count} consecutive "{calibrator_name}" tracks, '
                      f'total duration: {total_duration:.1f}s')
                
                # Validate scan count for this period
                if scan_count == self.n_pointings:
                    validated_periods.append(period)
                else:
                    print(f'{period}: INVALID - Found {scan_count} tracks, need exactly {self.n_pointings}')
            else:
                print(f'{period}: No valid "{calibrator_name}" tracks found')
        
        # Validate: at least one calibrator period must be found
        if len(validated_periods) == 0:
            print(f'ERROR: No valid calibrator periods found')
            return False, validated_periods

        if len(validated_periods) < self.n_calibrator_observations:
            print(f'WARNING: Found {len(validated_periods)} valid period(s), '
                  f'expected {self.n_calibrator_observations}. Continuing with what is available.')

        for period in validated_periods:
            if period == 'before_scan':
                calibrator_name = self.calibrator_names[0]
            else:  # after_scan
                calibrator_name = self.calibrator_names[-1]
            dump_indices, scan_count, total_duration = calibrator_results[period]
            print(f'SUCCESS: {period} calibrator "{calibrator_name}" validated with {len(dump_indices)} dumps')

        return True, validated_periods
    
    def _plot_calibrator_positions(self, track_data, validated_periods, calibrator_results):
        """Plot RA, Dec positions for validated calibrator tracks (first receiver only)."""
        # Use first receiver only
        first_receiver = track_data.receivers[0]
        antenna_index = track_data.antenna_index_of_receiver(receiver=first_receiver)
        
        # Get RA, Dec data for first receiver
        ra_data = track_data.right_ascension.get(recv=antenna_index)
        dec_data = track_data.declination.get(recv=antenna_index)
        
        plt.figure(figsize=(10, 8))

        # Plot for each validated period
        colors = ['blue', 'red']  # Different colors for before_scan vs after_scan
        for i, period in enumerate(validated_periods):
            dump_indices, scan_count, total_duration = calibrator_results[period]

            # Get boolean mask for absolute dump indices
            select = track_data.dump_mask(dump_indices)

            # Extract RA, Dec using boolean indexing
            ra_values = ra_data.squeeze[select]
            dec_values = dec_data.squeeze[select]
            
            # Create scatter plot (data is already in degrees)
            plt.scatter(ra_values, dec_values, 
                       c=colors[i % len(colors)], alpha=0.6, s=20,
                       label=f'{period} ({scan_count} tracks, {total_duration:.1f}s)')
        
        plt.xlabel('Right Ascension (degrees)')
        plt.ylabel('Declination (degrees)')
        if len(set(self.calibrator_names)) == 1:
            # Same calibrator for all periods
            plt.title(f'Calibrator Track Positions - {self.calibrator_names[0]}')
            plot_filename = f'calibrator_positions_{self.calibrator_names[0].lower()}.png'
        else:
            # Different calibrators
            calibrator_list = '_'.join([name.lower() for name in self.calibrator_names])
            plt.title(f'Calibrator Track Positions - {" & ".join(self.calibrator_names)}')
            plot_filename = f'calibrator_positions_{calibrator_list}.png'
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Calibrator position plot saved to: {plot_filename}')
        plt.close()
    
    def _plot_elevation_vs_time(self, track_data, validated_periods, calibrator_results):
        """Plot elevation vs time for validated calibrator tracks (first receiver only)."""
        # Use first receiver only
        first_receiver = track_data.receivers[0]
        antenna_index = track_data.antenna_index_of_receiver(receiver=first_receiver)
        
        # Get elevation and timestamp data for first receiver
        elevation_data = track_data.elevation.get(recv=antenna_index)
        timestamp_data = track_data.timestamps.get(recv=antenna_index)

        plt.figure(figsize=(12, 6))

        # Plot for each validated period
        colors = ['blue', 'red']
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
            plt.plot(time_minutes, elevation_values, 
                    color=colors[i % len(colors)], linewidth=2, marker='o', markersize=3,
                    label=f'{period} ({scan_count} tracks, {total_duration:.1f}s)')
        
        plt.xlabel('Time (minutes from start)')
        plt.ylabel('Elevation (degrees)')
        if len(set(self.calibrator_names)) == 1:
            # Same calibrator for all periods
            plt.title(f'Calibrator Elevation vs Time - {self.calibrator_names[0]}')
            plot_filename = f'calibrator_elevation_{self.calibrator_names[0].lower()}.png'
        else:
            # Different calibrators
            calibrator_list = '_'.join([name.lower() for name in self.calibrator_names])
            plt.title(f'Calibrator Elevation vs Time - {" & ".join(self.calibrator_names)}')
            plot_filename = f'calibrator_elevation_{calibrator_list}.png'
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Calibrator elevation plot saved to: {plot_filename}')
        plt.close()
