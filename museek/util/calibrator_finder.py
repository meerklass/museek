from typing import Dict

import numpy as np

from museek.time_ordered_data import TimeOrderedData
from museek.enums.scan_state_enum import ScanStateEnum


def _matches_calibrator_name(target_name: str, expected_calibrator: str) -> bool:
    """Check if a target name matches the expected calibrator (case-insensitive prefix match)."""
    return target_name.lower().startswith(expected_calibrator.lower())


def _find_calibrator_scans_in_period(track_data: TimeOrderedData, calibrator_name: str, period: str, 
                                    scan_start: float, scan_end: float, 
                                    min_duration_seconds: float, max_gap_seconds: float) -> tuple[range, int] | None:
    """
    Find calibrator scans in a specific time period (before_scan or after_scan).
    
    :param track_data: Time ordered track data
    :param calibrator_name: Name of calibrator to search for
    :param period: Either 'before_scan' or 'after_scan'
    :param scan_start: Scan start time boundary
    :param scan_end: Scan end time boundary
    :param min_duration_seconds: Minimum scan duration to be considered valid
    :param max_gap_seconds: Maximum gap between consecutive scans
    :return: Tuple of (dump indices list, scan count) or None if no valid sequence found
    """
    matching_scans = []
    original_timestamps = track_data.original_timestamps.squeeze
    
    # Find matching scans in the specified period
    for scan_tuple in track_data._scan_tuple_list:
        if scan_tuple.state == ScanStateEnum.TRACK:
            # Safety check: ensure we have at least 3 dumps to trim edges
            if len(scan_tuple.dumps) < 3:
                continue  # Skip tracks that are too short to trim
            
            # Trim edge dumps (remove first and last dump from each track)
            trimmed_dumps = scan_tuple.dumps[1:-1]
            
            # Recalculate timestamps based on trimmed dumps
            scan_start_time = original_timestamps[trimmed_dumps[0]]
            scan_end_time = original_timestamps[trimmed_dumps[-1]]
            scan_duration = scan_end_time - scan_start_time
            
            # Time-based filtering for period
            in_period = False
            if period == 'before_scan' and scan_end_time < scan_start:
                in_period = True
            elif period == 'after_scan' and scan_start_time > scan_end:
                in_period = True
            
            # Apply all filters: period, target name, and duration
            if (in_period and 
                _matches_calibrator_name(scan_tuple.target.name, calibrator_name) and
                scan_duration > min_duration_seconds):
                matching_scans.append((scan_tuple.target.name, trimmed_dumps, scan_start_time, scan_end_time))
    
    if not matching_scans:
        return None
    
    # Sort by start time
    matching_scans.sort(key=lambda x: x[2])
    
    # Check time gaps between consecutive scans
    for i in range(1, len(matching_scans)):
        prev_end_time = matching_scans[i-1][3]
        curr_start_time = matching_scans[i][2]
        gap = curr_start_time - prev_end_time
        
        if gap > max_gap_seconds:
            return None
    
    # Collect all dumps from matching scans
    all_matching_dumps = []
    for target_name, scan_dumps, _, _ in matching_scans:
        all_matching_dumps.extend(scan_dumps)
    
    all_matching_dumps.sort()
    
    # Return the list of valid dump indices and the scan count
    return all_matching_dumps, len(matching_scans)


def find_calibrators(track_data: TimeOrderedData, scan_start: float, scan_end: float, 
                    calibrator_names: list[str], min_duration_seconds: float = 10.0, 
                    max_gap_seconds: float = 30.0) -> Dict[str, tuple[range, int, float] | None]:
    """
    Find calibrator scans in before_scan and after_scan periods.
    
    :param track_data: Time ordered track data
    :param scan_start: Time of scan observation start
    :param scan_end: Time of scan observation end
    :param calibrator_names: List of calibrator names - uses first for before_scan, last for after_scan
    :param min_duration_seconds: Minimum scan duration to be considered valid
    :param max_gap_seconds: Maximum gap between consecutive scans
    :return: Dictionary with 'before_scan' and 'after_scan' keys, each containing
             (dump_indices_list, scan_count, total_duration) tuple or None if not found
    """
    results = {'before_scan': None, 'after_scan': None}
    
    # Before scan: use first calibrator name
    before_result = _find_calibrator_scans_in_period(
        track_data, calibrator_names[0], 'before_scan', scan_start, scan_end, 
        min_duration_seconds, max_gap_seconds
    )
    if before_result is not None:
        dump_indices, scan_count = before_result
        original_timestamps = track_data.original_timestamps.squeeze
        total_duration = original_timestamps[dump_indices[-1]] - original_timestamps[dump_indices[0]]
        results['before_scan'] = (dump_indices, scan_count, total_duration)
    
    # After scan: use last calibrator name
    after_result = _find_calibrator_scans_in_period(
        track_data, calibrator_names[-1], 'after_scan', scan_start, scan_end, 
        min_duration_seconds, max_gap_seconds
    )
    if after_result is not None:
        dump_indices, scan_count = after_result
        original_timestamps = track_data.original_timestamps.squeeze
        total_duration = original_timestamps[dump_indices[-1]] - original_timestamps[dump_indices[0]]
        results['after_scan'] = (dump_indices, scan_count, total_duration)
    
    return results
