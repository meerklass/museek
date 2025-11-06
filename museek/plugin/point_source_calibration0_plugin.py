import os
from typing import Generator

import numpy as np
import numpy.ma as ma
import datetime
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
import astropy.units as u

from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.model.atmospheric_opacity import AtmosphericModel
from museek.model.point_sources import get_point_source_model
from museek.model.primary_beam import PrimaryBeam
from museek.model.receiver_temperature import ReceiverTemperature
from museek.model.spillover_temperature import SpilloverTemperature
from museek.noise_diode import NoiseDiode
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import git_version_info


def calculate_median_coordinates_excluding_flagged_antennas(
    track_data: TimeOrderedData,
    flag_name_list: list
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate median azimuth and elevation per time dump, excluding antennas
    flagged by 'outlier_antenna_flag' or 'elevation_flag'.

    Uses numpy masked arrays for efficient vectorized calculation. For each time dump,
    calculates the median pointing across all unflagged antennas. If all antennas are
    flagged at a particular time, uses antenna 0 as fallback.

    Parameters
    ----------
    track_data : TimeOrderedData
        The track data object containing coordinates and flags
    flag_name_list : list
        List of flag names corresponding to track_data.flags._flags

    Returns
    -------
    median_az : np.ndarray
        Median azimuth per time dump, shape (n_time,), in degrees
    median_el : np.ndarray
        Median elevation per time dump, shape (n_time,), in degrees
    """

    # Extract and combine 'outlier_antenna_flag' and 'elevation_flag'
    flag_indices = []
    for flag_name in ['outlier_antenna_flag', 'elevation_flag']:
        if flag_name in flag_name_list:
            flag_indices.append(flag_name_list.index(flag_name))

    if not flag_indices:
        # No antenna flags present - use all antennas
        az_data = track_data.azimuth.squeeze  # Shape: (n_time, n_antennas)
        el_data = track_data.elevation.squeeze
        median_az = np.median(az_data, axis=1)
        median_el = np.median(el_data, axis=1)
        return median_az, median_el

    # Combine the two flags using logical OR
    combined_flag_array = np.zeros(track_data.flags.shape, dtype=bool)
    for idx in flag_indices:
        combined_flag_array = np.logical_or(
            combined_flag_array,
            track_data.flags._flags[idx].get_array()
        )

    # Create per-antenna, per-time flag array
    # Flags shape: (n_time, n_freq, n_receivers)
    # Receivers: n_receivers = 2 * n_antennas (H and V polarization)

    # Pick one receiver per antenna (H polarization = even indices: 0, 2, 4, ...)
    # Pick middle frequency channel (any channel works - if antenna flagged, all are)
    mid_freq_channel = combined_flag_array.shape[1] // 2
    receiver_indices = np.arange(0, combined_flag_array.shape[2], 2)  # [0, 2, 4, 6, ...]

    # Extract per-antenna, per-time flag array
    # Shape: (n_time, n_receivers//2)
    antenna_flags = combined_flag_array[:, mid_freq_channel, :][:, receiver_indices]

    # Get azimuth and elevation data
    az_data = track_data.azimuth.squeeze  # Shape: (n_time, n_antennas)
    el_data = track_data.elevation.squeeze

    # Create masked arrays where mask=True means "exclude this antenna"
    az_masked = ma.masked_array(az_data, mask=antenna_flags)
    el_masked = ma.masked_array(el_data, mask=antenna_flags)

    # Calculate median across antennas (axis=1), keeping time dimension
    # Result has shape (n_time,)
    median_az = ma.median(az_masked, axis=1)
    median_el = ma.median(el_masked, axis=1)

    # Handle edge case - all antennas flagged at a time
    # If all antennas are masked at a time, use antenna 0 as fallback
    all_flagged_mask = np.all(antenna_flags, axis=1)  # Shape: (n_time,)

    if np.any(all_flagged_mask):
        # Convert masked array to regular numpy array
        median_az = median_az.filled(fill_value=0)
        median_el = median_el.filled(fill_value=0)

        # For times where all antennas are flagged, use antenna 0
        median_az[all_flagged_mask] = az_data[all_flagged_mask, 0]
        median_el[all_flagged_mask] = el_data[all_flagged_mask, 0]
    else:
        # Convert masked array to regular numpy array
        median_az = np.asarray(median_az)
        median_el = np.asarray(median_el)

    return median_az, median_el


class PointSourceCalibration0Plugin(AbstractParallelJoblibPlugin):
    """
    Plugin to calculate gain calibration using point source calibrators (e.g., HydraA, PictorA).

    This plugin processes track_data (calibrator pointings) to derive gain solutions
    by comparing measured visibility to expected flux from known calibrator sources.
    """

    def __init__(self,
                 flag_combination_threshold: int,
                 beam_file_path: str,
                 receiver_models_dir: str,
                 spillover_model_file: str,
                 do_store_context: bool,
                 **kwargs):
        """
        Initialize the plugin.

        :param flag_combination_threshold: Threshold for combining sets of flags, usually 1
        :param beam_file_path: Path to MeerKAT primary beam model file
        :param receiver_models_dir: Directory containing receiver temperature model files
        :param spillover_model_file: Path to spillover temperature model file
        :param do_store_context: If True, context is stored to disc after finishing
        """
        super().__init__(**kwargs)
        self.flag_combination_threshold = flag_combination_threshold
        self.beam_file_path = beam_file_path
        self.receiver_models_dir = receiver_models_dir
        self.spillover_model_file = spillover_model_file
        self.do_store_context = do_store_context

    def set_requirements(self):
        """Set the requirements for the plugin."""
        self.requirements = [
            Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
            Requirement(location=ResultEnum.CALIBRATOR_VALIDATED_PERIODS, variable='calibrator_validated_periods'),
            Requirement(location=ResultEnum.CALIBRATOR_DUMP_INDICES, variable='calibrator_dump_indices'),
            Requirement(location=ResultEnum.CALIBRATOR_NAMES, variable='calibrator_names'),
            Requirement(location=ResultEnum.FLAG_NAME_LIST, variable='flag_name_list'),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
            Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
            Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')
        ]

    def map(self,
            track_data: TimeOrderedData,
            calibrator_validated_periods: list,
            calibrator_dump_indices: dict,
            calibrator_names: dict,
            flag_name_list: list,
            flag_report_writer: ReportWriter,
            output_path: str,
            block_name: str) -> Generator[tuple, None, None]:
        """
        Yield data for each (period, receiver) combination to be processed in parallel.

        Loops over periods FIRST (outer loop), then receivers (inner loop).
        This allows point source flux and RA/Dec->Az/El conversion to be calculated once per period.

        :param track_data: Time ordered data containing calibrator tracking observations
        :param calibrator_validated_periods: List of validated calibrator periods (e.g., ['before_scan', 'after_scan'])
        :param calibrator_dump_indices: Dict mapping periods to dump indices
        :param calibrator_names: Dict mapping periods to calibrator names (from ExtractCalibratorsPlugin)
        :param flag_report_writer: Report writer for flagging info
        :param output_path: Path to store results
        :param block_name: Name of the observation block
        :yield: Tuple of (receiver_path, period, calibrator_name, vis_period, flag_period,
                         freq, atm_period, point_source_temp_period, dump_indices_period)
        """
        track_data.load_visibility_flags_weights(polars='auto')
        initial_flags = track_data.flags.combine(threshold=self.flag_combination_threshold)

        freq = track_data.frequencies.squeeze  # Frequencies in Hz
        freq_MHz = freq / 1e6  # Convert to MHz for beam model
        dumps = np.array(track_data._dumps())  # Absolute dump indices in track_data

        # Physical constants
        k_B = 1.380649e-23  # Boltzmann constant [J/K]
        c = 2.99792458e8    # Speed of light [m/s]

        # Calculate wavelength from frequency
        wavelength_m = c / freq  # Shape: (n_freq,)

        # Initialize beam model once
        beam_model = PrimaryBeam(self.beam_file_path)

        # Initialize spillover model once
        spillover_model = SpilloverTemperature(self.spillover_model_file)

        # Get observatory location from first antenna
        antenna0 = track_data.antennas[0]
        location = EarthLocation(
            lat=antenna0.ref_observer.lat * u.rad,
            lon=antenna0.ref_observer.lon * u.rad,
            height=antenna0.ref_observer.elevation * u.m
        )

        # Convert timestamps to astropy Time
        times = Time(track_data.timestamps.squeeze, format='unix')

        # Calculate atmospheric model once (before all loops)
        atmospheric_model = AtmosphericModel(track_data)

        # Calculate median coordinates per time, excluding flagged antennas
        median_az_all, median_el_all = \
            calculate_median_coordinates_excluding_flagged_antennas(track_data, flag_name_list)

        # Extract visibility and flags for all receivers (before period loop)
        visibility_data = {}
        flag_data = {}
        atm_emission_data = {}
        rec_temp_data = {}

        for i_receiver, receiver in enumerate(track_data.receivers):
            if track_data.visibility is not None:
                visibility_data[i_receiver] = track_data.visibility.get(recv=i_receiver)
            if initial_flags is not None:
                flag_data[i_receiver] = initial_flags.get(recv=i_receiver)

            # Get antenna index and atmospheric emission for this receiver
            antenna_idx = track_data.antenna_index_of_receiver(receiver=receiver)
            atm_emission_data[i_receiver] = atmospheric_model.emission_temperature[:, :, antenna_idx]

            # Calculate receiver temperature for this specific receiver
            rec_temp_model = ReceiverTemperature(receiver, self.receiver_models_dir)
            rec_temp_data[i_receiver] = rec_temp_model(freq_MHz)  # Shape: (n_freq,)

        # Pre-calculate beam solid angles (frequency-dependent only, same for all antennas)
        beam_solid_angle_HH = beam_model.get_beam_solid_angle_at_freq(
            frequency_MHz=freq_MHz,
            polarization='HH'
        )  # Shape: (n_freq,)
        beam_solid_angle_VV = beam_model.get_beam_solid_angle_at_freq(
            frequency_MHz=freq_MHz,
            polarization='VV'
        )  # Shape: (n_freq,)

        # Initialize model_components to store receiver-independent data
        self.model_components = {}

        # Loop over periods FIRST (outer loop)
        for period in calibrator_validated_periods:
            # Get calibrator name for this period (from ExtractCalibratorsPlugin results)
            calibrator_name = calibrator_names[period]

            # Calculate point source flux and position for THIS period (once per period, not per receiver)
            # There is also a catalogue in the data folder with all point sources above 1Jy. One could also use that 
            # to check for other sources next to the pointing. Note also that the spectral index in the
            # catalogue and the one used here is slightly different.
            point_source_flux, ra_deg, dec_deg = get_point_source_model(calibrator_name, freq)
            # point_source_flux shape: (n_freq,)

            # Convert RA/Dec to Az/El for all times (once per period)
            source_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
            altaz_frame = AltAz(obstime=times, location=location)
            source_altaz = source_coord.transform_to(altaz_frame)
            az_source_all = source_altaz.az.deg  # Shape: (n_time,)
            el_source_all = source_altaz.alt.deg  # Shape: (n_time,)

            # Get dump indices for this period
            dump_indices_period = calibrator_dump_indices[period]

            # Boolean mask for this period
            select = np.isin(dumps, dump_indices_period)

            # Subset median coordinates for this period (same for all receivers)
            az_median_period = median_az_all[select]  # (n_dumps_period,)
            el_median_period = median_el_all[select]  # (n_dumps_period,)
            az_source_period = az_source_all[select]  # (n_dumps_period,)
            el_source_period = el_source_all[select]  # (n_dumps_period,)

            # Pre-calculate beam gain for both polarizations using median pointing
            # This avoids recalculating for each receiver (32x speedup)
            beam_gain_HH = beam_model.get_beam_gain(
                az_pointing=az_median_period,
                el_pointing=el_median_period,
                az_source=az_source_period,
                el_source=el_source_period,
                frequency_MHz=freq_MHz,
                polarization='HH'
            )  # Shape: (n_dumps_period, n_freq)

            beam_gain_VV = beam_model.get_beam_gain(
                az_pointing=az_median_period,
                el_pointing=el_median_period,
                az_source=az_source_period,
                el_source=el_source_period,
                frequency_MHz=freq_MHz,
                polarization='VV'
            )  # Shape: (n_dumps_period, n_freq)

            # Pre-calculate spillover temperature for both polarizations
            # Uses median elevation (same for all antennas in calibrator tracking)
            spillover_temp_HH = spillover_model.get_temperature(
                elevation=el_median_period,
                frequency_MHz=freq_MHz,
                polarization='HH'
            )  # Shape: (n_dumps_period, n_freq)

            spillover_temp_VV = spillover_model.get_temperature(
                elevation=el_median_period,
                frequency_MHz=freq_MHz,
                polarization='VV'
            )  # Shape: (n_dumps_period, n_freq)

            # Calculate point source temperature for both polarizations
            # T = (λ² × 10^-26 / (2 k_B)) × S_ν[Jy] × G_beam / Ω_beam
            conversion_factor = (wavelength_m**2 * 1e-26) / (2 * k_B)  # [K·sr/Jy], shape (n_freq,)
            point_source_temp_HH = (
                conversion_factor * point_source_flux * beam_gain_HH / beam_solid_angle_HH
            )  # Shape: (n_dumps_period, n_freq)
            point_source_temp_VV = (
                conversion_factor * point_source_flux * beam_gain_VV / beam_solid_angle_VV
            )  # Shape: (n_dumps_period, n_freq)

            # Initialize model_components for this period (once, outside receiver loop)
            self.model_components[period] = {
                'calibrator': calibrator_name,
                'dump_indices': dump_indices_period,
                'gain': [],  # Will be filled in gather_and_set_result
                'beam_gain_HH': beam_gain_HH,  # (n_dumps, n_freq) - same for all receivers
                'beam_gain_VV': beam_gain_VV,  # (n_dumps, n_freq) - same for all receivers
                'temperatures': {
                    'atmospheric': [],  # Will be (n_dumps, n_freq, n_receivers)
                    'point_source_HH': point_source_temp_HH,  # (n_dumps, n_freq)
                    'point_source_VV': point_source_temp_VV,  # (n_dumps, n_freq)
                    'receiver': [],  # Will be (n_freq, n_receivers)
                    'spillover_HH': spillover_temp_HH,  # (n_dumps, n_freq)
                    'spillover_VV': spillover_temp_VV  # (n_dumps, n_freq)
                }
            }

            # Now loop over receivers for this period (inner loop)
            for i_receiver, receiver in enumerate(track_data.receivers):

                # Determine polarization and select pre-calculated values
                polarization = 'HH' if receiver.polarisation == 'h' else 'VV'
                spillover_temp_period = spillover_temp_HH if polarization == 'HH' else spillover_temp_VV
                point_source_temp_period = point_source_temp_HH if polarization == 'HH' else point_source_temp_VV

                # Get pre-extracted visibility, flags, atmospheric emission, and receiver temperature
                visibility_recv = visibility_data[i_receiver]
                initial_flag_recv = flag_data[i_receiver]
                atm_emission_recv = atm_emission_data[i_receiver]
                rec_temp_recv = rec_temp_data[i_receiver]  # Shape: (n_freq,), constant in time

                # Slice data to this specific period
                vis_period = visibility_recv.squeeze[select]  # (n_dumps_period, n_freq)
                flag_period = initial_flag_recv.squeeze[select]  # (n_dumps_period, n_freq)
                atm_period = atm_emission_recv[select, :]  # (n_dumps_period, n_freq)

                # Store receiver-specific data only
                self.model_components[period]['temperatures']['atmospheric'].append(atm_period)
                self.model_components[period]['temperatures']['receiver'].append(rec_temp_recv)

                yield (period, calibrator_name, vis_period, flag_period,
                       freq, atm_period, point_source_temp_period, rec_temp_recv,
                       spillover_temp_period)


    def run_job(self, anything: tuple) -> dict:
        """
        Calculate gain solution for one (receiver, period) combination.

        :param anything: Tuple from map() containing data for one receiver-period combo
        :return: Dictionary containing gain solution and metadata for reconstruction
        """
        (period, calibrator_name, vis_period, flag_period,
         freq, atm_period, point_source_temp_period, rec_temp_recv,
         spillover_temp_period) = anything

        # Verify all temperature components are available
        # Find frequency channel closest to 750 MHz (avoids band edges)
        freq_MHz = freq / 1e6  # Convert Hz to MHz
        idx_750 = np.argmin(np.abs(freq_MHz - 750.0))

        # Calculate gain using masked array approach
        # Total model temperature (rec_temp_recv broadcasts automatically)
        model_total = atm_period + point_source_temp_period + rec_temp_recv + spillover_temp_period

        # Create masked arrays using flags
        vis_masked = ma.masked_array(vis_period, mask=flag_period)
        model_masked = ma.masked_array(model_total, mask=flag_period)

        # Remove time average per frequency (axis=0 is time)
        vis_mean = ma.mean(vis_masked, axis=0)  # Shape: (n_freq,)
        vis_zeromean = vis_masked - vis_mean  # Broadcasting automatic

        model_mean = ma.mean(model_masked, axis=0)  # Shape: (n_freq,)
        model_zeromean = model_masked - model_mean  # Broadcasting automatic

        # Calculate gain per frequency (sum over time axis=0)
        numerator = ma.sum(vis_zeromean * model_zeromean, axis=0)  # Shape: (n_freq,)
        denominator = ma.sum(model_zeromean**2, axis=0)  # Shape: (n_freq,)

        # Calculate gain with edge case handling (gain=0 where denominator too small)
        threshold = 1e-10
        gain_per_freq = ma.where(
            ma.abs(denominator) > threshold,
            numerator / denominator,
            0.0  # Set to 0.0 for edge cases
        )  # Shape: (n_freq,) - masked array, constant in time

        # Print diagnostic values at 750 MHz
        print(f"Period {period}, Calibrator {calibrator_name}: "
              f"vis shape = {vis_period.shape}, "
              f"Maximum values at 750 MHz: "
              f"atm = {np.max(atm_period[:, idx_750]):.2f} K, "
              f"point_source = {np.max(point_source_temp_period[:, idx_750]):.2e} K, "
              f"rec_temp = {rec_temp_recv[idx_750]:.2f} K, "
              f"spillover = {np.max(spillover_temp_period[:, idx_750]):.2f} K, "
              f"gain = {gain_per_freq[idx_750]:.6f}")

        # Return only period and gain (other components already stored in map())
        return {
            'period': period,
            'gain': gain_per_freq  # Shape: (n_freq,) - compact form, constant in time
        }

    def gather_and_set_result(self,
                              result_list: list[dict],
                              track_data: TimeOrderedData,
                              calibrator_validated_periods: list,
                              calibrator_dump_indices: dict,
                              calibrator_names: dict,
                              flag_name_list: list,
                              flag_report_writer: ReportWriter,
                              output_path: str,
                              block_name: str):
        """
        Combine gain solutions from all (receiver, period) combinations and set results.

        Result reconstruction: Each job returns a dictionary with gain for one (receiver, period).
        We need to reconstruct the full gain array (n_time, n_freq, n_receivers).

        :param result_list: List of dictionaries, one per (receiver, period) combination
        :param track_data: Time ordered data
        :param calibrator_validated_periods: Validated periods
        :param calibrator_dump_indices: Calibrator dump indices
        :param calibrator_names: Dict mapping periods to calibrator names
        :param flag_report_writer: Report writer
        :param output_path: Output path
        :param block_name: Block name
        """

        n_receivers = len(track_data.receivers)

        # Append gain solutions from result_list to self.model_components (already populated in map())
        # result_list is ordered as: [period0_recv0, period0_recv1, period1_recv0, period1_recv1, ...]
        for i_result, result_dict in enumerate(result_list):
            gain_period = result_dict['gain']  # Shape: (n_freq,)
            period = result_dict['period']

            # Determine receiver index from result ordering
 #           i_receiver = i_result % n_receivers

            # Append gain to model_components
            self.model_components[period]['gain'].append(gain_period)

        # Stack receiver-specific components along receiver axis
        for period in self.model_components:
            # Stack gain: list of (n_freq,) -> (n_freq, n_receivers)
            self.model_components[period]['gain'] = np.ma.stack(
                self.model_components[period]['gain'],
                axis=1
            )
            # Stack atmospheric: list of (n_dumps, n_freq) -> (n_dumps, n_freq, n_receivers)
            self.model_components[period]['temperatures']['atmospheric'] = np.stack(
                self.model_components[period]['temperatures']['atmospheric'],
                axis=2
            )
            # Stack receiver: list of (n_freq,) -> (n_freq, n_receivers)
            self.model_components[period]['temperatures']['receiver'] = np.stack(
                self.model_components[period]['temperatures']['receiver'],
                axis=1
            )

        # Write report
        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()

        # Count calibrators processed (from model_components)
        calibrators_processed = set([self.model_components[period]['calibrator']
                                     for period in self.model_components])

        lines = [
            '...........................',
            f'Running PointSourceCalibration0Plugin with MuSEEK version: {branch} ({commit})',
            f'Finished at {current_datetime.strftime("%Y-%m-%d %H:%M:%S")}',
            f'Calibrator periods processed: {calibrator_validated_periods}',
            f'Calibrators used: {", ".join(calibrators_processed)}',
            f'Number of receivers: {n_receivers}',
            f'Model components saved for {len(self.model_components)} periods'
        ]
        flag_report_writer.write_to_report(lines)

        # Create diagnostic plot
#        self._plot_coordinate_comparison(
#            track_data=track_data,
#            flag_name_list=flag_name_list,
#            output_path=output_path
#        )

        # Set results
        self.set_result(result=Result(location=ResultEnum.TRACK_DATA, result=track_data, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.MODEL_COMPONENTS, result=self.model_components, allow_overwrite=True))

        if self.do_store_context:
            context_file_name = 'point_source_calibration0_plugin.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=output_path)



    def _plot_coordinate_comparison(self,
                                    track_data: TimeOrderedData,
                                    flag_name_list: list,
                                    output_path: str):
        """
        Diagnostic plot comparing median coordinates (excluding flagged antennas)
        vs first available receiver coordinates for entire track data timeline.

        :param track_data: Time ordered data
        :param flag_name_list: List of flag names
        :param output_path: Directory to save plot
        """
        # Recalculate median coordinates using existing helper function
        median_az_all, median_el_all = \
            calculate_median_coordinates_excluding_flagged_antennas(track_data, flag_name_list)

        # Use the first available antenna in the dataset for comparison
        az_data = track_data.azimuth.squeeze  # Shape: (n_time, n_antennas)
        el_data = track_data.elevation.squeeze
        antenna_idx = 0

        # Get antenna name from track_data receivers (first H-pol receiver)
        first_receiver = track_data.receivers[0]
        antenna_name = first_receiver.name

        # Extract selected antenna's coordinates
        az_recv = az_data[:, antenna_idx]  # Shape: (n_time,)
        el_recv = el_data[:, antenna_idx]

        # Create time axis for x-axis (dump indices)
        n_time = len(median_az_all)
        time_indices = np.arange(n_time)

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Azimuth plot
        ax1.plot(time_indices, median_az_all,
                color='blue', linewidth=2, label='Median (excluding flagged)', linestyle='--')
        ax1.plot(time_indices, az_recv,
                color='red', linewidth=1, label=f'Receiver {antenna_name}', linestyle=':', alpha=0.7)
        ax1.set_xlabel('Time dump index')
        ax1.set_ylabel('Azimuth (degrees)')
        ax1.set_title(f'Azimuth: Median (excluding flagged antennas) vs Receiver {antenna_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Elevation plot
        ax2.plot(time_indices, median_el_all,
                color='blue', linewidth=2, label='Median (excluding flagged)', linestyle='--')
        ax2.plot(time_indices, el_recv,
                color='red', linewidth=1, label=f'Receiver {antenna_name}', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Time dump index')
        ax2.set_ylabel('Elevation (degrees)')
        ax2.set_title(f'Elevation: Median (excluding flagged antennas) vs Receiver {antenna_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(output_path, f'coordinate_comparison_median_vs_{antenna_name}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Coordinate comparison plot saved to: {plot_filename}')
        plt.close()
