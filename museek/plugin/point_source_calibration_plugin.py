import os
from typing import Generator

import numpy as np
import datetime
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


class PointSourceCalibrationPlugin(AbstractParallelJoblibPlugin):
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

        # Combine all flags EXCEPT 'noise_diode_on' (we need ND-firing dumps for calibration)
        result_array = np.zeros(track_data.flags.shape)
        for i, flag_name in enumerate(flag_name_list):
            if flag_name != 'noise_diode_on':
                result_array += track_data.flags._flags[i].get_array()
        result_array[result_array < self.flag_combination_threshold] = 0
        initial_flags = track_data.flags._flag_element_factory.create(
            array=np.asarray(result_array, dtype=bool)
        )

        freq = track_data.frequencies.squeeze  # Frequencies in Hz
        freq_MHz = freq / 1e6  # Convert to MHz for beam model
        dumps = np.array(track_data._dumps())  # Absolute dump indices in track_data

        # Calculate noise diode ratios for track data (needed for calibration)
        noise_diode = NoiseDiode(
            dump_period=track_data.dump_period,
            observation_log=track_data.obs_script_log
        )
        noise_diode_cycle_start_times = noise_diode._get_noise_diode_cycle_start_times(
            timestamps=track_data.timestamps
        )
        noise_diode_ratios = noise_diode._get_noise_diode_ratios(
            timestamps=track_data.timestamps,
            noise_diode_cycle_starts=noise_diode_cycle_start_times,
            dump_period=noise_diode._dump_period
        )  # Shape: (n_time,)

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
        times = Time(track_data.timestamps, format='unix')

        # Calculate atmospheric model once (before all loops)
        atmospheric_model = AtmosphericModel(track_data)

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
            antenna_idx = receiver.antenna_index(receivers=track_data.receivers)
            atm_emission_data[i_receiver] = atmospheric_model.emission_temperature[:, :, antenna_idx]

            # Calculate receiver temperature for this specific receiver
            rec_temp_model = ReceiverTemperature(receiver, self.receiver_models_dir)
            rec_temp_data[i_receiver] = rec_temp_model(freq_MHz)  # Shape: (n_freq,)

        # Loop over periods FIRST (outer loop)
        for period in calibrator_validated_periods:
            # Get calibrator name for this period (from ExtractCalibratorsPlugin results)
            calibrator_name = calibrator_names[period]

            # Calculate point source flux and position for THIS period (once per period, not per receiver)
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

            # Now loop over receivers for this period (inner loop)
            for i_receiver, receiver in enumerate(track_data.receivers):
                receiver_path = os.path.join(output_path, receiver.name)
                if not os.path.isdir(receiver_path):
                    os.makedirs(receiver_path)

                # Get pointing for this receiver's antenna
                az_pointing_all = track_data.azimuth.get(recv=i_receiver).squeeze  # (n_time,)
                el_pointing_all = track_data.elevation.get(recv=i_receiver).squeeze  # (n_time,)

                # Subset for this period
                az_pointing_period = az_pointing_all[select]  # (n_dumps_period,)
                el_pointing_period = el_pointing_all[select]  # (n_dumps_period,)
                az_source_period = az_source_all[select]  # (n_dumps_period,)
                el_source_period = el_source_all[select]  # (n_dumps_period,)

                # Determine polarization for beam calculation
                polarization = 'HH' if receiver.polarisation == 'h' else 'VV'

                # Calculate beam gain for this receiver and period
                # This is not very efficient because at the moment the beam is antenna independent. But if we want to calculate it
                # outside the receiver cycle, we will need to pick an antenna to get the pointing and if we pick an antenna that is not 
                # working it will be a problem. There are ways around that but for now I just decided to leave it inside the loop.
                # We could also use Katbeam for a faster calculation
                beam_gain_period = beam_model.get_beam_gain(
                    az_pointing=az_pointing_period,
                    el_pointing=el_pointing_period,
                    az_source=az_source_period,
                    el_source=el_source_period,
                    frequency_MHz=freq_MHz,
                    polarization=polarization
                )  # Shape: (n_dumps_period, n_freq)

                # Get beam solid angle for this polarization
                beam_solid_angle = beam_model.get_beam_solid_angle_at_freq(
                    frequency_MHz=freq_MHz,
                    polarization=polarization
                )  # Shape: (n_freq,)

                # Calculate point source temperature [K]
                # T = (λ² × 10^-26 / (2 k_B)) × S_ν[Jy] × G_beam / Ω_beam
                conversion_factor = (wavelength_m**2 * 1e-26) / (2 * k_B)  # [K·sr/Jy], shape (n_freq,)
                point_source_temp_period = (
                    conversion_factor * point_source_flux * beam_gain_period / beam_solid_angle
                )  # Shape: (n_dumps_period, n_freq)

                # Calculate spillover temperature [K]
                spillover_temp_period = spillover_model.get_temperature(
                    elevation=el_pointing_period,  # (n_dumps_period,)
                    frequency_MHz=freq_MHz,         # (n_freq,)
                    polarization=polarization       # 'HH' or 'VV'
                )  # Shape: (n_dumps_period, n_freq)

                # Get pre-extracted visibility, flags, atmospheric emission, and receiver temperature
                visibility_recv = visibility_data[i_receiver]
                initial_flag_recv = flag_data[i_receiver]
                atm_emission_recv = atm_emission_data[i_receiver]
                rec_temp_recv = rec_temp_data[i_receiver]  # Shape: (n_freq,), constant in time

                # Slice data to this specific period
                vis_period = visibility_recv.squeeze[select]  # (n_dumps_period, n_freq)
                flag_period = initial_flag_recv.squeeze[select]  # (n_dumps_period, n_freq)
                atm_period = atm_emission_recv[select, :]  # (n_dumps_period, n_freq)
                noise_diode_ratios_period = noise_diode_ratios[select]  # (n_dumps_period,)

                yield (receiver_path, period, calibrator_name, vis_period, flag_period,
                       freq, atm_period, point_source_temp_period, rec_temp_recv,
                       spillover_temp_period, noise_diode_ratios_period, dump_indices_period)

    def run_job(self, anything: tuple) -> dict:
        """
        Calculate gain solution for one (receiver, period) combination.

        :param anything: Tuple from map() containing data for one receiver-period combo
        :return: Dictionary containing gain solution and metadata for reconstruction
        """
        (receiver_path, period, calibrator_name, vis_period, flag_period,
         freq, atm_period, point_source_temp_period, rec_temp_recv,
         spillover_temp_period, noise_diode_ratios_period, dump_indices_period) = anything

        # Verify all temperature components are available
        print(f"Period {period}, Calibrator {calibrator_name}: "
              f"vis shape = {vis_period.shape}, "
              f"atm mean = {np.mean(atm_period):.2f} K, "
              f"point_source_temp mean = {np.mean(point_source_temp_period):.2e} K, "
              f"rec_temp mean = {np.mean(rec_temp_recv):.2f} K, "
              f"spillover_temp mean = {np.mean(spillover_temp_period):.2f} K, "
              f"noise_diode_ratios mean = {np.mean(noise_diode_ratios_period):.3f}")

        # TODO: Calculate gain = measured / (model + atm_period + point_source_temp_period + rec_temp_recv + spillover_temp_period + ...)
        # Note: rec_temp_recv has shape (n_freq,), all others have (n_dumps_period, n_freq)
        # Broadcasting handles rec_temp_recv automatically
        # For now, return ones as placeholder
        gain_solution_period = np.ones(vis_period.shape, dtype=complex)  # (n_dumps_period, n_freq)

        # Return dictionary with gain and metadata for reconstruction
        return {
            'period': period,
            'calibrator_name': calibrator_name,
            'dump_indices': dump_indices_period,
            'gain': gain_solution_period
        }

    def gather_and_set_result(self,
                              result_list: list[dict],
                              track_data: TimeOrderedData,
                              calibrator_validated_periods: list,
                              calibrator_dump_indices: dict,
                              calibrator_names: dict,
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
        # Get dimensions
        if track_data.visibility is not None:
            n_time = track_data.visibility.shape[0]
            n_freq = track_data.visibility.shape[1]
        else:
            # Fallback (should not happen after load_visibility_flags_weights)
            raise RuntimeError("track_data.visibility is None - data not loaded properly")

        n_receivers = len(track_data.receivers)

        # Initialize full gain solution array with ones
        gain_solution_array = np.ones((n_time, n_freq, n_receivers), dtype=complex)

        # Get absolute dump indices for track_data
        dumps = np.array(track_data._dumps())

        # Reconstruct gain solution from per-period results
        # result_list is ordered as: [period0_recv0, period0_recv1, period1_recv0, period1_recv1, ...]
        for i_result, result_dict in enumerate(result_list):
            gain_period = result_dict['gain']  # Shape: (n_dumps_period, n_freq)
            dump_indices_period = result_dict['dump_indices']

            # Determine receiver index from result ordering
            # Results are yielded as: period0->recv0, period0->recv1, period1->recv0, period1->recv1, ...
            i_receiver = i_result % n_receivers

            # Create boolean mask for this period's dumps
            select = np.isin(dumps, dump_indices_period)

            # Insert gain solution for this (receiver, period) into the full array
            gain_solution_array[select, :, i_receiver] = gain_period

        # TODO: Set gain solution on track_data
        # track_data.set_gain_solution(gain_solution_array, mask_array)

        # Write report
        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()

        # Count calibrators processed
        calibrators_processed = set([r['calibrator_name'] for r in result_list])

        lines = [
            '...........................',
            f'Running PointSourceCalibrationPlugin with MuSEEK version: {branch} ({commit})',
            f'Finished at {current_datetime.strftime("%Y-%m-%d %H:%M:%S")}',
            f'Calibrator periods processed: {calibrator_validated_periods}',
            f'Calibrators used: {", ".join(calibrators_processed)}',
            f'Number of receivers: {n_receivers}',
            f'Gain solution shape: {gain_solution_array.shape}'
        ]
        flag_report_writer.write_to_report(lines)

        # Set results
        self.set_result(result=Result(location=ResultEnum.TRACK_DATA, result=track_data, allow_overwrite=True))

        if self.do_store_context:
            context_file_name = 'point_source_calibration_plugin.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=output_path)
