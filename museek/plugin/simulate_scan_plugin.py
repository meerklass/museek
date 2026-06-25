"""
Plugin that simulates scan-data visibilities from physical models and replaces
``scan_data.visibility`` so the rest of the museek pipeline runs on the simulation
transparently.

Phase 1 (this version): a deterministic, unity-gain Time-Ordered Data simulation in
Kelvin. The antenna temperature per dump/channel/receiver is

    T_total = T_foreground(beam-convolved) + T_atmospheric + T_spillover + T_receiver + T_noise_diode

and the visibility is set to ``T_total`` (gain = 1). The diffuse Galactic foreground is
beam-convolved with the real MeerKLASS primary beam using Simeer's ``integrate_tod``; the
remaining components come from the existing museek/model temperature models. Gain, 1/f gain
noise, white noise, point sources and the HI signal are deferred to a later phase.
"""

import datetime
import os

import astropy.units as u
import healpy as hp
import numpy as np
import pysm3
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result

from museek.enums.result_enum import ResultEnum
from museek.model.atmospheric_opacity import AtmosphericModel
from museek.model.noise_diode_temperature import NoiseDiodeTemperature
from museek.model.receiver_temperature import ReceiverTemperature
from museek.model.spillover_temperature import SpilloverTemperature
from museek.data_element import DataElement
from museek.noise_diode import NoiseDiode
from museek.plugin.point_source_calibration_plugin import (
    calculate_median_coordinates_excluding_flagged_antennas,
)
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import git_version_info

from simeer import MeerKLASSBeam, integrate_tod


class SimulateScanPlugin(AbstractPlugin):
    """Simulate the scan-data antenna temperature and overwrite ``scan_data.visibility``."""

    def __init__(self,
                 beam_file_path: str,
                 receiver_models_dir: str,
                 noise_diodes_dir: str,
                 spillover_model_file: str,
                 synch_model: str = 's1',
                 synch_nside: int = 128,
                 n_sim_freq: int = 30,
                 disc_radius_deg: float = 8.0,
                 include_noise_diode: bool = True,
                 n_jobs: int = 1,
                 do_store_context: bool = False,
                 verbose: int = 0,
                 **kwargs):
        """
        :param beam_file_path: MeerKLASS primary beam .npz (the `aa_highres` array-average file)
        :param receiver_models_dir: directory with the receiver-temperature model files
        :param noise_diodes_dir: directory with the noise-diode-temperature model files
        :param spillover_model_file: spillover temperature data file
        :param synch_model: pysm3 preset string for the Galactic synchrotron foreground (e.g. 's1')
        :param synch_nside: HEALPix nside of the foreground sky cube
        :param n_sim_freq: number of beam channels (within the scan band) at which the foreground is
            beam-convolved; the result is interpolated onto the full scan frequency grid
        :param disc_radius_deg: HEALPix disc radius for Simeer's beam integration
        :param include_noise_diode: if True, inject the noise-diode temperature at the
            `noise_diode_on` dumps (scaled by the diode duty cycle)
        :param n_jobs: joblib workers for Simeer's `integrate_tod`
        :param do_store_context: if True, store the context to disc after running
        :param verbose: verbosity
        """
        super().__init__(**kwargs)
        self.beam_file_path = beam_file_path
        self.receiver_models_dir = receiver_models_dir
        self.noise_diodes_dir = noise_diodes_dir
        self.spillover_model_file = spillover_model_file
        self.synch_model = synch_model
        self.synch_nside = synch_nside
        self.n_sim_freq = n_sim_freq
        self.disc_radius_deg = disc_radius_deg
        self.include_noise_diode = include_noise_diode
        self.n_jobs = n_jobs
        self.do_store_context = do_store_context
        self.verbose = verbose
        self.report_file_name = 'flag_report.md'

    def set_requirements(self):
        self.requirements = [
            Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
            Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer'),
        ]

    def run(self, scan_data: TimeOrderedData, output_path: str, flag_report_writer: ReportWriter):
        if scan_data.visibility is None:
            scan_data.load_visibility_flags_weights(polars='auto')

        n_time, n_freq, n_recv = scan_data.visibility.shape
        freq_scan_MHz = scan_data.frequencies.squeeze / 1e6  # (n_freq,)

        # --- observatory geometry, pointing and LST ---
        antenna0 = scan_data.antennas[0]
        location = EarthLocation(lat=antenna0.ref_observer.lat * u.rad,
                                 lon=antenna0.ref_observer.lon * u.rad,
                                 height=antenna0.ref_observer.elevation * u.m)
        lat_deg = float(np.degrees(antenna0.ref_observer.lat))
        times = Time(scan_data.timestamps.squeeze, format='unix', location=location)
        lst_deg = times.sidereal_time('apparent').deg  # (n_time,)
        median_az, median_el = calculate_median_coordinates_excluding_flagged_antennas(scan_data)

        # --- beam + foreground frequency grid (beam channels inside the scan band) ---
        beam = MeerKLASSBeam(self.beam_file_path, antenna='array_average', polarizations=('HH', 'VV'))
        in_band = (beam.freq_MHz >= freq_scan_MHz.min()) & (beam.freq_MHz <= freq_scan_MHz.max())
        band_idx = np.where(in_band)[0]
        sim_idx = band_idx[np.linspace(0, len(band_idx) - 1, self.n_sim_freq).astype(int)]
        sim_freq_MHz = beam.freq_MHz[sim_idx]  # (n_sim,) exact beam channels

        # --- unsmoothed pysm3 foreground cube in equatorial coords ---
        sky_cube = self._build_foreground_cube(sim_freq_MHz)  # (n_sim, n_pix)

        # --- beam-convolved foreground sky TOD per polarisation, interpolated onto scan freqs ---
        foreground = {}
        for pol in ('HH', 'VV'):
            sky_tod = integrate_tod(
                lst_deg_list=lst_deg, az_deg_list=median_az, el_deg_list=median_el,
                lat_deg=lat_deg, beam=beam, sky_maps=sky_cube, freq_MHz=sim_freq_MHz,
                disc_radius_deg=self.disc_radius_deg, polarization=pol, n_jobs=self.n_jobs,
            )  # (n_sim, n_time)
            fg = np.empty((n_time, n_freq))
            for t in range(n_time):
                fg[t] = np.interp(freq_scan_MHz, sim_freq_MHz, sky_tod[:, t])
            foreground[pol] = fg

        # --- elevation-dependent components ---
        atm_emission = AtmosphericModel(scan_data).emission_temperature  # (n_time, n_freq, n_ant)
        spillover = SpilloverTemperature(self.spillover_model_file)
        noise_on, nd_duty = self._noise_diode_timing(scan_data, n_time)

        # --- assemble per receiver ---
        vis = np.zeros((n_time, n_freq, n_recv))
        for i_recv, receiver in enumerate(scan_data.receivers):
            pol = 'HH' if receiver.polarisation.lower() == 'h' else 'VV'
            i_ant = scan_data.antenna_index_of_receiver(receiver)

            temperature = foreground[pol].copy()                                  # foreground
            temperature += atm_emission[:, :, i_ant]                              # atmospheric
            temperature += spillover.get_temperature(median_el, freq_scan_MHz, pol)  # spillover
            temperature += ReceiverTemperature(receiver, self.receiver_models_dir)(freq_scan_MHz)[np.newaxis, :]
            if self.include_noise_diode:
                t_nd = NoiseDiodeTemperature(receiver, self.noise_diodes_dir)(freq_scan_MHz)  # (n_freq,)
                temperature += (noise_on * nd_duty)[:, np.newaxis] * t_nd[np.newaxis, :]
            vis[:, :, i_recv] = temperature

        # --- inject simulated visibility (unity gain: counts == Kelvin) ---
        # `vis` is already on the scan time axis, so build the DataElement directly: the scan
        # element factory would otherwise re-slice a full-length array by scan dumps and overrun.
        scan_data.visibility = DataElement(array=vis)

        branch, commit = git_version_info()
        flag_report_writer.write_to_report([
            '...........................',
            f'Running SimulateScanPlugin with MuSEEK version: {branch} ({commit})',
            f'Finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'Replaced scan visibility with simulated antenna temperature, shape {vis.shape}',
            f'Foreground: pysm3 {self.synch_model} beam-convolved (Simeer) at {self.n_sim_freq} channels',
            f'Noise diode injected: {self.include_noise_diode} ({int(noise_on.sum())} on-dumps, duty {nd_duty:.3f})',
        ])
        if self.verbose:
            print(f'SimulateScanPlugin: vis range [{vis.min():.2f}, {vis.max():.2f}] K, '
                  f'median {np.median(vis):.2f} K', flush=True)

        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data, allow_overwrite=True))

        if self.do_store_context:
            self.store_context_to_disc(context_file_name='simulate_scan_plugin.pickle',
                                       context_directory=output_path)

    def _build_foreground_cube(self, sim_freq_MHz: np.ndarray) -> np.ndarray:
        """Unsmoothed pysm3 Galactic foreground HEALPix cube, rotated to equatorial (ICRS).

        Returns shape ``(n_sim_freq, n_pix)`` in K_RJ. Unsmoothed because Simeer applies the
        real beam; rotated G->C because Simeer's sky cube must be in equatorial coordinates.
        """
        sky = pysm3.Sky(nside=self.synch_nside, preset_strings=[self.synch_model])
        galactic_to_equatorial = hp.Rotator(coord=['G', 'C'])
        cube = np.zeros((len(sim_freq_MHz), hp.nside2npix(self.synch_nside)))
        for i, f_MHz in enumerate(sim_freq_MHz):
            emission = sky.get_emission(f_MHz / 1e3 * u.GHz).value[0] / 1e6  # uK_RJ -> K_RJ (Galactic)
            cube[i] = galactic_to_equatorial.rotate_map_pixel(emission)
        return cube

    def _noise_diode_timing(self, scan_data: TimeOrderedData, n_time: int) -> tuple[np.ndarray, float]:
        """Return the per-dump noise-diode-on boolean and the diode duty cycle (duration/dump_period)."""
        flag_names = scan_data.flags.flag_names
        if 'noise_diode_on' in flag_names:
            i = flag_names.index('noise_diode_on')
            noise_on = np.asarray(scan_data.flags.get(freq=0, recv=0).array[i]).squeeze().astype(bool)
        else:
            noise_on = np.zeros(n_time, dtype=bool)
        noise_diode = NoiseDiode(dump_period=scan_data.dump_period,
                                 observation_log=scan_data.obs_script_log)
        duty = float(noise_diode.duration / scan_data.dump_period)
        return noise_on, duty
