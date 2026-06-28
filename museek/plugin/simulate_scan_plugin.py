"""
Plugin that simulates scan-data visibilities from physical models and replaces
``scan_data.visibility`` so the rest of the museek pipeline runs on the simulation
transparently.

The antenna temperature per dump/channel/receiver is

    T_total = T_foreground(beam-convolved) + T_CMB + T_atmospheric + T_spillover + T_receiver
              + T_noise_diode + T_point_sources + T_HI

and the visibility is ``vis = gain * (1 + delta_g_1f) * T_total * (1 + white)``. The diffuse Galactic
foreground is beam-convolved with the real MeerKLASS primary beam using Simeer's ``integrate_tod``;
the additive temperatures come from the existing museek/model temperature models. Optional components:
point sources (catalog-driven, two methods), an HI signal (limTOD Gaussian-field mock or a user cube,
Gaussian-beam smoothed), the gain (a saved calibrator gain via ``ReadCalibratorGainsPlugin`` and/or a
synthetic smooth-polynomial x standing-wave term), 1/f gain noise (``limTOD.flicker_model.sim_noise``)
and radiometer white noise. With everything off, ``vis = T_total`` (Kelvin, unity gain).
"""

import datetime
import os

import astropy.units as u
import healpy as hp
import numpy as np
import pysm3
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result

from museek.enums.result_enum import ResultEnum
from museek.model.atmospheric_opacity import AtmosphericModel
from museek.model.noise_diode_temperature import NoiseDiodeTemperature
from museek.model.receiver_temperature import ReceiverTemperature
from museek.model.spillover_temperature import SpilloverTemperature
from museek.model.bandpass_model import BandpassModel
from museek.model.primary_beam import PrimaryBeam
from museek.model import point_source_catalog as psc
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
                 include_cmb: bool = True,
                 point_source_method: str | None = None,
                 point_source_catalog: str | None = None,
                 primary_beam_file: str | None = None,
                 point_source_min_flux_Jy: float = 1.0,
                 point_source_flux_cut_freq_MHz: float = 800.0,
                 point_source_radius_deg: float = 6.0,
                 gain_smooth_poly: list | None = None,
                 gain_standing_waves: list | None = None,
                 hi_method: str | None = None,
                 hi_n_freq: int = 256,
                 hi_target_rms_mK: float | None = 0.3,
                 hi_gaussian_params: dict | None = None,
                 hi_file: str | None = None,
                 oneoverf_params: dict | None = None,
                 include_white_noise: bool = False,
                 noise_seed: int | None = None,
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
        :param include_cmb: if True, add the CMB monopole as a Rayleigh-Jeans brightness temperature
            (a uniform sky, so it beam-convolves to a constant ~2.7 K added per channel)
        :param gain_smooth_poly: optional polynomial coefficients [c0, c1, c2, ...] for the smooth
            multiplicative gain factor `c0 + c1*x + c2*x^2 + ...` in normalised frequency
            `x = (f - f_mid) / (f_halfspan)` in [-1, 1]; `c0` sets the absolute level (counts/K).
            `None` -> no smooth factor (1).
        :param gain_standing_waves: optional list of `(displacement_m, amplitude, phase)` standing
            waves, evaluated with `BandpassModel._sinus` (wavelength = 2*displacement); contributes a
            factor `(1 + sum of sinusoids)`. `None` -> no standing-wave factor (1).
        :param hi_method: HI signal source: `None` (no HI) | `'limtod'` (a Gaussian-field mock from
            `limTOD.sky_model.generate_gaussian_field`) | `'file'` (a user-supplied HI cube .npz with
            keys `maps` (n_freq, n_pix) HEALPix in K and `freq_MHz`). The HI is Gaussian-beam smoothed
            (not the real beam) and sampled at the pointing, then interpolated onto the scan freq grid.
        :param hi_n_freq: number of channels at which the `'limtod'` HI is generated (then interpolated
            onto the scan grid); chosen to sample the field's frequency correlation length.
        :param hi_target_rms_mK: if set, the HI cube is rescaled to this map RMS in mK (overrides the
            generator's `amp`); `None` -> use the amplitude from `hi_gaussian_params`.
        :param hi_gaussian_params: kwargs for `generate_gaussian_field` (e.g. `alpha`, `beta`, `xi`,
            `nu_ref`, `ell_ref`, `seed`); defaults to a "vaguely cosmological" field.
        :param hi_file: path to the HI cube .npz for `hi_method='file'`.
        :param oneoverf_params: if set, add 1/f gain noise as a multiplicative `(1 + delta_g(t))` per
            receiver, with `delta_g` drawn from `limTOD.flicker_model.sim_noise`; dict of its kwargs
            `f0, fc, alpha, white_n_variance` (f0/fc are angular frequencies). `None` -> no 1/f noise.
        :param include_white_noise: if True, add radiometer white noise as `(1 + N(0,1)/sqrt(dnu*tau))`
            per (time, freq, receiver), with `dnu` the channel width and `tau` the dump period.
        :param noise_seed: base seed for the 1/f and white noise (reproducibility).
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
        self.include_cmb = include_cmb
        self.point_source_method = point_source_method  # None | 'healpix' | 'primary_beam'
        self.point_source_catalog = point_source_catalog
        self.primary_beam_file = primary_beam_file or beam_file_path
        self.gain_smooth_poly = gain_smooth_poly
        self.gain_standing_waves = gain_standing_waves
        self.hi_method = hi_method  # None | 'limtod' | 'file'
        self.hi_n_freq = hi_n_freq
        self.hi_target_rms_mK = hi_target_rms_mK
        self.hi_gaussian_params = hi_gaussian_params or {}
        self.hi_file = hi_file
        self.oneoverf_params = oneoverf_params
        self.include_white_noise = include_white_noise
        self.noise_seed = noise_seed
        self.point_source_min_flux_Jy = point_source_min_flux_Jy
        self.point_source_flux_cut_freq_MHz = point_source_flux_cut_freq_MHz
        self.point_source_radius_deg = point_source_radius_deg
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
        median_el = np.clip(median_el, 0.0, 90.0)  # guard against stowed-antenna artefacts

        # --- point-source catalog selection (shared by both methods) ---
        ps_catalog = None
        if self.point_source_method is not None:
            ps_catalog = self._select_point_sources(median_az, median_el, times, location)

        # --- beam + foreground frequency grid (beam channels inside the scan band) ---
        beam = MeerKLASSBeam(self.beam_file_path, antenna='array_average', polarizations=('HH', 'VV'))
        in_band = (beam.freq_MHz >= freq_scan_MHz.min()) & (beam.freq_MHz <= freq_scan_MHz.max())
        band_idx = np.where(in_band)[0]
        sim_idx = band_idx[np.linspace(0, len(band_idx) - 1, self.n_sim_freq).astype(int)]
        sim_freq_MHz = beam.freq_MHz[sim_idx]  # (n_sim,) exact beam channels

        # --- unsmoothed pysm3 foreground cube in equatorial coords ---
        sky_cube = self._build_foreground_cube(sim_freq_MHz)  # (n_sim, n_pix)

        # --- method A: rasterise point sources into the cube so Simeer convolves them ---
        if self.point_source_method == 'healpix' and len(ps_catalog['ra_deg']):
            self._add_point_sources_to_cube(sky_cube, sim_freq_MHz, ps_catalog)

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
        del sky_cube

        # --- HI signal (uses the Simeer beam for the FWHM); computed before PrimaryBeam so the two
        #     beam objects (~2 GB each) never coexist in memory ---
        hi_tod = None
        if self.hi_method is not None:
            radec = SkyCoord(az=median_az * u.deg, alt=median_el * u.deg,
                             frame=AltAz(obstime=times, location=location)).icrs
            hi_tod = self._build_hi_tod(freq_scan_MHz, np.radians(90.0 - radec.dec.deg),
                                        np.radians(radec.ra.deg % 360.0), beam)
        del beam  # free the Simeer beam before loading PrimaryBeam

        # --- method B: per-pol point-source TOD via the primary beam (exact positions) ---
        point_source = {'HH': None, 'VV': None}
        if self.point_source_method == 'primary_beam' and len(ps_catalog['ra_deg']):
            primary_beam = PrimaryBeam(self.primary_beam_file)
            valid_dumps = ~self._antenna_flagged_dumps(scan_data, n_time)  # skip stowed/flagged pointings
            for pol in ('HH', 'VV'):
                point_source[pol] = self._point_source_tod(
                    ps_catalog, freq_scan_MHz, median_az, median_el, times, location, pol,
                    primary_beam, valid_dumps)
            del primary_beam  # free PrimaryBeam before the per-receiver loop

        # --- elevation-dependent components ---
        atm_emission = AtmosphericModel(scan_data).emission_temperature  # (n_time, n_freq, n_ant)
        spillover = SpilloverTemperature(self.spillover_model_file)
        noise_on, nd_duty = self._noise_diode_timing(scan_data, n_time)

        # --- CMB monopole (Rayleigh-Jeans brightness temperature); a uniform sky beam-convolves to a constant ---
        t_cmb = None
        if self.include_cmb:
            h, k_b, t_cmb_kelvin = 6.62607015e-34, 1.380649e-23, 2.725
            x = h * (freq_scan_MHz * 1e6) / (k_b * t_cmb_kelvin)
            t_cmb = t_cmb_kelvin * x / np.expm1(x)  # (n_freq,) RJ-corrected, ~2.7 K across the band

        # --- gain: read calibrator gain (if ReadCalibratorGainsPlugin ran) x synthetic poly/standing-wave ---
        read_gain_spectrum = None  # (n_freq, n_recv) per-receiver gain, constant in time
        if scan_data.gain_solution is not None:
            read_gain_spectrum = np.asarray(scan_data.gain_solution.array)[0]
        synth_gain = self._synthetic_gain(freq_scan_MHz)  # (n_freq,) or None

        # --- noise: 1/f gain fluctuation (limTOD) + radiometer white noise ---
        if self.noise_seed is not None:
            np.random.seed(self.noise_seed)
        delta_g = None  # (n_recv, n_time) fractional gain fluctuation, common-mode across frequency
        if self.oneoverf_params is not None:
            from limTOD.flicker_model import sim_noise
            delta_g = sim_noise(time_list=scan_data.timestamps.squeeze, n_samples=n_recv,
                                **self.oneoverf_params)
        radiometer = None  # fractional white-noise level 1/sqrt(dnu*tau)
        if self.include_white_noise:
            dnu_Hz = float(np.abs(np.median(np.diff(scan_data.frequencies.squeeze))))
            radiometer = 1.0 / np.sqrt(dnu_Hz * float(scan_data.dump_period))

        # --- assemble per receiver (float32 visibility to keep memory down) ---
        vis = np.zeros((n_time, n_freq, n_recv), dtype=np.float32)
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
            if point_source[pol] is not None:                                     # method B point sources
                temperature += point_source[pol]
            if hi_tod is not None:                                                # HI signal (beam-smoothed)
                temperature += hi_tod
            if t_cmb is not None:                                                  # CMB monopole (RJ)
                temperature += t_cmb[np.newaxis, :]

            # gain[f] = read_gain[f,recv] (if present) x synthetic factor (if configured); else 1
            gain_recv = np.ones(n_freq)
            if read_gain_spectrum is not None:
                gain_recv = gain_recv * read_gain_spectrum[:, i_recv]
            if synth_gain is not None:
                gain_recv = gain_recv * synth_gain

            vis_recv = gain_recv[np.newaxis, :] * temperature                      # vis = gain * T_total
            if delta_g is not None:                                                # x (1 + 1/f gain noise)
                vis_recv = vis_recv * (1.0 + delta_g[i_recv][:, np.newaxis])
            if radiometer is not None:                                             # x (1 + radiometer white)
                vis_recv = vis_recv * (1.0 + radiometer * np.random.standard_normal((n_time, n_freq)))
            vis[:, :, i_recv] = vis_recv.astype(np.float32)
        del foreground, atm_emission, point_source  # free the large per-pol intermediates

        # --- inject simulated visibility (vis = gain * T_total; gain == 1 -> counts == Kelvin) ---
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
            f'Point sources: method={self.point_source_method} '
            f'({0 if ps_catalog is None else len(ps_catalog["ra_deg"])} sources >= {self.point_source_min_flux_Jy} Jy)',
            f'Gain: read_calibrator={read_gain_spectrum is not None}, '
            f'synth_poly={self.gain_smooth_poly is not None}, '
            f'synth_standing_waves={0 if self.gain_standing_waves is None else len(self.gain_standing_waves)}',
            f'HI: method={self.hi_method}'
            + ('' if hi_tod is None else f' (rms {np.std(hi_tod) * 1e3:.3f} mK at the pointing)'),
            f'Noise: 1/f gain={self.oneoverf_params is not None}'
            + ('' if delta_g is None else f' (rms {delta_g.std() * 100:.3f}%)')
            + f', white_radiometer={self.include_white_noise}'
            + ('' if radiometer is None else f' ({radiometer * 100:.3f}%/sample)'),
        ])
        if self.verbose:
            unit = 'counts' if (read_gain_spectrum is not None or synth_gain is not None) else 'K'
            print(f'SimulateScanPlugin: vis range [{vis.min():.2f}, {vis.max():.2f}] {unit}, '
                  f'median {np.median(vis):.2f} {unit}', flush=True)

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

    def _build_hi_tod(self, freq_scan_MHz: np.ndarray, theta_pointing: np.ndarray,
                      phi_pointing: np.ndarray, beam) -> np.ndarray:
        """HI antenna-temperature TOD `(n_time, n_freq)`.

        Builds an HI HEALPix cube (limTOD Gaussian-field mock or a user-supplied .npz), Gaussian-beam
        smooths each channel and samples it at the pointing, then interpolates onto the scan grid. Uses
        the Gaussian-beam approximation (cheap, adequate for a faint mock): the HI's frequency
        statistics are exact, only the spatial beam shape is approximated.
        """
        nside = self.synch_nside
        if self.hi_method == 'limtod':
            from limTOD.sky_model import generate_gaussian_field
            hi_freqs = np.linspace(freq_scan_MHz.min(), freq_scan_MHz.max(), self.hi_n_freq)
            params = {'amp': 1.0, 'alpha': -1.0, 'beta': 1.0, 'xi': 0.01}  # "vaguely cosmological"
            params.update(self.hi_gaussian_params)
            cube = generate_gaussian_field(freqs=hi_freqs, nside=nside, **params)  # (n_hi, n_pix) K
        elif self.hi_method == 'file':
            data = np.load(self.hi_file)
            cube = np.asarray(data['maps'], dtype=float)
            hi_freqs = np.asarray(data['freq_MHz'], dtype=float)
            if hp.npix2nside(cube.shape[1]) != nside:
                cube = np.array([hp.ud_grade(m, nside) for m in cube])
        else:
            raise ValueError(f"Unknown hi_method '{self.hi_method}'; use None, 'limtod' or 'file'.")

        if self.hi_target_rms_mK is not None and cube.std() > 0:
            cube = cube * (self.hi_target_rms_mK * 1e-3 / cube.std())

        # per-channel Gaussian-beam FWHM from the beam solid angle; smooth and sample at the pointing
        omega_b = 0.5 * (beam.beam_solid_angle('HH') + beam.beam_solid_angle('VV'))
        fwhm_rad = np.interp(hi_freqs, beam.freq_MHz, np.sqrt(omega_b / 1.133))
        n_time = len(theta_pointing)
        hi_at_hifreq = np.empty((n_time, len(hi_freqs)))
        for k in range(len(hi_freqs)):
            smoothed = hp.smoothing(cube[k], fwhm=float(fwhm_rad[k]))
            hi_at_hifreq[:, k] = hp.get_interp_val(smoothed, theta_pointing, phi_pointing)

        hi_tod = np.empty((n_time, len(freq_scan_MHz)))
        for t in range(n_time):
            hi_tod[t] = np.interp(freq_scan_MHz, hi_freqs, hi_at_hifreq[t])
        return hi_tod

    def _select_point_sources(self, az_p: np.ndarray, el_p: np.ndarray, times: Time,
                              location: EarthLocation) -> dict:
        """Load the catalog and keep sources within `point_source_radius_deg` of the pointing track."""
        cat = psc.load_catalog(self.point_source_catalog, min_flux_Jy=self.point_source_min_flux_Jy,
                               flux_cut_freq_Hz=self.point_source_flux_cut_freq_MHz * 1e6)
        step = max(1, len(az_p) // 400)  # subsample the track for the selection prefilter
        radec = SkyCoord(az=az_p[::step] * u.deg, alt=el_p[::step] * u.deg,
                         frame=AltAz(obstime=times[::step], location=location)).icrs
        sel = psc.select_near_track(cat, radec.ra.deg, radec.dec.deg, self.point_source_radius_deg)
        if self.verbose:
            print(f'SimulateScanPlugin: {len(sel)} point sources within '
                  f'{self.point_source_radius_deg} deg of the scan track '
                  f'(>= {self.point_source_min_flux_Jy} Jy)', flush=True)
        return {k: v[sel] for k, v in cat.items()}

    def _add_point_sources_to_cube(self, sky_cube: np.ndarray, sim_freq_MHz: np.ndarray, cat: dict):
        """Method A: add each source as a HEALPix pixel `T_pix = S lambda^2 / (2 k_B Omega_pix)`."""
        omega_pix = 4.0 * np.pi / hp.nside2npix(self.synch_nside)
        freq_Hz = sim_freq_MHz * 1e6
        flux = psc.flux_Jy(cat, freq_Hz)                                  # (n_src, n_sim)
        t_pix = psc.jy_to_kelvin(flux, freq_Hz[np.newaxis, :], omega_pix)  # (n_src, n_sim)
        pix = hp.ang2pix(self.synch_nside, np.radians(90.0 - cat['dec_deg']),
                         np.radians(cat['ra_deg'] % 360.0))
        for k in range(len(pix)):
            sky_cube[:, pix[k]] += t_pix[k]

    def _point_source_tod(self, cat: dict, freq_scan_MHz: np.ndarray, az_p: np.ndarray,
                          el_p: np.ndarray, times: Time, location: EarthLocation, pol: str,
                          primary_beam: PrimaryBeam, valid_dumps: np.ndarray) -> np.ndarray:
        """Method B: PrimaryBeam-convolved point-source antenna temperature, shape (n_time, n_freq).

        Only dumps in `valid_dumps` (not antenna-flagged) are evaluated -- there is no point computing
        the sky where the pointing is already flagged (e.g. the stowed/zenith excursion)."""
        n_time, n_freq = len(az_p), len(freq_scan_MHz)
        out = np.zeros((n_time, n_freq))
        freq_Hz = freq_scan_MHz * 1e6
        omega_b = primary_beam.get_beam_solid_angle_at_freq(freq_scan_MHz, pol)   # (n_freq,)
        flux = psc.flux_Jy(cat, freq_Hz)                                          # (n_src, n_freq)
        t_peak = psc.jy_to_kelvin(flux, freq_Hz[np.newaxis, :], omega_b[np.newaxis, :])  # (n_src, n_freq)

        az_p_rad, el_p_rad = np.radians(az_p), np.radians(el_p)
        for k in range(len(cat['ra_deg'])):
            altaz = SkyCoord(ra=cat['ra_deg'][k] * u.deg, dec=cat['dec_deg'][k] * u.deg).transform_to(
                AltAz(obstime=times, location=location))
            az_s, el_s = np.radians(altaz.az.deg), np.radians(altaz.alt.deg)
            sep = np.degrees(np.arccos(np.clip(
                np.sin(el_s) * np.sin(el_p_rad) + np.cos(el_s) * np.cos(el_p_rad) * np.cos(az_s - az_p_rad),
                -1.0, 1.0)))
            near = (sep < self.point_source_radius_deg) & valid_dumps
            if not near.any():
                continue
            gain = primary_beam.get_beam_gain(az_p[near], el_p[near],
                                              np.degrees(az_s[near]), np.degrees(el_s[near]),
                                              freq_scan_MHz, pol)  # (n_near, n_freq)
            out[near] += gain * t_peak[k][np.newaxis, :]
        return out

    def _synthetic_gain(self, freq_scan_MHz: np.ndarray) -> np.ndarray | None:
        """Synthetic gain factor `(n_freq,)`: smooth polynomial x `(1 + sum standing waves)`.

        Returns None if neither the polynomial nor the standing waves are configured (-> unity gain).
        The polynomial is evaluated in normalised frequency `x = (f - f_mid)/f_halfspan in [-1, 1]`
        (so the constant coefficient sets the absolute level); the standing waves reuse
        `BandpassModel._sinus` with wavelength = 2*displacement.
        """
        if self.gain_smooth_poly is None and self.gain_standing_waves is None:
            return None
        gain = np.ones(len(freq_scan_MHz))
        if self.gain_smooth_poly is not None:
            f_mid = 0.5 * (freq_scan_MHz[0] + freq_scan_MHz[-1])
            f_half = 0.5 * (freq_scan_MHz[-1] - freq_scan_MHz[0])
            x = (freq_scan_MHz - f_mid) / f_half
            gain = gain * np.polynomial.polynomial.polyval(x, self.gain_smooth_poly)
        if self.gain_standing_waves is not None:
            sw = sum(BandpassModel._sinus(freq_scan_MHz, (phase, amplitude, 2.0 * displacement_m))
                     for displacement_m, amplitude, phase in self.gain_standing_waves)
            gain = gain * (1.0 + sw)
        return gain

    def _antenna_flagged_dumps(self, scan_data: TimeOrderedData, n_time: int) -> np.ndarray:
        """Per-dump boolean: True where antenna_flagger flagged the pointing (e.g. stowed/zenith dumps)."""
        flag_names = scan_data.flags.flag_names
        flagged = np.zeros(n_time, dtype=bool)
        for name in ('elevation_flag', 'outlier_antenna_flag'):
            if name in flag_names:
                i = flag_names.index(name)
                flagged |= np.asarray(scan_data.flags.get(freq=0, recv=0).array[i]).squeeze().astype(bool)
        return flagged

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
