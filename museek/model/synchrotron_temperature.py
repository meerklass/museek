import healpy as hp
import numpy as np
import pysm3
import astropy.units as u
from astropy.coordinates import SkyCoord


class SynchrotronTemperature:
    """
    Galactic synchrotron brightness temperature at a given telescope pointing.

    Uses a pysm3 sky model smoothed with a Gaussian beam. This is an approximation
    of the true MeerKAT beam, which is non-Gaussian and polarization-dependent.
    The result is receiver-independent when all antennas track the same source.

    Smoothed HEALPix maps are computed at a coarse frequency grid in __init__ and
    stored. get_temperature() then performs cheap sky interpolation (hp.get_interp_val)
    and frequency interpolation (np.interp) per period — no expensive spherical
    harmonic transforms at call time.
    """

    def __init__(self,
                 freq: np.ndarray,
                 model: str = 's1',
                 nside: int = 128,
                 fwhm_ref_deg: float = 1.68,
                 fwhm_ref_freq_MHz: float = 850.0,
                 freq_step_MHz: float = 20.0):
        """
        :param freq: Full frequency array in Hz, shape (n_freq,). Used to build the
            coarse grid and stored for interpolation in get_temperature().
        :param model: pysm3 preset string for synchrotron model, e.g. 's1'
        :param nside: HEALPix nside resolution for the sky map
        :param fwhm_ref_deg: Reference Gaussian FWHM in degrees at fwhm_ref_freq_MHz.
            Approximation of the true (non-Gaussian, polarization-dependent) MeerKAT beam.
        :param fwhm_ref_freq_MHz: Reference frequency in MHz for fwhm_ref_deg
        :param freq_step_MHz: Step size in MHz for the coarse frequency grid. Synchrotron
            varies smoothly with frequency, so 20 MHz steps introduce negligible error
            while reducing the number of spherical harmonic transforms ~200-fold.
        """
        self._freq_MHz = freq / 1e6

        # Build coarse frequency grid
        freq_MHz_min = self._freq_MHz[0]
        freq_MHz_max = self._freq_MHz[-1]
        self._coarse_freq_MHz = np.arange(freq_MHz_min, freq_MHz_max + freq_step_MHz, freq_step_MHz)
        n_coarse = len(self._coarse_freq_MHz)
        n_pix = hp.nside2npix(nside)

        print(f'Computing synchrotron model on {n_coarse} coarse frequencies '
              f'({freq_MHz_min:.0f}–{freq_MHz_max:.0f} MHz, step {freq_step_MHz:.0f} MHz)...', flush=True)

        sky = pysm3.Sky(nside=nside, preset_strings=[model])
        self._smoothed_maps = np.zeros((n_coarse, n_pix))

        for i, f_MHz in enumerate(self._coarse_freq_MHz):
            emission_map = sky.get_emission(f_MHz / 1e3 * u.GHz).value[0] / 1e6  # uK_RJ → K_RJ
            fwhm = fwhm_ref_deg * (fwhm_ref_freq_MHz / f_MHz) * u.deg
            self._smoothed_maps[i] = pysm3.apply_smoothing_and_coord_transform(emission_map, fwhm=fwhm)

        print('Synchrotron model ready.', flush=True)

    def get_temperature(self,
                        ra: np.ndarray,
                        dec: np.ndarray) -> np.ndarray:
        """
        Compute synchrotron brightness temperature at the given RA/Dec pointing.

        Uses pre-computed smoothed maps from __init__. Performs sky interpolation
        at the coarse grid, then frequency interpolation to full resolution.

        :param ra: Right ascension per dump in degrees, shape (n_dumps,)
        :param dec: Declination per dump in degrees, shape (n_dumps,)
        :return: Synchrotron temperature in K_RJ, shape (n_dumps, n_freq)
        """
        n_dumps = len(ra)
        n_coarse = len(self._coarse_freq_MHz)
        n_freq = len(self._freq_MHz)

        # Convert RA/Dec → Galactic → HEALPix coordinates (once, frequency-independent)
        pointing = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        galactic = pointing.galactic
        theta = np.radians(90.0 - galactic.b.deg)  # HEALPix colatitude
        phi = np.radians(galactic.l.deg)            # HEALPix longitude

        # Sky interpolation at coarse frequencies
        synch_coarse = np.zeros((n_dumps, n_coarse))
        for i in range(n_coarse):
            synch_coarse[:, i] = hp.get_interp_val(self._smoothed_maps[i], theta, phi)

        # Frequency interpolation to full grid
        synch_temp = np.zeros((n_dumps, n_freq))
        for i_dump in range(n_dumps):
            synch_temp[i_dump] = np.interp(self._freq_MHz, self._coarse_freq_MHz, synch_coarse[i_dump])

        return synch_temp
