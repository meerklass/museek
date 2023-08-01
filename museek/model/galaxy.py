import os

import healpy
import numpy as np
import pysm3
from astropy import units
from astropy.coordinates import SkyCoord

from definitions import ROOT_DIR, MEGA
from museek.data_element import DataElement
from museek.factory.data_element_factory import DataElementFactory
from museek.model.abstract_model import AbstractModel


class Galaxy(AbstractModel):

    def __init__(self,
                 data_element_factory: DataElementFactory,
                 haslam_path: str = 'data/haslam408_dsds_Remazeilles2014.fits',
                 spectral_index_path: str = 'data/synch_beta.fits'):
        """
        Initialise with a `data_element_factory` and `file_name`.
        :param data_element_factory: to create `DataElement`s
        :param haslam_path: path to haslam map fits file
        :param spectral_index_path: path to spectral index fits file
        """
        super().__init__(data_element_factory=data_element_factory)
        self._haslam_path = haslam_path
        self._spectral_index_path = spectral_index_path

    def temperature(
            self,
            frequencies: DataElement,
            right_ascension: DataElement,
            declination: DataElement,
            nside: int
    ) -> DataElement:
        """
        Returns the galaxy model temperature as a `DataElement`
        :param frequencies: frequency values to model
        :param right_ascension: right ascension values to model
        :param declination: declination values to model
        :param nside: integer `healpix` resolution parameter, forwarded to `pysm3`
        :return: galaxy model temperature for the desired channels and coordinates
        """
        sky_coord = SkyCoord(ra=right_ascension.array * units.degree,
                             dec=declination.array * units.degree,
                             frame='icrs')
        theta = 90 - (sky_coord.galactic.b / units.degree).value
        phi = (sky_coord.galactic.l / units.degree).value
        result_array = self.galaxy_temperature_array(frequencies=frequencies.array, theta=theta, phi=phi, nside=nside)
        return self.data_element_factory.create(array=result_array)

    def galaxy_temperature_array(
            self,
            frequencies: np.ndarray,
            theta: np.ndarray,
            phi: np.ndarray,
            nside: int
    ) -> np.ndarray:
        """
        Extrapolates the haslam map to the desired frequency using the spectral index and returns the result.
        :param frequencies:
        :param theta: sky angle in degrees
        :param phi: sky angle in degrees
        :param nside: integer `healpix` resolution parameter, forwarded to `pysm3`
        :return: galaxy model as `numpy.ndarray`
        """
        model = self.interpolated_haslam(theta=theta, phi=phi)
        index = self.interpolated_spectral_index(theta=theta, phi=phi, nside=nside)
        return model * (frequencies / (408 * MEGA)) ** index

    def interpolated_haslam(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Returns the interpolated haslam map
        :param theta: sky angle in degrees
        :param phi: sky angle in degrees
        :return: `numpy.ndarray` of the same shape as `theta` and `phi`
        """
        model_haslam = healpy.read_map(os.path.join(ROOT_DIR, self._haslam_path)) - 8.9
        interpolated_model = healpy.pixelfunc.get_interp_val(model_haslam, theta / 180 * np.pi, phi / 180 * np.pi)
        return interpolated_model

    def interpolated_spectral_index(self, theta: np.ndarray, phi: np.ndarray, nside: int) -> np.ndarray:
        """
        Returns the interpolated spectral index map
        :param theta: sky angle in degrees
        :param phi: sky angle in degrees
        :param nside: integer `healpix` resolution parameter, forwarded to `pysm3`
        :return: `numpy.ndarray` of the same shape as `theta` and `phi`
        """
        model_index = pysm3.read_map(os.path.join(ROOT_DIR, self._spectral_index_path), nside)
        interpolated_index = healpy.pixelfunc.get_interp_val(model_index, theta / 180 * np.pi, phi / 180 * np.pi)
        return interpolated_index
