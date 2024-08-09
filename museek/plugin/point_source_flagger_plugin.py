from typing import Generator
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.flag_factory import FlagFactory
from museek.flag_element import FlagElement
from museek.data_element import DataElement
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess
import pysm3.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import h5py
import pickle
import csv

class PointSourceFlaggerPlugin(AbstractParallelJoblibPlugin):
    """ Plugin to calculate TOD masks for point sources. """

    def __init__(self, 
            point_source_file_path: str, 
            beam_threshold: float,
            point_sources_match_raregion: float,
            point_sources_match_decregion: float,
            point_sources_match_flux: float,
            beamsize: float, 
            beam_frequency: float,
            struct_size: tuple[int, int] | None,
            **kwargs):
        """
        Initialise the plugin
        :param point_source_file_path: path to the point source location file
        :param beam_threshold: times of the beam size around the point source to be masked 
        :param point_sources_match_raregion: the ra distance to the median of observed ra to select the point sources [deg]
        :param point_sources_match_decregion: the dec region to the median of observed dec to select the point sources [deg]
        :param point_sources_match_flux: flux threshold above which the point sources are selected
        :param beamsize: the beam fwhm [arcmin]
        :param beam_frequency: reference frequency at which the beam fwhm are defined [MHz]
        :param struct_size: structure size for binary dilation, closing etc
        """
        super().__init__(**kwargs)
        self.point_source_file_path = point_source_file_path
        self.beam_threshold = beam_threshold
        self.point_sources_match_raregion = point_sources_match_raregion
        self.point_sources_match_decregion = point_sources_match_decregion
        self.point_sources_match_flux = point_sources_match_flux
        self.beamsize = beamsize
        self.beam_frequency = beam_frequency
        self.struct_size = struct_size

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data')]

    def map(self,
            scan_data: TimeOrderedData) \
            -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, tuple], None, None]:
        """
        Yield a `tuple` of the right_ascension, declination, frequency, and shape of visibility for one antenna
        :param scan_data: time ordered data containing the scanning part of the observation
        """
        frequency = scan_data.frequencies.squeeze
        for i_antenna, antenna in enumerate(scan_data.antennas):
            right_ascension = scan_data.right_ascension.get(recv=i_antenna).squeeze
            declination = scan_data.declination.get(recv=i_antenna).squeeze
            shape = (len(right_ascension), len(frequency))
            yield right_ascension, declination, frequency, shape

    def run_job(self, anything: tuple[np.ndarray, np.ndarray, np.ndarray, tuple]) -> np.ndarray:
        """ Run the plugin and calculate the TOD masks for point sources in the footprint of `scan_data`. """

        right_ascension, declination, frequency, shape = anything
        point_source_mask = self.get_point_source_mask(shape=shape,
                                                       right_ascension=right_ascension,
                                                       declination=declination,
                                                       frequency=frequency)

        return point_source_mask
        
    def gather_and_set_result(self,
                              result_list: list[np.ndarray],
                              scan_data: TimeOrderedData):
        """
        Combine the masks in `result_list` into a new flag and set that as a result.
        :param result_list: `list` of `FlagElement`s created from the RFI flagging
        :param scan_data: `TimeOrderedData` containing the scanning part of the observation
        """

        result_list = np.array(result_list).transpose(1, 2, 0)
        self.set_result(result=Result(location=ResultEnum.POINT_SOURCE_MASK, result=result_list, allow_overwrite=True))


    def point_sources_coordinate_list(self,
                                      ra: np.ndarray, 
                                      dec: np.ndarray, 
                                      point_source_file_path: str, 
                                      point_sources_match_raregion: float, 
                                      point_sources_match_decregion: float, 
                                      point_sources_match_flux: float) -> list[SkyCoord]:
        """
        Return a `list` of `SkyCoord` coordinates of point source data
        loaded from a file located at `point_source_file_path`.
        """
        #f_sources = h5py.File(point_source_file_path+'/NVSS_catalogue.hdf5','r')
        #ra_sources = np.array(f_sources['ra_NVSS'], dtype='f4')
        #dec_sources = np.array(f_sources['dec_NVSS'], dtype='f4')
        #flux_sources = np.array(f_sources['flux_NVSS'], dtype='f4')  # [mJy]
        #f_sources.close()

        ra_sources = []
        dec_sources = []
        flux_sources = []
        with open(point_source_file_path+'AS110_Derived_Catalogue_racs_dr1_sources_galacticcut_v2021_08_v02_5725.csv', mode='r') as file:
            csv_reader = csv.reader(file)
            # Read the header (first row)
            header = next(csv_reader)  # can not be quoted
            # Read the rest of the rows
            for row in csv_reader:
                ra_sources.append(float(row[8]))
                dec_sources.append(float(row[9]))
                flux_sources.append(float(row[12]))  # [mJy]

        ra_sources = np.array(ra_sources)
        dec_sources = np.array(dec_sources)
        flux_sources = np.array(flux_sources)

        ra_selection = (ra_sources>=np.median(ra)-point_sources_match_raregion) & (ra_sources<=np.median(ra)+point_sources_match_raregion)
        dec_selection = (dec_sources>=np.median(dec)-point_sources_match_decregion) & (dec_sources<=np.median(dec)+point_sources_match_decregion)  
        flux_selection = (flux_sources>=point_sources_match_flux)

        select_sourcesregion = ra_selection & dec_selection & flux_selection

        ra_sources_select = ra_sources[select_sourcesregion]
        dec_sources_select = dec_sources[select_sourcesregion]

        result = [SkyCoord(ra_ps * u.deg, dec_ps * u.deg, frame='icrs') for ra_ps, dec_ps in zip(ra_sources_select, dec_sources_select)]

        return result


    def get_point_source_mask(self,
                              shape: tuple[int, int],
                              right_ascension: np.ndarray,
                              declination: np.ndarray,
                              frequency: np.ndarray):
        """
        Return a mask that is `True` wherever a dump is close enough to a point source.
        :param shape: the returned mask will have this `shape`
        :param right_ascension: celestial coordinate right ascension [Deg]
        :param declination: celestial coordinate declination [Deg]
        :param frequency: frequency [Hz]
        :return: a mask which is `True` for all masked pixels
        """
        point_source_mask = np.zeros(shape, dtype=bool)
        mask_points = self.point_sources_coordinate_list(
                      point_source_file_path=self.point_source_file_path,
                      ra=right_ascension,
                      dec=declination,
                      point_sources_match_raregion=self.point_sources_match_raregion,
                      point_sources_match_decregion=self.point_sources_match_decregion,
                      point_sources_match_flux=self.point_sources_match_flux)

        for i_freq, freq in enumerate(frequency): 
            fwhm = self.beamsize / 60. * ((self.beam_frequency*u.MHz)/(freq * u.Hz)).decompose().value  ## in deg
            point_source_mask_dump_list = self._coordinates_mask_dumps(
                    right_ascension=right_ascension,
                    declination=declination,
                    mask_points=mask_points,
                    angle_threshold=self.beam_threshold * fwhm)

            point_source_mask[point_source_mask_dump_list, i_freq] = True
        return point_source_mask


    @staticmethod
    def _coordinates_mask_dumps(right_ascension: np.ndarray,
                                declination: np.ndarray,
                                mask_points: list[SkyCoord],
                                angle_threshold: float) \
            -> list[int]:
        """
        Return a list of dump indices that are less than `angle_threshold` away from a point in `mask_points`.
        :param right_ascension: celestial coordinate right ascension
        :param declination: celestial coordinate declination
        :param mask_points: `list` of `SkyCoord` coordinates of the point sources
        :param angle_threshold: all points up to this angular separation (degrees) are masked
        :return: `list` of masked dump indices
        """
        result = []
        data_points = SkyCoord(right_ascension * u.deg, declination * u.deg, frame='icrs')

        for mask_coord in mask_points:
            separation = (mask_coord.separation(data_points) / u.deg)
            result.extend(np.where(separation < angle_threshold)[0])
        result = list(set(result))
        result.sort()
        return result


