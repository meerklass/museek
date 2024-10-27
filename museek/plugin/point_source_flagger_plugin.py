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
from museek.util.tools import point_sources_coordinate, point_source_flag
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
            point_sources_match_flux: float,
            beamsize: float, 
            beam_frequency: float,
            **kwargs):
        """
        Initialise the plugin
        :param point_source_file_path: path to the point source location file
        :param beam_threshold: times of the beam size around the point source to be masked 
        :param point_sources_match_flux: flux threshold above which the point sources are selected
        :param beamsize: the beam fwhm [arcmin]
        :param beam_frequency: reference frequency at which the beam fwhm are defined [MHz]
        """
        super().__init__(**kwargs)
        self.point_source_file_path = point_source_file_path
        self.beam_threshold = beam_threshold
        self.point_sources_match_flux = point_sources_match_flux
        self.beamsize = beamsize
        self.beam_frequency = beam_frequency

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

        ra_point_source, dec_point_source, _ = point_sources_coordinate(self.point_source_file_path, self.point_sources_match_flux)

        frequency = scan_data.frequencies.squeeze
        for i_antenna, antenna in enumerate(scan_data.antennas):
            right_ascension = scan_data.right_ascension.get(recv=i_antenna).squeeze
            declination = scan_data.declination.get(recv=i_antenna).squeeze
            yield right_ascension, declination, frequency, ra_point_source, dec_point_source

    def run_job(self, anything: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """ Run the plugin and calculate the TOD masks for point sources in the footprint of `scan_data`. """

        right_ascension, declination, frequency, ra_point_source, dec_point_source = anything

        point_source_mask = point_source_flag(ra_point_source, dec_point_source, right_ascension, declination, frequency, self.beam_threshold, self.beamsize, self.beam_frequency)

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


