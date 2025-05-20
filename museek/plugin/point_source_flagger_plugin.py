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
from museek.util.report_writer import ReportWriter
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess
from museek.util.tools import point_sources_coordinate, point_source_flagger
from museek.util.tools import flag_percent_recv, git_version_info
import pysm3.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import h5py
import pickle
import csv
import datetime

class PointSourceFlaggerPlugin(AbstractParallelJoblibPlugin):
    """ Plugin to calculate TOD masks for point sources. """

    def __init__(self, 
            point_source_file_path: str, 
            beam_threshold: float,
            point_sources_match_flux: float,
            point_sources_match_raregion: float,
            point_sources_match_decregion: float,
            beamsize: float, 
            beam_frequency: float,
            **kwargs):
        """
        Initialise the plugin
        :param point_source_file_path: path to the point source location file
        :param beam_threshold: times of the beam size around the point source to be flagged 
        :param point_sources_match_flux: flux threshold above which the point sources are selected
        :param point_sources_match_raregion: the ra distance to the median of observed ra to select the point sources [deg]
        :param point_sources_match_decregion: the dec region to the median of observed dec to select the point sources [deg]
        :param beamsize: the beam fwhm [arcmin]
        :param beam_frequency: reference frequency at which the beam fwhm are defined [MHz]
        """
        super().__init__(**kwargs)
        self.point_source_file_path = point_source_file_path
        self.beam_threshold = beam_threshold
        self.point_sources_match_flux = point_sources_match_flux
        self.point_sources_match_raregion = point_sources_match_raregion
        self.point_sources_match_decregion = point_sources_match_decregion
        self.beamsize = beamsize
        self.beam_frequency = beam_frequency

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')]

    def map(self,
            scan_data: TimeOrderedData,
            flag_report_writer: ReportWriter,
            block_name: str) \
            -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, tuple], None, None]:
        """
        Yield a `tuple` of the right_ascension, declination, frequency, and shape of visibility for one antenna
        :param scan_data: time ordered data containing the scanning part of the observation
        :param flag_report_writer: report of the flagged fraction
        :param block_name: name of the data block
        """

        right_ascension_median = np.median(scan_data.right_ascension.array)
        declination_median = np.median(scan_data.declination.array)
        ra_point_source, dec_point_source, _ = point_sources_coordinate(self.point_source_file_path, right_ascension_median, declination_median, self.point_sources_match_flux, self.point_sources_match_raregion, self.point_sources_match_decregion)

        frequency = scan_data.frequencies.squeeze
        for i_antenna, antenna in enumerate(scan_data.antennas):
            right_ascension = scan_data.right_ascension.get(recv=i_antenna).squeeze
            declination = scan_data.declination.get(recv=i_antenna).squeeze
            yield right_ascension, declination, frequency, ra_point_source, dec_point_source

    def run_job(self, anything: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """ Run the plugin and calculate the TOD masks for point sources in the footprint of `scan_data`. """

        right_ascension, declination, frequency, ra_point_source, dec_point_source = anything

        point_source_flag = point_source_flagger(ra_point_source, dec_point_source, right_ascension, declination, frequency, self.beam_threshold, self.beamsize, self.beam_frequency)

        return point_source_flag
        
    def gather_and_set_result(self,
                              result_list: list[np.ndarray],
                              scan_data: TimeOrderedData,
                              flag_report_writer: ReportWriter,
                              block_name: str):
        """
        Combine the flags in `result_list` into a new flag and set that as a result.
        :param result_list: `list` of `FlagElement`s created from the RFI flagging
        :param scan_data: `TimeOrderedData` containing the scanning part of the observation
        :param flag_report_writer: report of the flagged fraction
        :param block_name: name of the observation block
        """

        result_list = np.array(result_list).transpose(1, 2, 0)
        self.set_result(result=Result(location=ResultEnum.POINT_SOURCE_FLAG, result=result_list, allow_overwrite=True))

        flag_percent = []
        receivers_list = []
        for i_receiver, receiver in enumerate(scan_data.receivers):
            flag_recv = scan_data.flags.get(recv=i_receiver)
            flag_recv_combine = flag_recv.combine(threshold=1)

            i_antenna = scan_data.antenna_index_of_receiver(receiver=receiver)
            flag_recv_combine = (flag_recv_combine.squeeze + result_list[:,:,i_antenna])>=1
            flag_percent.append(round(np.sum(flag_recv_combine)/len(flag_recv_combine.flatten()), 4))
            receivers_list.append(str(receiver))

        ## Note that the flag for point sources will be recovered after the aoflagger 
        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = ['...........................', 'Running PointSourceFlaggerPlugin with '+f"MuSEEK version: {branch} ({commit})", 'Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"), 'The flag fraction for each receiver: '] + [f'{x}  {y}' for x, y in zip(receivers_list, flag_percent)]
        flag_report_writer.write_to_report(lines)


