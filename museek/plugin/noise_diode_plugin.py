from typing import Generator
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.noise_diode import NoiseDiode
from museek.flag_factory import FlagFactory
from museek.flag_element import FlagElement
from museek.flag_list import FlagList
from museek.data_element import DataElement
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import flag_percent_recv, git_version_info
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess
from museek.util.tools import remove_outliers_zscore_mad, polynomial_flag_outlier
import pysm3.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import scipy
import h5py
import pickle
import csv
from scipy.interpolate import splrep, BSpline
import warnings
import datetime

class NoiseDiodePlugin(AbstractParallelJoblibPlugin):
    """ Plugin to calculate noise diode excess signal, flag receivers and calibrate raw vis based on the noise diode excess signal """

    def __init__(self,
                 flag_combination_threshold: int,
                 zscoreflag_threshold: float,
                 polyflag_deg: int,
                 polyflag_threshold: float,
                 polyfit_deg: int,
                 zscore_antenaflag_threshold: float,
                 noise_diode_excess_lowlim: float,
                 **kwargs):
        """
        Initialise the plugin
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        :param zscoreflag_threshold: threshold for flagging noise diode excess using modified zscore method
        :param polyflag_deg: degree of the polynomials used for fitting and flagging noise diode excess
        :param polyflag_threshold: threshold for flagging noise diode excess using polynomials fit
        :param polyfit_deg: degree of the polynomials used for fitting flagged noise diode excess
        :param zscore_antenaflag_threshold: threshold for flagging the rms of noise diode excess of receivers using modified zscore method
        :param noise_diode_excess_lowlim: threshold for flagging the mean value of noise diode excess of receivers
        """
        super().__init__(**kwargs)
        self.flag_combination_threshold = flag_combination_threshold
        self.zscoreflag_threshold = zscoreflag_threshold
        self.polyflag_deg = polyflag_deg
        self.polyflag_threshold = polyflag_threshold
        self.polyfit_deg = polyfit_deg
        self.zscore_antenaflag_threshold = zscore_antenaflag_threshold
        self.noise_diode_excess_lowlim = noise_diode_excess_lowlim
        self.report_file_name = 'flag_report.md'

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')]

    def map(self,
            scan_data: TimeOrderedData,
            flag_report_writer: ReportWriter,) \
            -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, tuple], None, None]:
        """
        Yield a `tuple` of the scanning visibility data for one receiver and the initial flags for one receiver.
        :param scan_data: time ordered data containing the scanning part of the observation
        :param flag_report_writer: report of the flag
        """

        scan_data.load_visibility_flags_weights(polars='auto')
        initial_flags = scan_data.flags.combine(threshold=self.flag_combination_threshold)

        noise_diode = NoiseDiode(dump_period=scan_data.dump_period, observation_log=scan_data.obs_script_log)
        noise_diode_off_dumps = noise_diode.get_noise_diode_off_scan_dumps(timestamps=scan_data.timestamps)
        noise_diode_cycle_start_times = noise_diode._get_noise_diode_cycle_start_times(timestamps=scan_data.timestamps)
        noise_diode_ratios = noise_diode._get_noise_diode_ratios(timestamps=scan_data.timestamps,
                                                                 noise_diode_cycle_starts=noise_diode_cycle_start_times,
                                                                 dump_period=noise_diode._dump_period)

        for i_receiver, receiver in enumerate(scan_data.receivers):
            visibility = scan_data.visibility.get(recv=i_receiver).squeeze
            initial_flag = initial_flags.get(recv=i_receiver).squeeze
            yield visibility, initial_flag, noise_diode_off_dumps, noise_diode_ratios


    def run_job(self, anything: tuple[DataElement, np.ndarray, np.ndarray, np.ndarray]) -> list[np.ndarray, np.ndarray]:
        """ Run the plugin and calculate noise diode excess signal. """

        visibility, initial_flag, noise_diode_off_dumps, noise_diode_ratios = anything

        noise_on = np.ones(np.shape(visibility)[0], dtype='bool')
        for i in noise_diode_off_dumps:
            noise_on[i] = False

        # Find continuous noise-on regions
        continuous_noise_on_index = self.find_continuous_noise_on_regions(noise_on)

        # calculate the noise diode firing index. Because there will be noise diode firing in two continuous timestamps, we need to re-calculate the index of noise diode firing by use of noise_diode_ratios weighted average 
        noise_on_index = []
        for i_index_list, index_list in enumerate(continuous_noise_on_index):
            if len(index_list) == 1:   # if noise diode firing is in one timestamp
                noise_on_index.append(index_list[0])
            elif len(index_list) > 1:  # if noise diode firing is in more than one timestamps
                index_array = np.array(index_list)
                noise_on_index.append(np.sum(index_array*noise_diode_ratios[index_array]) / np.sum(noise_diode_ratios[index_array]))

        # calculate the noise diode firing - diode off, noise diode off is estimated by averaging two points (one before, one after) adjacent to the noise diode firing point
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            visibility = np.ma.masked_array(visibility.copy(), mask=initial_flag.copy())
            visibility.mask[noise_on,:] = False

        noise_diode_excess = np.ma.zeros((len(continuous_noise_on_index), visibility.shape[1]))
        for i_freq in range(visibility.shape[1]):
            for i_index_list, index_list in enumerate(continuous_noise_on_index):
                if len(index_list) == 1:
                    noise_on_value = visibility[index_list[0],i_freq]
                    noise_off_value = np.ma.mean([visibility[i,i_freq] for i in range(index_list[0]-1, index_list[0]+2) if i!=index_list[0]])
                    noise_diode_excess[i_index_list,i_freq] = noise_on_value - noise_off_value
        
                elif len(index_list) > 1:
                    noise_on_value = np.ma.sum([visibility[i,i_freq] for i in index_list])
                    noise_off_value = np.ma.mean([visibility[i,i_freq] for i in range(np.min(index_list)-1, np.max(index_list)+2) if i not in index_list])
                    noise_diode_excess[i_index_list,i_freq] = noise_on_value - noise_off_value*len(index_list)

        noise_diode_excess.mask[np.isnan(noise_diode_excess.data)] = True

        return noise_diode_excess, noise_on_index 
        
    def gather_and_set_result(self,
                              result_list: list[np.ndarray],
                              scan_data: TimeOrderedData,
                              flag_report_writer: ReportWriter):
        """
        Combine the masks in `result_list` into a new flag and set that as a result.
        :param result_list: `list` of `FlagElement`s created from the RFI flagging
        :param scan_data: `TimeOrderedData` containing the scanning part of the observation
        :param flag_report_writer: report of the flag
        """

        noise_on_index = np.array(result_list[0][1])
        noise_diode_excess = np.ma.array([result_list[i][0] for i in range(len(result_list))]).transpose(1, 2, 0)

        # fit the noise diode excess and mask the receiver if the frequency median of its noise diode excess can not be fitted well by a 2-order polynomial
        rmsnorm_poly_fit_list = []
        for i_receiver, receiver in enumerate(scan_data.receivers):
            noise_excess_timemedian = np.ma.median(noise_diode_excess[:,:,i_receiver], axis=1)
            noise_excess_timemedian.mask = remove_outliers_zscore_mad(noise_excess_timemedian.data, noise_excess_timemedian.mask, self.zscoreflag_threshold)
            noise_excess_timemedian.mask, p_fit = polynomial_flag_outlier(noise_on_index, noise_excess_timemedian.data, noise_excess_timemedian.mask, self.polyflag_deg, self.polyflag_threshold)
            if noise_excess_timemedian.mask.all():
                rmsnorm_poly_fit_list.append(np.nan)
            else:
                p_poly = np.polyfit(noise_on_index[~noise_excess_timemedian.mask], noise_excess_timemedian.data[~noise_excess_timemedian.mask], deg=self.polyfit_deg)
                rmsnorm_poly_fit_list.append(np.std(np.polyval(p_poly, noise_on_index) / np.median(np.polyval(p_poly, noise_on_index))))

        rmsnorm_poly_fit_list = np.array(rmsnorm_poly_fit_list)
        receiver_mask_poly_fit = remove_outliers_zscore_mad(rmsnorm_poly_fit_list, np.isnan(rmsnorm_poly_fit_list), self.zscore_antenaflag_threshold)


        shape = scan_data.visibility.shape
        new_flag = FlagList(flags=[FlagFactory().empty_flag(shape=shape)])
        new_flag_array = np.zeros((shape[0], len(scan_data.receivers)))
        for i_receiver, receiver in enumerate(scan_data.receivers):
            if receiver_mask_poly_fit[i_receiver] > 0 or np.ma.median(noise_diode_excess[:,:,i_receiver]) <= self.noise_diode_excess_lowlim:
                outlier_antenna_flag = np.ones((shape[0]))
            else:
                outlier_antenna_flag = new_flag_array[:,i_receiver]

            flag_array = np.repeat(outlier_antenna_flag, shape[1]).reshape((shape[0], shape[1], 1))
            new_flag.insert_receiver_flag(flag=DataElement(array=flag_array), i_receiver=i_receiver, index=0)

        scan_data.flags.add_flag(flag=new_flag)

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        receivers_list, flag_percent = flag_percent_recv(scan_data)
        lines = ['...........................', 'Running NoiseDiodePlugin with '+f"MuSEEK version: {branch} ({commit})", 'Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"), 'The flag fraction for each receiver: '] + [f'{x}  {y}' for x, y in zip(receivers_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.NOISE_DIODE_EXCESS, result=noise_diode_excess, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.NOISE_ON_INDEX, result=noise_on_index, allow_overwrite=True))


    def find_continuous_noise_on_regions(self,
                                        noise_on: np.ndarray):
        """
        Function to find continuous "noise on" sub-regions (the region where there are two timestamps sharing one noise diode firing)

        :param noise_on: bool array where noise diode firing is True, noise diode off is False
        :return: a list where noise diode is firing 
        """
        regions = []
        current_region = []
    
        for i, val in enumerate(noise_on):
            if val == 1:
                current_region.append(i)
            else:
                if current_region:
                    regions.append(current_region)
                    current_region = []
        # Append the last region if still present
        if current_region:
            regions.append(current_region)
            
        return regions


