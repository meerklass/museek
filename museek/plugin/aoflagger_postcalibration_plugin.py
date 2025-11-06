import os
from typing import Generator

from matplotlib import pyplot as plt

from museek.definitions import ROOT_DIR
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.flag_element import FlagElement
from museek.flag_factory import FlagFactory
from museek.rfi_mitigation.aoflagger import get_rfi_mask
from museek.rfi_mitigation.aoflagger_1d import get_rfi_mask_1d
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import Synch_model_sm
from museek.visualiser import waterfall
from museek.util.tools import flag_percent_recv, git_version_info
from museek.util.tools import point_sources_coordinate, point_source_flagger
from museek.util.tools import remove_outliers_zscore_mad
import pickle
import numpy as np
import scipy
import datetime
from scipy.stats import spearmanr
import pysm3
import pysm3.units as u
import healpy as hp
from astropy.coordinates import SkyCoord
from scipy import ndimage


class AoflaggerPostCalibrationPlugin(AbstractParallelJoblibPlugin):
    """ Plugin to calculate RFI flags using the aoflagger algorithm and to post-process them, for calibrated data """

    def __init__(self,
                 first_threshold_rms: float,
                 first_threshold_flag_fraction: float,
                 threshold_scales: list[float],
                 smoothing_kernel_rms: int,
                 smoothing_sigma_rms: float,
                 smoothing_kernel_flag_fraction: int,
                 smoothing_sigma_flag_fraction: float,
                 struct_size: tuple[int, int] | None,
                 channel_flag_threshold: float,
                 time_dump_flag_threshold: float,
                 flag_combination_threshold: int,
                 poly_fit_degree: int,
                 poly_fit_threshold: float,
                 correlation_threshold_ant: float,
                 synch_model:[str],
                 nside: int,
                 beamsize: float,
                 beam_frequency: float,
                 zscore_antenatempflag_threshold: float,
                 do_store_context: bool,
                 **kwargs):
        """
        Initialise the plugin
        :param first_threshold_rms: initial threshold to be used for the aoflagger for the rms of powerlaw fitting residuals
        :param first_threshold_flag_fraction: initial threshold to be used for the aoflagger for the flagged fraction
        :param threshold_scales: list of sensitivities
        :param smoothing_kernel_rms: smoothing kernel window size for axis 0, used by aoflagger on the RMS of power-law fitting residuals
        :param smoothing_sigma_rms: smoothing kernel sigma for axes 0, used by aoflagger on the RMS of power-law fitting residuals
        :param smoothing_kernel_flag_fraction: smoothing kernel window size for axes 0, used by aoflagger on the flagged fraction
        :param smoothing_sigma_flag_fraction: smoothing kernel sigma for axes 0, used by aoflagger on the flagged fraction
        :param struct_size: structure size for binary dilation, closing etc
        :param channel_flag_threshold: if the fraction of flagged channels exceeds this, all channels are flagged
        :param time_dump_flag_threshold: if the fraction of flagged time dumps exceeds this, all time dumps are flagged
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        :param poly_fit_degree: degree of polynomials used to fit the data with the time median removed
        :param poly_fit_threshold: threshold (times of MAD) of polynomials fitting flagging
        :param correlation_threshold_ant: correlation coefficient threshold between calibrated data and synch model for excluding bad antennas
        :param synch_model: model used to create synchrotron sky
        :param nside: resolution parameter at which the synchrotron model is to be calculated
        :param beamsize: the beam fwhm used to smooth the Synch model [arcmin]
        :param beam_frequency: reference frequencies at which the beam fwhm are defined [MHz]
        :param zscore_antenatempflag_threshold: threshold for flagging the antennas based on their average temperature using modified zscore method
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        """
        super().__init__(**kwargs)
        self.first_threshold_rms = first_threshold_rms
        self.first_threshold_flag_fraction = first_threshold_flag_fraction
        self.threshold_scales = threshold_scales
        self.smoothing_kernel_rms = smoothing_kernel_rms
        self.smoothing_sigma_rms = smoothing_sigma_rms
        self.smoothing_kernel_flag_fraction = smoothing_kernel_flag_fraction
        self.smoothing_sigma_flag_fraction = smoothing_sigma_flag_fraction
        self.struct_size = struct_size
        self.flag_combination_threshold = flag_combination_threshold
        self.channel_flag_threshold = channel_flag_threshold
        self.time_dump_flag_threshold = time_dump_flag_threshold
        self.poly_fit_degree = poly_fit_degree
        self.poly_fit_threshold = poly_fit_threshold
        self.correlation_threshold_ant = correlation_threshold_ant
        self.synch_model = synch_model
        self.nside = nside
        self.beamsize = beamsize
        self.beam_frequency = beam_frequency
        self.zscore_antenatempflag_threshold = zscore_antenatempflag_threshold
        self.do_store_context = do_store_context
        self.report_file_name = 'flag_report.md'

    def set_requirements(self):
        """
        Set the requirements, the scanning data `scan_data`, a path to store results and the name of the data block.
        """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.CALIBRATED_VIS, variable='calibrated_data'),
                             Requirement(location=ResultEnum.FREQ_SELECT, variable='freq_select'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer'),
                             Requirement(location=ResultEnum.POINT_SOURCE_FLAG, variable='point_source_flag'),
                             Requirement(location=ResultEnum.CALIBRATED_VIS_FLAG, variable='calibrated_data_flag'),
                             Requirement(location=ResultEnum.CALIBRATED_VIS_FLAG_NAME_LIST, variable='calibrated_data_flag_name_list')]

    def map(self,
            scan_data: TimeOrderedData,
            calibrated_data: np.ma.MaskedArray,
            point_source_flag: np.ndarray,
            freq_select: np.ndarray,
            flag_report_writer: ReportWriter,
            output_path: str,
            block_name: str,
            calibrated_data_flag: list[np.ndarray],
            calibrated_data_flag_name_list: list) \
            -> Generator[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str], None, None]:
        """
        Yield a `tuple` of the results path for one antenna, the scanning calibrated data for one antenna and the flag for one antenna.
        :param scan_data: time ordered data containing the scanning part of the observation
        :param calibrated_data: calibrated data containing the scanning part of the observation
        :param point_source_flag: mask for point sources
        :param freq_select: frequency for calibrated data, in [Hz]
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        :param block_name: name of the data block
        :param calibrated_data_flag: list of the existing flags for calibrated data
        :param calibrated_data_flag_name_list: list of the name of existing flags for calibrated data
        """
        print(f'flag frequency and antennas using correlation with synch model: Producing synch sky: synch model {self.synch_model} used')

        ######## mask antennas that have temperature far from the median temperature for all antennas
        calibrated_data_median = np.ma.masked_array(np.zeros(len(scan_data.antennas)), mask=np.zeros(len(scan_data.antennas)))
        for i_antenna, antenna in enumerate(scan_data.antennas):
            calibrated_data_median[i_antenna] = np.ma.median(calibrated_data[:,:,i_antenna])
        antenna_mask_temp = remove_outliers_zscore_mad(calibrated_data_median.data, calibrated_data_median.mask, self.zscore_antenatempflag_threshold)
        calibrated_data.mask[:,:,antenna_mask_temp] = True
        calibrated_data_flag.append(calibrated_data.mask)
        calibrated_data_flag_name_list.append('temp_outlier_flag')

        ##########   report of the flagging
        flag_percent = []
        antennas_list = []
        for i_antenna, antenna in enumerate(scan_data.antennas):
            flag_percent.append(round(np.sum(calibrated_data.mask[:,:,i_antenna]>=1)/len(calibrated_data.mask[:,:,i_antenna].flatten()), 4))
            antennas_list.append(str(antenna.name))

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = ['...........................', 'Running AoflaggerPostCalibrationPlugin with '+f"MuSEEK version: {branch} ({commit})", 'temperature outlier flagger finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"), 'The flag fraction for each antenna: '] + [f'{x}  {y}' for x, y in zip(antennas_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        ######## mask antennas that have larger temperature fluctuations 
        calibrated_data_std = np.ma.masked_array(np.zeros(len(scan_data.antennas)), mask=np.zeros(len(scan_data.antennas)))
        for i_antenna, antenna in enumerate(scan_data.antennas):
            calibrated_data_std[i_antenna] = np.ma.std(calibrated_data[:,:,i_antenna])
        antenna_mask_temp = remove_outliers_zscore_mad(calibrated_data_std.data, calibrated_data_std.mask, self.zscore_antenatempflag_threshold)
        calibrated_data.mask[:,:,antenna_mask_temp] = True
        calibrated_data_flag.append(calibrated_data.mask)
        calibrated_data_flag_name_list.append('temp_fluctuation_flag')

        ##########   report of the flagging
        flag_percent = []
        antennas_list = []
        for i_antenna, antenna in enumerate(scan_data.antennas):
            flag_percent.append(round(np.sum(calibrated_data.mask[:,:,i_antenna]>=1)/len(calibrated_data.mask[:,:,i_antenna].flatten()), 4))
            antennas_list.append(str(antenna.name))

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = ['...........................', 'Running AoflaggerPostCalibrationPlugin with '+f"MuSEEK version: {branch} ({commit})", 'temperature fluctuation flagger, finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"), 'The flag fraction for each antenna: '] + [f'{x}  {y}' for x, y in zip(antennas_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        ########################################
        receiver_path = None
        for i_antenna, antenna in enumerate(scan_data.antennas):
            right_ascension = scan_data.right_ascension.get(recv=i_antenna).squeeze
            declination = scan_data.declination.get(recv=i_antenna).squeeze
            yield receiver_path, calibrated_data.data[:,:,i_antenna], calibrated_data.mask[:,:,i_antenna], point_source_flag[:,:,i_antenna], right_ascension, declination, freq_select, antenna.name

    def run_job(self, anything: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]) -> np.ndarray:
        """
        Run the Aoflagger algorithm and post-process the result. Done for one antenna at a time.
        :param anything: `tuple` of the output path, calibrated data, flag, right ascension, declination, frequency and antenna name
        :return: updated mask
        """
        receiver_path, visibility_recv, initial_flag_recv, point_source_flag_recv, right_ascension, declination, freq_select, antenna  = anything

        ###########  fitting a powerlaw to the time median removed data and then mask outlier   ######## 
        visibility_recv_masked = np.ma.masked_array(visibility_recv, mask=initial_flag_recv)
        visibility_recv_masked = visibility_recv_masked - np.ma.median(visibility_recv_masked, axis=0)
        for i in range(visibility_recv.shape[0]):
            visibility_recv_masked.mask[i], p_fit = self.polynomial_flag_outlier(freq_select, visibility_recv_masked.data[i], visibility_recv_masked.mask[i], self.poly_fit_degree, self.poly_fit_threshold)

        #############  second run  #########
        visibility_recv_masked = np.ma.masked_array(visibility_recv, mask=visibility_recv_masked.mask)
        visibility_recv_masked = visibility_recv_masked - np.ma.median(visibility_recv_masked, axis=0)
        residuals = np.zeros_like(visibility_recv)
        for i in range(visibility_recv.shape[0]):
            visibility_recv_masked.mask[i], p_fit = self.polynomial_flag_outlier(freq_select, visibility_recv_masked.data[i], visibility_recv_masked.mask[i], self.poly_fit_degree, self.poly_fit_threshold)
            if visibility_recv_masked.mask[i].all():
                pass
            else:
                residuals[i] = visibility_recv_masked.data[i] - np.polyval(p_fit, freq_select) 
        residuals = np.ma.masked_array(residuals, mask=visibility_recv_masked.mask)


        ###############  aoflagger on the rms of powerlaw fitting residuals  #################
        residuals_rms = np.ma.std(residuals, axis=0).data
        residuals_rms_mask = np.ma.std(residuals, axis=0).mask

        rfi_flag = get_rfi_mask_1d(time_ordered=DataElement(array=residuals_rms[:,np.newaxis,np.newaxis]),
                                mask=FlagElement(array=residuals_rms_mask[:,np.newaxis,np.newaxis]),
                                mask_type='vis',
                                first_threshold=self.first_threshold_rms,
                                threshold_scales=self.threshold_scales,
                                output_path=receiver_path,
                                smoothing_window_size=self.smoothing_kernel_rms,
                                smoothing_sigma=self.smoothing_sigma_rms)


        rfi_flag_tile = np.tile(rfi_flag.squeeze, (visibility_recv.shape[0], 1))
        initial_flag_tile = np.tile(residuals_rms_mask, (visibility_recv.shape[0], 1))
        initial_flag_post = self.post_process_flag(flag=FlagElement(array=rfi_flag_tile[:,:,np.newaxis]), initial_flag=FlagElement(array=initial_flag_tile[:,:,np.newaxis]))

        initial_flag_postrms = initial_flag_post.squeeze + visibility_recv_masked.mask


        ###############  aoflagger on the flagged fraction  #################
        flag_fraction = np.mean(initial_flag_postrms>0, axis=0)
        flag_fraction_mask = flag_fraction > self.time_dump_flag_threshold

        rfi_flag = get_rfi_mask_1d(time_ordered=DataElement(array=flag_fraction[:,np.newaxis,np.newaxis]),
                                mask=FlagElement(array=flag_fraction_mask[:,np.newaxis,np.newaxis]),
                                mask_type='vis',
                                first_threshold=self.first_threshold_flag_fraction,
                                threshold_scales=self.threshold_scales,
                                output_path=receiver_path,
                                smoothing_window_size=self.smoothing_kernel_flag_fraction,
                                smoothing_sigma=self.smoothing_sigma_flag_fraction)


        rfi_flag_tile = np.tile(rfi_flag.squeeze, (visibility_recv.shape[0], 1))
        initial_flag_tile = np.tile(flag_fraction_mask, (visibility_recv.shape[0], 1))
        initial_flag_post = self.post_process_flag(flag=FlagElement(array=rfi_flag_tile[:,:,np.newaxis]), initial_flag=FlagElement(array=initial_flag_tile[:,:,np.newaxis]))

        initial_flag_postflagfraction = initial_flag_post.squeeze + initial_flag_postrms

        return self.flag_antennas_using_correlation_with_synch_model(visibility_recv, initial_flag_postflagfraction, point_source_flag_recv, right_ascension, declination, freq_select, antenna)

    def gather_and_set_result(self,
                              result_list: list[np.ndarray],
                              scan_data: TimeOrderedData,
                              calibrated_data: np.ma.MaskedArray,
                              point_source_flag: np.ndarray,
                              freq_select: np.ndarray,
                              flag_report_writer: ReportWriter,
                              output_path: str,
                              block_name: str,
                              calibrated_data_flag: list[np.ndarray],
                              calibrated_data_flag_name_list: list):
        """
        Combine the `np.ma.MaskedArray`s in `result_list` into a new data set, and mask the frequencies that flag fraction is high (taking all antennas into consideration)
        :param result_list: `list` of `np.ndarray`s created from the RFI flagging
        :param scan_data: time ordered data containing the scanning part of the observation
        :param calibrated_data: calibrated data containing the scanning part of the observation
        :param point_source_flag: mask for point sources
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        :param block_name: name of the observation block
        :param calibrated_data_flag: list of the existing flags for calibrated data
        :param calibrated_data_flag_name_list: list of the name of existing flags for calibrated data
        """

        result_list = np.array(result_list, dtype='object')
        correlation_coefficient_ant = [result_list[i][1] for i in range(np.shape(result_list)[0])]
        calibrated_data.mask = np.array([result_list[i][0] for i in range(np.shape(result_list)[0])]).transpose(1, 2, 0)
        polynomial_fit_flag = np.array([result_list[i][2] for i in range(np.shape(result_list)[0])]).transpose(1, 2, 0)

        ##########  if a certain fraction of a frequency channel is flagged at any timestamp and antennas, the remainder is flagged as well
        good_antennas = [~calibrated_data[:,:,i_antenna].mask.all() for i_antenna, antenna in enumerate(scan_data.antennas)]
        for i_freq, freq in enumerate(freq_select):
            if np.mean(calibrated_data.mask[:,i_freq,good_antennas]) > self.time_dump_flag_threshold:
                calibrated_data.mask[:,i_freq,:] = True
            else:
                pass

        calibrated_data_flag.append(polynomial_fit_flag)
        calibrated_data_flag_name_list.append('polynomial_fit_flag')
        calibrated_data_flag.append(calibrated_data.mask)
        calibrated_data_flag_name_list.append('synch_correlation_flag')


        ##########   report of the flagging  
        flag_percent = []
        antennas_list = []
        for i_antenna, antenna in enumerate(scan_data.antennas):
            flag_percent.append(round(np.sum(calibrated_data.mask[:,:,i_antenna]>=1)/len(calibrated_data.mask[:,:,i_antenna].flatten()), 4))
            antennas_list.append(str(antenna.name))

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = ['...........................', 'Running AoflaggerPostCalibrationPlugin with '+f"MuSEEK version: {branch} ({commit})", 'Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"), 'The flag fraction for each antenna: '] + [f'{x}  {y}' for x, y in zip(antennas_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        #########   save results 
        self.set_result(result=Result(location=ResultEnum.CALIBRATED_VIS, result=calibrated_data, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.CORRELATION_COEFFICIENT_VIS_SYNCH_ANT, result=correlation_coefficient_ant, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.CALIBRATED_VIS_FLAG, result=calibrated_data_flag, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.CALIBRATED_VIS_FLAG_NAME_LIST, result=calibrated_data_flag_name_list, allow_overwrite=True))
        if self.do_store_context:
            context_file_name = 'aoflagger_plugin_postcalibration.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=output_path)


    def post_process_flag(
            self,
            flag: FlagElement,
            initial_flag: FlagElement
    ) -> FlagElement:
        """
        Post process `flag` and return the result.
        The following is done:
        - `flag` is dilated using `self.struct_size` if it is not `None`
        - binary closure is applied to `flag`
        - if a certain fraction of all channels is flagged at any timestamp, the remainder is flagged as well
        :param flag: binary mask to be post-processed
        :param initial_flag: initial flag on which `flag` was based
        :return: the result of the post-processing, a binary mask
        """
        # operations on the RFI mask only
        post_process = RfiPostProcess(new_flag=flag, initial_flag=initial_flag, struct_size=self.struct_size)
        post_process.binary_mask_dilation()
        post_process.binary_mask_closing()
        rfi_result = post_process.get_flag()


        # operations on the entire mask
        post_process = RfiPostProcess(new_flag=rfi_result + initial_flag,
                                      initial_flag=None,
                                      struct_size=self.struct_size)
        post_process.flag_all_channels(channel_flag_threshold=self.channel_flag_threshold)
        post_process.flag_all_time_dumps(time_dump_flag_threshold=self.time_dump_flag_threshold)
        overall_result = post_process.get_flag()
        return overall_result


    def flag_antennas_using_correlation_with_synch_model(
            self, 
            calibrated_data: np.ndarray, 
            mask: np.ndarray, 
            point_source_flag: np.ndarray,
            ra: np.ndarray, 
            dec: np.ndarray, 
            freq: np.ndarray,
            antenna: str
        ) -> np.ndarray:
        """
        Excludes bad antennas based on the Spearman correlation coefficient between the synch model at the median of frequency and the frequency median of calibrated data.

        Parameters:
        calibrated_data: The calibrated visibility data [K]
        mask: The mask for calibrated visibility data [bool]
        ra: The right ascension for calibrated visibility data [deg]
        dec: The declination for calibrated visibility data [deg]
        freq: The frequency for calibrated visibility data [Hz]
        antenna: The antenna name for calibrated visibility data [str]

        Returns:
        bool array: mask all data for antennas considered to be bad.
        """

        mask_ori = mask.copy()
        if mask.all():
            spearman_corr = np.nan
        else:
            #####  Update the mask to eliminate inhomogeneities when take the median along time axis  ######
            mask_fraction = np.mean(mask[:,~np.all(mask, axis=0)], axis=1) # ingnore the all-timepoint masked frequency points, calculate the mask fraction
            time_points_to_mask = mask_fraction > 0.05 
            mask_update = mask.copy()
            mask_update[time_points_to_mask, :] = True  #mask time points where the mask fraction is greater than the threshold
            ########  calculate the frequency corrsponding to the frequency median of calibrated data   ########
            mask_freq = np.median(mask_update[~time_points_to_mask,:], axis=0) == 1.
            assert mask_freq.all() == False, f"antenna {antenna}, too much data is masked, please check data, maybe increase the threshold for mask_fraction"
            freq_median = np.median(freq[~mask_freq])

            #########   produce synch model at freq_median and smooth   #########
            sky = pysm3.Sky(nside=self.nside, preset_strings=self.synch_model)  
            map_reference = sky.get_emission(freq_median * u.Hz).value
            map_reference_smoothed = pysm3.apply_smoothing_and_coord_transform(map_reference, fwhm=self.beamsize*u.arcmin * ((self.beam_frequency*u.MHz)/(freq_median * u.Hz)).decompose().value )

            #########   map the smoothed synch model to the same sky covered by ra,dec of calibrated data
            c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
            theta = 90. - (c.galactic.b / u.degree).value
            phi = (c.galactic.l / u.degree).value
            synch_I = hp.pixelfunc.get_interp_val(map_reference_smoothed[0], theta / 180. * np.pi, phi / 180. * np.pi)

            map_freqmedian = np.ma.median(np.ma.masked_array(calibrated_data, mask = mask_update+point_source_flag), axis=1)
            synch_I = np.ma.masked_array(synch_I, mask=map_freqmedian.mask)
            synch_I = synch_I / 10**6.   #####  convert from uK to K

            spearman_corr, spearman_p = spearmanr(map_freqmedian.data[~map_freqmedian.mask], synch_I.data[~synch_I.mask])

            if spearman_corr > self.correlation_threshold_ant:
                pass
            else:
                mask[:] = True
                print(f'antenna {antenna} is masked, Spearman correlation coefficient is '+str(round(spearman_corr,3)))

        return mask, spearman_corr, mask_ori


    def polynomial_flag_outlier(self, x, y, mask, degree, threshold):
        """
        fitting a powerlaw to the input data and mask outliers which deviate the powerlaw larger than `threshold' times median absolute deviation

        Parameters:
        x: x-coordinates of the sample points [array]
        y: y-coordinates of the sample points [array]
        mask: mask for the sample points [bool]
        degree: Degree of the fitting polynomial [int]
        threshold: threshold of the masking [float]

        Returns:
        bool array: updated mask
        1darray: fitting polynomial coefficients
        """
        
        initial_mask = mask.copy()
        #struct = np.ones(self.struct_size[1], dtype=bool)
        ########  mask the whole data if the flagged fraction is larger than 0.5
        if np.mean(mask>0) > self.channel_flag_threshold:
            initial_mask[:] = True
            p_fit = None
        else:
           fit_x = x[~mask]
           fit_y = y[~mask]

           # Fit a polynomial
           p_fit = np.polyfit(fit_x, fit_y, degree)
           y_fit = np.polyval(p_fit, fit_x)

           # Calculate residuals
           residuals = fit_y - y_fit
           mad_residuals = scipy.stats.median_abs_deviation(residuals)

           # Identify outliers
           initial_mask[~mask] |= np.abs(residuals) >= threshold * mad_residuals

           #to_dilate = initial_mask ^ mask
           #dilated = ndimage.binary_dilation(to_dilate, structure=struct, iterations=1)
           #initial_mask = initial_mask + dilated

        return initial_mask, p_fit




