import os
from typing import Generator

from matplotlib import pyplot as plt

from definitions import ROOT_DIR
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.flag_element import FlagElement
from museek.flag_factory import FlagFactory
from museek.rfi_mitigation.aoflagger import get_rfi_mask
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.visualiser import waterfall
from museek.util.tools import flag_percent_recv
import pickle
import numpy as np
import datetime
from scipy.stats import spearmanr
import pysm3
import pysm3.units as u
import healpy as hp
from astropy.coordinates import SkyCoord


class AoflaggerPostCalibrationPlugin(AbstractParallelJoblibPlugin):
    """ Plugin to calculate RFI flags using the aoflagger algorithm and to post-process them, for calibrated data """

    def __init__(self,
                 mask_type: str,
                 first_threshold: float,
                 threshold_scales: list[float],
                 smoothing_kernel: tuple[int, int],
                 smoothing_sigma: tuple[float, float],
                 struct_size: tuple[int, int] | None,
                 channel_flag_threshold: float,
                 time_dump_flag_threshold: float,
                 flag_combination_threshold: int,
                 correlation_threshold: float,
                 synch_model:[str],
                 nside: int,
                 beamsize: float,
                 beam_frequency: float,
                 do_store_context: bool,
                 **kwargs):
        """
        Initialise the plugin
        :param mask_type: the data to which the flagger will be applied
        :param first_threshold: initial threshold to be used for the aoflagger algorithm
        :param threshold_scales: list of sensitivities
        :param smoothing_kernel: smoothing kernel window size tuple for axes 0 and 1
        :param smoothing_sigma: smoothing kernel sigma tuple for axes 0 and 1
        :param struct_size: structure size for binary dilation, closing etc
        :param channel_flag_threshold: if the fraction of flagged channels exceeds this, all channels are flagged
        :param time_dump_flag_threshold: if the fraction of flagged time dumps exceeds this, all time dumps are flagged
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        :param correlation_threshold: correlation coefficient threshold between calibrated data and synch model for excluding bad antennas
        :param synch_model: model used to create synchrotron sky
        :param nside: resolution parameter at which the synchrotron model is to be calculated
        :param beamsize: the beam fwhm used to smooth the Synch model [arcmin]
        :param beam_frequency: reference frequencies at which the beam fwhm are defined [MHz]
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        """
        super().__init__(**kwargs)
        self.mask_type = mask_type
        self.first_threshold = first_threshold
        self.threshold_scales = threshold_scales
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_sigma = smoothing_sigma
        self.struct_size = struct_size
        self.flag_combination_threshold = flag_combination_threshold
        self.channel_flag_threshold = channel_flag_threshold
        self.time_dump_flag_threshold = time_dump_flag_threshold
        self.correlation_threshold = correlation_threshold
        self.synch_model = synch_model
        self.nside = nside
        self.beamsize = beamsize
        self.beam_frequency = beam_frequency
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
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')]

    def map(self,
            scan_data: TimeOrderedData,
            calibrated_data: np.ma.MaskedArray,
            freq_select: np.ndarray,
            flag_report_writer: ReportWriter,
            output_path: str,
            block_name: str) \
            -> Generator[tuple[str, DataElement, FlagElement, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str], None, None]:
        """
        Yield a `tuple` of the results path for one antenna, the scanning calibrated data for one antenna and the flag for one antenna.
        :param scan_data: time ordered data containing the scanning part of the observation
        :param calibrated_data: calibrated data containing the scanning part of the observation
        :param freq_select: frequency for calibrated data, in [Hz]
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        :param block_name: name of the data block, not used here but for setting results
        """
        print(f'flag_antennas_using_correlation_with_synch_model: Producing synch sky: synch model {self.synch_model} used')
        receiver_path = None
        for i_antenna, antenna in enumerate(scan_data.antennas):
            visibility = calibrated_data.data[:,:,i_antenna]
            initial_flag = calibrated_data.mask[:,:,i_antenna]
            right_ascension = scan_data.right_ascension.get(recv=i_antenna).squeeze
            declination = scan_data.declination.get(recv=i_antenna).squeeze
            yield receiver_path, DataElement(array=visibility[:,:,np.newaxis]), FlagElement(array=initial_flag[:,:,np.newaxis]), right_ascension, declination, freq_select, antenna.name

    def run_job(self, anything: tuple[str, DataElement, FlagElement, np.ndarray, np.ndarray, np.ndarray, str]) -> np.ndarray:
        """
        Run the Aoflagger algorithm and post-process the result. Done for one antenna at a time.
        :param anything: `tuple` of the output path, calibrated data, initial flag, right ascension, declination, frequency and antenna name
        :return: updated mask
        """
        receiver_path, visibility, initial_flag, right_ascension, declination, freq_select, antenna = anything
        rfi_flag = get_rfi_mask(time_ordered=visibility,
                                mask=initial_flag,
                                mask_type=self.mask_type,
                                first_threshold=self.first_threshold,
                                threshold_scales=self.threshold_scales,
                                output_path=receiver_path,
                                smoothing_window_size=self.smoothing_kernel,
                                smoothing_sigma=self.smoothing_sigma)

        initial_flag = self.post_process_flag(flag=rfi_flag, initial_flag=initial_flag).squeeze
        return self.flag_antennas_using_correlation_with_synch_model(visibility.squeeze, initial_flag, right_ascension, declination, freq_select, antenna) 

    def gather_and_set_result(self,
                              result_list: list[np.ndarray],
                              scan_data: TimeOrderedData,
                              calibrated_data: np.ma.MaskedArray,
                              freq_select: np.ndarray,
                              flag_report_writer: ReportWriter,
                              output_path: str,
                              block_name: str):
        """
        Combine the `np.ma.MaskedArray`s in `result_list` into a new data set.
        :param result_list: `list` of `np.ndarray`s created from the RFI flagging
        :param scan_data: time ordered data containing the scanning part of the observation
        :param calibrated_data: calibrated data containing the scanning part of the observation
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        :param block_name: name of the observation block
        """

        calibrated_data.mask = np.array(result_list).transpose(1, 2, 0)

        flag_percent = []
        antennas_list = []
        for i_antenna, antenna in enumerate(scan_data.antennas):
            flag_percent.append(round(np.sum(calibrated_data.mask[:,:,i_antenna]>=1)/len(calibrated_data.mask[:,:,i_antenna].flatten()), 4))
            antennas_list.append(str(antenna.name))

        current_datetime = datetime.datetime.now()
        lines = ['...........................', 'Running AoflaggerPostCalibrationPlugin...Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"), 'The flag fraction for each antenna: '] + [f'{x}  {y}' for x, y in zip(antennas_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        self.set_result(result=Result(location=ResultEnum.CALIBRATED_VIS, result=calibrated_data, allow_overwrite=True))
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

        if mask.all():
            pass
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

            map_freqmedian = np.ma.median(np.ma.masked_array(calibrated_data, mask = mask_update), axis=1)
            synch_I = np.ma.masked_array(synch_I, mask=map_freqmedian.mask)
            synch_I = synch_I / 10**6.   #####  convert from uK to K

            spearman_corr, spearman_p = spearmanr(map_freqmedian.data[~map_freqmedian.mask], synch_I.data[~synch_I.mask])

            if spearman_corr > self.correlation_threshold:
                pass
            else:
                mask[:] = True
                print(f'antenna {antenna} is masked, Spearman correlation coefficient is '+str(round(spearman_corr,3)))

        return mask




