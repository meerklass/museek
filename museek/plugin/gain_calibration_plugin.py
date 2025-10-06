import os
from typing import Generator

from matplotlib import pyplot as plt

from definitions import ROOT_DIR
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.flag_element import FlagElement
from museek.flag_factory import FlagFactory
from museek.rfi_mitigation.aoflagger import get_rfi_mask
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess
from museek.rfi_mitigation.aoflagger_1d import gaussian_filter_1d
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import Synch_model_sm, git_version_info
from museek.util.tools import remove_outliers_zscore_mad, polynomial_flag_outlier
from museek.util.tools import (
    moving_median_masked,
    gaussian_filter_masked,
    interpolate_1d_masked_array,
)
from museek.visualiser import waterfall
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import pickle
import gc
import scipy
import warnings
import datetime


class GainCalibrationPlugin(AbstractPlugin):
    """Plugin to calibrtion the gain using synchrotron produced from pysm3"""

    def __init__(
        self,
        synch_model: [str],
        nside: int,
        beamsize: float,
        beam_frequency: float,
        frequency_high: float,
        frequency_low: float,
        flag_combination_threshold: int,
        do_store_context: bool,
        zscoreflag_threshold: float,
        polyflag_deg: int,
        polyflag_threshold: float,
        polyfit_deg: int,
        cali_method: str,
        window_movingmedian: int,
        nd_gausm_sigma: int,
        **kwargs,
    ):
        """
        Initialise the plugin
        :param synch_model: model used to create synchrotron sky
        :param nside: resolution parameter at which the synchrotron model is to be calculated
        :param beamsize: the beam fwhm used to smooth the Synch model [arcmin]
        :param beam_frequency: reference frequencies at which the beam fwhm are defined [MHz]
        :param frequency_high: high frequency cut
        :param frequency_low: low frequency cut
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        :param zscoreflag_threshold: threshold for flagging noise diode excess using modified zscore method
        :param polyflag_deg: degree of the polynomials used for fitting and flagging noise diode excess
        :param polyflag_threshold: threshold for flagging noise diode excess using polynomials fit
        :param polyfit_deg: degree of the polynomials used for fitting flagged noise diode excess
        :param cali_method: Method for calibration: 'corr' or 'rms'.
               'corr': Uses the correlation between the synchrotron model and visibility data for calibration.
               'rms': Uses the ratio of the RMS values of the synchrotron model and visibility data for calibration.
        :param window_movingmedian: The size of the window for the moving median calculation for frequency spectrum of noise diode signal
        :param nd_gausm_sigma: The size of the window for the Gaussian Smooth of Noise Diode Excess frequency spectrum.

        """
        super().__init__(**kwargs)
        self.synch_model = synch_model
        self.nside = nside
        self.beamsize = beamsize
        self.beam_frequency = beam_frequency
        self.frequency_high = frequency_high
        self.frequency_low = frequency_low
        self.flag_combination_threshold = flag_combination_threshold
        self.do_store_context = do_store_context
        self.zscoreflag_threshold = zscoreflag_threshold
        self.polyflag_deg = polyflag_deg
        self.polyflag_threshold = polyflag_threshold
        self.polyfit_deg = polyfit_deg
        self.cali_method = cali_method
        self.window_movingmedian = window_movingmedian
        self.nd_gausm_sigma = nd_gausm_sigma

    def set_requirements(self):
        """
        Set the requirements, the scanning data `scan_data`, a path to store results and the name of the data block.
        """
        self.requirements = [
            Requirement(location=ResultEnum.SCAN_DATA, variable="scan_data"),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable="output_path"),
            Requirement(location=ResultEnum.BLOCK_NAME, variable="block_name"),
            Requirement(
                location=ResultEnum.POINT_SOURCE_FLAG, variable="point_source_flag"
            ),
            Requirement(
                location=ResultEnum.NOISE_DIODE_EXCESS, variable="noise_diode_excess"
            ),
            Requirement(location=ResultEnum.NOISE_ON_INDEX, variable="noise_on_index"),
            Requirement(
                location=ResultEnum.FLAG_REPORT_WRITER, variable="flag_report_writer"
            ),
        ]

    def run(
        self,
        scan_data: TimeOrderedData,
        point_source_flag: np.ndarray,
        noise_diode_excess: np.ndarray,
        noise_on_index: np.ndarray,
        output_path: str,
        block_name: str,
        flag_report_writer: ReportWriter,
    ):
        """
        Run the gain calibration
        :return: the calibrated scan_data
        :param data: the time ordered scan data
        :param point_source_flag: mask for point sources
        :param noise_diode_excess: noise diode excess signal
        :param noise_on_index: the index of the noise firing timestamps
        :param output_path: path to store results
        :param block_name: name of the observation block
        :param flag_report_writer: report of the flag info
        """

        ########  load the visibility  ###########
        scan_data.load_visibility_flags_weights(polars="auto")
        initial_flags = scan_data.flags.combine(
            threshold=self.flag_combination_threshold
        )
        freq = (
            scan_data.frequencies.squeeze
        )  ####  the unit of scan_data.frequencies is Hz
        temperature = np.zeros(scan_data.visibility.array.shape)

        print(f"Producing synch sky: synch model {self.synch_model} used")
        synch = Synch_model_sm(
            scan_data, self.synch_model, self.nside, self.beamsize, self.beam_frequency
        )

        #######  loop for each receiver   ########
        for i_receiver, receiver in enumerate(scan_data.receivers):

            visibility_recv = scan_data.visibility.get(recv=i_receiver).squeeze
            initial_flag = initial_flags.get(recv=i_receiver).squeeze

            ########  fit the noise diode on - off, and normalise the trend in time  #########
            if initial_flag.all():
                noise_excess_recv_fit = np.ones(visibility_recv.shape)
            else:
                noise_excess_recv = noise_diode_excess[:, :, i_receiver]
                noise_excess_recv_fit = np.ones(visibility_recv.shape)
                for i_timestamp in np.arange(noise_excess_recv.shape[0]):
                    if noise_excess_recv.mask[i_timestamp, :].all():
                        pass
                    else:
                        noise_excess_time = noise_excess_recv[i_timestamp, :].copy()
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ###  calculate the moving_median of the noise diode on - off signal
                            noise_excess_time_mf = moving_median_masked(
                                noise_excess_time, window_size=self.window_movingmedian
                            )
                            ###  flagging outliers
                            noise_excess_time_residual = (
                                noise_excess_time_mf - noise_excess_time
                            )
                            noise_excess_time.mask = remove_outliers_zscore_mad(
                                noise_excess_time_residual.data,
                                noise_excess_time_residual.mask,
                                self.zscoreflag_threshold,
                            )
                            ###  interpolating masked regions
                            noise_excess_time = interpolate_1d_masked_array(
                                noise_excess_time, kind="linear"
                            )
                            ###  gaussian smoothing
                            noise_excess_recv[i_timestamp] = (
                                scipy.ndimage.gaussian_filter(
                                    noise_excess_time,
                                    sigma=self.nd_gausm_sigma,
                                    mode="nearest",
                                )
                            )

                for i_freq in np.arange(noise_excess_recv.shape[1]):
                    if noise_excess_recv.mask[:, i_freq].all():
                        pass
                    else:
                        noise_excess_freq = noise_excess_recv[:, i_freq].copy()
                        noise_excess_freq.mask = remove_outliers_zscore_mad(
                            noise_excess_freq.data,
                            noise_excess_freq.mask,
                            self.zscoreflag_threshold,
                        )
                        noise_excess_freq.mask, p_fit = polynomial_flag_outlier(
                            noise_on_index,
                            noise_excess_freq.data,
                            noise_excess_freq.mask,
                            self.polyflag_deg,
                            self.polyflag_threshold,
                        )
                        p_poly = np.polyfit(
                            noise_on_index[~noise_excess_freq.mask],
                            noise_excess_freq.data[~noise_excess_freq.mask],
                            deg=self.polyfit_deg,
                        )
                        noise_excess_recv_fit[:, i_freq] = np.polyval(
                            p_poly, np.arange(visibility_recv.shape[0])
                        )

            visibility_recv = visibility_recv / noise_excess_recv_fit
            del noise_excess_recv_fit
            gc.collect()

            #####  update the mask to avoid the incontinuous in the std along frequency ######
            select_freq = np.all(
                initial_flag, axis=0
            )  # select the all-timepoint masked frequency points and ingnore them in the mask_fraction calculation
            mask_fraction = np.mean(initial_flag[:, ~select_freq], axis=1)
            time_points_to_mask = (
                mask_fraction > 0.05
            )  # Find time points where the mask fraction is greater than 0.05

            i_antenna = scan_data.antenna_index_of_receiver(receiver=receiver)
            mask_update = initial_flag.copy() + point_source_flag[:, :, i_antenna]
            mask_update[time_points_to_mask, :] = (
                True  # For those time points, mask all frequency points
            )

            visibility_recv = np.ma.masked_array(visibility_recv, mask=mask_update)
            synch_recv = np.ma.masked_array(synch[:, :, i_receiver], mask=mask_update)

            if self.cali_method == "rms":
                ######  calculate std  ######
                vis_mean_time = np.ma.mean(visibility_recv, axis=0, keepdims=True)
                visibility_recv_norm = visibility_recv.data / vis_mean_time
                visibility_recv_norm = np.ma.masked_array(
                    visibility_recv_norm, mask=mask_update
                )
                visrms_in_time = np.ma.std(visibility_recv_norm, axis=0)
                synchrms_in_time = np.ma.std(synch_recv, axis=0)
                gain = visrms_in_time / synchrms_in_time

                temperature[:, :, i_receiver] = visibility_recv_norm.data / (
                    gain[np.newaxis, :]
                )

                del visibility_recv_norm
                gc.collect()

            elif self.cali_method == "corr":
                #######  calculate corr  ######
                vis_mean_time = np.ma.mean(visibility_recv, axis=0, keepdims=True)
                vis_synch_sum = np.ma.sum(
                    (visibility_recv - vis_mean_time)
                    * (synch_recv - np.ma.mean(synch_recv, axis=0, keepdims=True)),
                    axis=0,
                )
                synch_synch_sum = np.ma.sum(
                    (synch_recv - np.ma.mean(synch_recv, axis=0, keepdims=True)) ** 2,
                    axis=0,
                )
                gain = vis_synch_sum / synch_synch_sum

                temperature[:, :, i_receiver] = visibility_recv.data / (
                    gain[np.newaxis, :]
                )

        #########  select the frequency region we want to use  #######
        freqlow_index = np.argmin(np.abs(freq / 10.0**6 - self.frequency_low))
        freqhigh_index = np.argmin(np.abs(freq / 10.0**6 - self.frequency_high))
        temperature = temperature[:, freqlow_index:freqhigh_index, :]
        freq_select = freq[freqlow_index:freqhigh_index]
        point_source_flag = point_source_flag[:, freqlow_index:freqhigh_index, :]

        temperature = np.ma.masked_array(
            temperature, mask=initial_flags.array[:, freqlow_index:freqhigh_index, :]
        )

        ################  combine HH and VV, combine the mask of HH and VV firstly ###################
        temperature_antennas = []
        mask_antennas = []
        antenna_list = scan_data._antenna_name_list
        receivers_list = [
            str(receiver) for i_receiver, receiver in enumerate(scan_data.receivers)
        ]

        for antenna in antenna_list:
            indices = [
                index
                for index, receiver in enumerate(receivers_list)
                if antenna in receiver
            ]

            selected_mask = [temperature.mask[:, :, i] for i in indices]
            mask_antennas.append(np.sum(selected_mask, axis=0))

            selected_vis = [temperature.data[:, :, i] for i in indices]
            temperature_antennas.append(np.mean(selected_vis, axis=0))

        temperature_antennas = np.ma.masked_array(
            temperature_antennas, mask=mask_antennas
        )
        temperature_antennas = temperature_antennas.transpose(1, 2, 0)

        calibrated_data_flag_name_list = ["HH_VV_combined"]
        calibrated_data_flag = []
        calibrated_data_flag.append(temperature_antennas.mask)

        self.set_result(
            result=Result(
                location=ResultEnum.CALIBRATED_VIS,
                result=temperature_antennas,
                allow_overwrite=True,
            )
        )
        self.set_result(
            result=Result(
                location=ResultEnum.FREQ_SELECT,
                result=freq_select,
                allow_overwrite=True,
            )
        )
        self.set_result(
            result=Result(
                location=ResultEnum.POINT_SOURCE_FLAG,
                result=point_source_flag,
                allow_overwrite=True,
            )
        )
        self.set_result(
            result=Result(
                location=ResultEnum.CALIBRATED_VIS_FLAG,
                result=calibrated_data_flag,
                allow_overwrite=True,
            )
        )
        self.set_result(
            result=Result(
                location=ResultEnum.CALIBRATED_VIS_FLAG_NAME_LIST,
                result=calibrated_data_flag_name_list,
                allow_overwrite=True,
            )
        )

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = [
            "...........................",
            "Running GainCalibrationPlugin with "
            + f"MuSEEK version: {branch} ({commit})",
            "Finished at " + current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        flag_report_writer.write_to_report(lines)

        if self.do_store_context:
            context_file_name = "gain_calibration_plugin.pickle"
            self.store_context_to_disc(
                context_file_name=context_file_name, context_directory=output_path
            )
