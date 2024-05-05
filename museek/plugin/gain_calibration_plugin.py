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
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import Synch_model_sm
from museek.visualiser import waterfall
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import pickle

class GainCalibrationPlugin(AbstractPlugin):
    """ Plugin to calibrtion the gain using synchrotron produced from pysm3 """

    def __init__(self,
                 nside: int,
                 beamsize: float,
                 beam_frequency: float,
                 frequency_high: float,
                 frequency_low: float,
                 flag_combination_threshold: int,
                 do_store_context: bool,
                 **kwargs):
        """
        Initialise the plugin
        :param nside: resolution parameter at which the synchrotron model is to be calculated
        :param beamsize: the beam fwhm used to smooth the Synch model [arcmin]
        :param beam_frequency: reference frequencies at which the beam fwhm are defined [MHz]
        :param frequency_high: high frequency cut 
        :param frequency_low: low frequency cut
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        """
        super().__init__(**kwargs)
        self.nside = nside
        self.beamsize = beamsize 
        self.beam_frequency = beam_frequency
        self.frequency_high = frequency_high
        self.frequency_low = frequency_low
        self.flag_combination_threshold = flag_combination_threshold
        self.do_store_context = do_store_context

    def set_requirements(self):
        """
        Set the requirements, the scanning data `scan_data`, a path to store results and the name of the data block.
        """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name')]

    def run(self, scan_data: TimeOrderedData, output_path: str, block_name: str):
        """
        Run the gain calibration
        :return: the calibrated scan_data
        :param data: the time ordered scan data
        :param output_path: path to store results
        :param block_name: name of the observation block
        """
        
        ########  load the visibility  ###########
        scan_data.load_visibility_flags_weights()
        initial_flags = scan_data.flags.combine(threshold=self.flag_combination_threshold)
        freq = scan_data.frequencies.squeeze    ####  the unit of scan_data.frequencies is Hz 
        temperature = np.zeros(scan_data.visibility.array.shape)

        synch = Synch_model_sm(scan_data, self.nside, self.beamsize, self.beam_frequency)

        #######  loop for each receiver   ########
        for i_receiver, receiver in enumerate(scan_data.receivers):

            visibility_recv = scan_data.visibility.get(recv=i_receiver).squeeze
            initial_flag = initial_flags.get(recv=i_receiver).squeeze

            #####  update the mask to avoid the incontinuous in the std along frequency ######
            select_freq = np.all(initial_flag, axis=0)  # select the all-timepoint masked frequency points and ingnore them in the mask_fraction calculation
            mask_fraction = np.mean(initial_flag[:,~select_freq], axis=1)
            time_points_to_mask = mask_fraction > 0.05  # Find time points where the mask fraction is greater than 0.05

            mask_update = initial_flag.copy()
            mask_update[time_points_to_mask, :] = True  # For those time points, mask all frequency points

            visibility_recv = np.ma.masked_array(visibility_recv, mask=mask_update)
            synch_recv = np.ma.masked_array(synch[:,:,i_receiver], mask=mask_update)

            vis_mean_time = np.ma.mean(visibility_recv, axis=0, keepdims=True)
            visibility_recv_norm = visibility_recv.data / vis_mean_time
            visibility_recv_norm = np.ma.masked_array(visibility_recv_norm, mask=mask_update)

            ######  calculate std  ######
            visrms_in_time = np.ma.std(visibility_recv_norm, axis=0)
            synchrms_in_time = np.ma.std(synch_recv, axis=0)
            gain = visrms_in_time / synchrms_in_time
            
            temperature[:,:,i_receiver] = visibility_recv_norm.data / (gain[np.newaxis,:])


        #########  select the frequency region we want to use  #######
        freqlow_index = np.argmin(np.abs(freq/10.**6 - self.frequency_low))
        freqhigh_index = np.argmin(np.abs(freq/10.**6 - self.frequency_high))
        temperature = temperature[:,freqlow_index:freqhigh_index,:]
        freq_select = freq[freqlow_index:freqhigh_index]

        temperature = np.ma.masked_array(temperature, mask=initial_flags.array[:,freqlow_index:freqhigh_index,:])

        temperature_antennas = []
        antenna_list = scan_data._antenna_name_list
        receivers_list = [str(receiver) for i_receiver, receiver in enumerate(scan_data.receivers)]

        for antenna in antenna_list:
            indices = [index for index, receiver in enumerate(receivers_list) if antenna in receiver]
            selected_vis = [temperature[:,:,i] for i in indices]
            temperature_antennas.append(np.ma.mean(selected_vis, axis=0))
        temperature_antennas = np.ma.masked_array(temperature_antennas)
        temperature_antennas = temperature_antennas.transpose(1, 2, 0)

        self.set_result(result=Result(location=ResultEnum.CALIBRATED_VIS, result=temperature_antennas, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.FREQ_SELECT, result=freq_select, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.COMBINED_FLAG, result=initial_flags.array[:,freqlow_index:freqhigh_index,:], allow_overwrite=True))

        if self.do_store_context:
            context_file_name = 'gain_calibration_plugin.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=output_path)

            arrays_dict = {
              'calibrated_visibility': temperature_antennas,
              'timestamps': scan_data.timestamps.array.squeeze(),
              'ra':scan_data.right_ascension.array.squeeze(),
              'dec':scan_data.declination.array.squeeze(),
              'freq':freq_select,
              'receivers_list':receivers_list,
              'antenna_list':antenna_list,
              }

            with open(output_path+block_name+'_calibrated_visibility.pkl', 'wb') as f:
                pickle.dump(arrays_dict, f)

