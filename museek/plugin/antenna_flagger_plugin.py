import numpy as np
from katpoint import Antenna

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.antenna_sanity.constant_elevation_scans import ConstantElevationScans
from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.flag_element import FlagElement
from museek.flag_factory import FlagFactory
from museek.flag_list import FlagList
from museek.time_ordered_data import TimeOrderedData
from museek.util.clustering import Clustering
from museek.util.tools import flag_percent_recv
from museek.util.report_writer import ReportWriter
import pickle
from scipy import ndimage

class AntennaFlaggerPlugin(AbstractPlugin):
    """ Plugin to flag misbehaving antennas. """

    def __init__(self,
                 elevation_std_threshold: float,
                 elevation_threshold: float,
                 elevation_flag_threshold: float,
                 outlier_threshold: float,
                 outlier_flag_threshold: float):
        """
        Initialise the plugin
        :param elevation_std_threshold: antennas with std of elevation reading exceeding this threshold are flagged [degree]
        :param elevation_threshold: time points with elevation reading deviations exceeding this threshold are flagged [degree]
        :param elevation_flag_threshold: if the fraction of flagged elevation exceeds this, all time dumps are flagged 
        :param outlier_threshold: threshold in degrees azimuth and elevation used to identify outliers
        :param outlier_flag_threshold: if the flag fraction of outlier flagging exceeds this, all time dumps are flagged
        """
        super().__init__()
        self.elevation_std_threshold = elevation_std_threshold
        self.elevation_threshold = elevation_threshold
        self.elevation_flag_threshold = elevation_flag_threshold
        self.outlier_threshold = outlier_threshold
        self.outlier_flag_threshold = outlier_flag_threshold
        self.report_file_name = 'flag_report.md'

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name')]

    def run(self, scan_data: TimeOrderedData, track_data: TimeOrderedData, flag_report_writer: ReportWriter, output_path: str, block_name: str):
        """
        Run the plugin
        :param scan_data: time ordered data of the scanning part
        :param track_data: time ordered data of the tracking part
        :param flag_report_writer: report_writer of the flag
        :param output_path: path to store results
        :param block_name: name of the observation block
        """
        scan_data.load_visibility_flags_weights()
        #self.flag_for_elevation(data=scan_data)
        self.flag_for_elevation_TOD(data=scan_data)
        track_data.load_visibility_flags_weights()
        #for data in [scan_data, track_data]:
        #    self.flag_outlier_antennas(data=data)
        for data in [scan_data, track_data]:
            self.flag_outlier_antennas_TOD(data=data)

        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data))
        self.set_result(result=Result(location=ResultEnum.TRACK_DATA, result=track_data))


        for data, label in zip([scan_data, track_data], ['scan_data', 'track_data']):
            receivers_list, flag_percent = flag_percent_recv(data)
            lines = ['...........................', 'Running AntennaFlaggerPlugin...', 'The '+label+' flag fraction for each receiver: '] + [f'{x}  {y}' for x, y in zip(receivers_list, flag_percent)]
            flag_report_writer.write_to_report(lines)


    def flag_outlier_antennas(self, data: TimeOrderedData):
        """ Add a new flag to `data` to exclude antennas with non-constant elevation readings. """
        shape = data.visibility.shape
        new_flag = FlagList(flags=[FlagFactory().empty_flag(shape=shape)])
        full_flag = FlagElement(array=np.ones((shape[0], shape[1], 1)))
        _, antennas = self.outlier_antenna_indices(data=data, distance_threshold=self.outlier_threshold)
        for antenna in antennas:
            print(f'Outliers: flagged antenna {antenna.name}.')
            i_receiver_list = data.receiver_indices_of_antenna(antenna)
            for i_receiver in i_receiver_list:
                new_flag.insert_receiver_flag(flag=full_flag, i_receiver=i_receiver, index=0)
        data.flags.add_flag(flag=new_flag)


    def flag_outlier_antennas_TOD(self, data: TimeOrderedData):
        """ Add a new flag to `data` to exclude outlier antennas, calculating at each time point """
        shape = data.visibility.shape
        new_flag_array = np.zeros((shape[0], len(data.antennas)))
        ######  flag outlier antennas at each time stamp #####
        for i_timestamps, timestamps in enumerate(data.timestamps):

            antenna_elevation = data.azimuth.get(time=i_timestamps).squeeze
            antenna_azimuth = data.azimuth.get(time=i_timestamps).squeeze

            feature = np.asarray([antenna_elevation,
                                  antenna_azimuth]).T
            outlier_indices = Clustering().iterative_outlier_indices(feature_vector=feature,
                                                                 distance_threshold=self.outlier_threshold)

            new_flag_array[i_timestamps,outlier_indices] = 1.

        #####  arrange the flag for each receiver  #####
        new_flag = FlagList(flags=[FlagFactory().empty_flag(shape=shape)])
        for i_antenna, antenna in enumerate(data.antennas):
            if np.mean(new_flag_array[:,i_antenna]) > self.outlier_flag_threshold:
                outlier_antenna_flag = np.ones((shape[0]))
                print(f'flag fraction of outlier antenna flagging exceeds outlier_flag_threshold: flagged antenna {antenna.name}.')
            else:
                outlier_antenna_flag = new_flag_array[:,i_antenna]

            i_receiver_list = data.receiver_indices_of_antenna(antenna)
            for i_receiver in i_receiver_list:
                flag_array = np.repeat(outlier_antenna_flag, shape[1]).reshape((shape[0], shape[1], 1))
                new_flag.insert_receiver_flag(flag=DataElement(array=flag_array), i_receiver=i_receiver, index=0)

        data.flags.add_flag(flag=new_flag)


    @staticmethod
    def outlier_antenna_indices(data: TimeOrderedData, distance_threshold: float) -> tuple[list[int], list[Antenna]]:
        """
        Return `Antenna`s and indices of `Antenna`s in `data` with coordinates that are outliers wrt the other antennas
        using `distance_threshold`.
        """
        antenna_elevation_mean = data.elevation.mean(axis=0).squeeze
        antenna_azimuth_max = data.azimuth.max(axis=0).squeeze
        antenna_azimuth_min = data.azimuth.min(axis=0).squeeze
        antenna_azimuth_start = data.azimuth.get(time=0).squeeze
        antenna_azimuth_end = data.azimuth.get(time=data.timestamps.shape[0] - 1).squeeze
        antenna_azimuth_middle = data.azimuth.get(time=data.timestamps.shape[0] // 2).squeeze

        feature = np.asarray([antenna_elevation_mean,
                              antenna_azimuth_min,
                              antenna_azimuth_max,
                              antenna_azimuth_start,
                              antenna_azimuth_end,
                              antenna_azimuth_middle]).T
        outlier_indices = Clustering().iterative_outlier_indices(feature_vector=feature,
                                                                 distance_threshold=distance_threshold)
        outlier_antennas = [data.antennas[index] for index in outlier_indices]
        return outlier_indices, outlier_antennas


    def flag_for_elevation(self, data: TimeOrderedData):
        """ Add a new flag to `data` to exclude antennas with non-constant elevation readings. """
        shape = data.visibility.shape
        new_flag = FlagList(flags=[FlagFactory().empty_flag(shape=shape)])
        full_flag = DataElement(array=np.ones((shape[0], shape[1], 1)))
        for antenna in ConstantElevationScans.get_antennas_with_non_constant_elevation(
                data=data,
                threshold=self.elevation_std_threshold
        ):
            print(f'Non-constant elevation: flagged antenna {antenna.name}.')
            i_receiver_list = data.receiver_indices_of_antenna(antenna)
            for i_receiver in i_receiver_list:
                new_flag.insert_receiver_flag(flag=full_flag, i_receiver=i_receiver, index=0)
        data.flags.add_flag(flag=new_flag)


    def flag_for_elevation_TOD(self, data: TimeOrderedData):
        """ Add a new flag to `data` to mask the time points where elevation is deviated from the median elevation, and then exclude antennas with non-constant elevation readings."""
        shape = data.visibility.shape
        new_flag = FlagList(flags=[FlagFactory().empty_flag(shape=shape)])
        full_flag = DataElement(array=np.ones((shape[0], shape[1], 1)))
        for i_antenna, antenna in enumerate(data.antennas):
            antenna_elevation = data.elevation.get(recv=i_antenna).squeeze
            bad_elevation_flag = abs(antenna_elevation - np.median(antenna_elevation)) > self.elevation_threshold 

            ##### dilate the flag #####
            struct = np.ones(5, dtype=bool)
            bad_elevation_flag = ndimage.binary_dilation(bad_elevation_flag, structure=struct, iterations=2)

            standard_deviation = np.std(antenna_elevation[~bad_elevation_flag])
            
            if np.mean(bad_elevation_flag) > self.elevation_flag_threshold:
                bad_elevation_flag = np.ones((shape[0]))
                print(f'fraction of flagged elevation exceeds elevation_flag_threshold: flagged antenna {antenna.name}.')
            else:
                pass

            i_receiver_list = data.receiver_indices_of_antenna(antenna)
            for i_receiver in i_receiver_list:
                if standard_deviation > self.elevation_std_threshold:
                    new_flag.insert_receiver_flag(flag=full_flag, i_receiver=i_receiver, index=0)
                    print(f'Non-constant elevation: flagged antenna {antenna.name}.')
                else:
                    flag_array = np.repeat(bad_elevation_flag, shape[1]).reshape((shape[0], shape[1], 1))
                    new_flag.insert_receiver_flag(flag=DataElement(array=flag_array), i_receiver=i_receiver, index=0)

        data.flags.add_flag(flag=new_flag)

