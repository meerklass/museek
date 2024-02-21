import os

import numpy as np
from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_list import FlagList
from museek.time_ordered_data import TimeOrderedData
from museek.visualiser import waterfall


class RawdataFlaggerPlugin(AbstractPlugin):
    """ Plugin to flag raw data with values below a minimum """

    def __init__(self,
                 flag_lower_threshold: float):
        """
        Initialise
        :param flag_lower_threshold: lower threshold to flag the data, 
                                     it relates to raw correlator units without any normalisation.
        """
        super().__init__()
        self.data_element_factory = FlagElementFactory()
        self.flag_lower_threshold = flag_lower_threshold

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, data: TimeOrderedData, output_path: str):
        """
        Flag raw data with values below a minimum
        :param data: time ordered data of the entire block
        :param output_path: path to store results
        """
        data.load_visibility_flags_weights()
        new_flag = np.zeros(data.visibility.shape, dtype=bool)
        new_flag[data.visibility.array < self.flag_lower_threshold] = True

        data.flags.add_flag(flag=FlagList.from_array(array=new_flag, element_factory=self.data_element_factory))
        self.set_result(result=Result(location=ResultEnum.DATA, result=data, allow_overwrite=True))

