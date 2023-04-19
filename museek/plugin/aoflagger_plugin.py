import os

from matplotlib import pyplot as plt

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.data_element import DataElement
from museek.enum.result_enum import ResultEnum
from museek.factory.data_element_factory import DataElementFactory
from museek.flag_element import FlagElement
from museek.flag_factory import FlagFactory
from museek.rfi_mitigation.aoflagger import get_rfi_mask
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess
from museek.time_ordered_data import TimeOrderedData
from museek.visualiser import waterfall


class AoflaggerPlugin(AbstractPlugin):
    """ Plugin to calculate RFI flags using the aoflagger algorithm and to post-process them. """

    def __init__(self,
                 first_threshold: float,
                 threshold_scales: list[float],
                 smoothing_kernel: tuple[int, int],
                 smoothing_sigma: tuple[float, float],
                 struct_size: tuple[int, int] | None,
                 channel_flag_threshold: float,
                 flag_combination_threshold: int):
        """
        Initialise the plugin
        :param first_threshold: initial threshold to be used for the aoflagger algorithm
        :param threshold_scales: list of sensitivities
        :param smoothing_kernel: smoothing kernel window size tuple for axes 0 and 1
        :param smoothing_sigma: smoothing kernel sigma tuple for axes 0 and 1
        :param struct_size: structure size for binary dilation, closing etc
        :param channel_flag_threshold: if the fraction of flagged channels exceeds this, all channels are flagged
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        """
        super().__init__()
        self.first_threshold = first_threshold
        self.threshold_scales = threshold_scales
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_sigma = smoothing_sigma
        self.struct_size = struct_size
        self.flag_combination_threshold = flag_combination_threshold
        self.data_element_factory = DataElementFactory()
        self.channel_flag_threshold = channel_flag_threshold

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def run(self, scan_data: TimeOrderedData, output_path: str):
        """
        Run the Aoflagger algorithm and post-process the result. Done for each receiver separately.
        :param scan_data: time ordered data containing the observation's scanning part
        :param output_path: path to store results
        """
        scan_data.load_visibility_flags_weights()
        initial_flags = scan_data.flags.combine(threshold=self.flag_combination_threshold)

        times = range(scan_data.visibility.shape[0])
        freqs = range(scan_data.visibility.shape[1])

        new_flag = FlagElement(flags=[FlagFactory().empty_flag(shape=scan_data.visibility.shape)])

        for i_receiver, receiver in enumerate(scan_data.receivers):
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            visibility = scan_data.visibility.get(recv=i_receiver, time=times, freq=freqs)
            initial_flag = initial_flags.get(recv=i_receiver, time=times, freq=freqs)
            rfi_flag = get_rfi_mask(time_ordered=visibility,
                                    mask=initial_flag,
                                    first_threshold=self.first_threshold,
                                    threshold_scales=self.threshold_scales,
                                    output_path=receiver_path,
                                    smoothing_window_size=self.smoothing_kernel,
                                    smoothing_sigma=self.smoothing_sigma)
            rfi_flag = self.post_process_flag(flag=rfi_flag, initial_flag=initial_flag)
            new_flag.insert_receiver_flag(flag=rfi_flag, i_receiver=i_receiver, index=0)

        scan_data.flags.add_flag(flag=new_flag)
        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data, allow_overwrite=True))

        waterfall(scan_data.visibility.get(recv=0, time=times, freq=freqs),
                  scan_data.flags.get(recv=0, time=times, freq=freqs),
                  cmap='gist_ncar',
                  norm='log')
        plt.savefig(os.path.join(output_path, 'rfi_mitigation_result_receiver_0.png'))

    def post_process_flag(self, flag: DataElement, initial_flag: DataElement) -> DataElement:
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
        post_process = RfiPostProcess(new_flag=flag, initial_flag=initial_flag, struct_size=self.struct_size)
        post_process.binary_mask_dilation()
        post_process.binary_mask_closing()
        post_process.flag_all_channels(channel_flag_threshold=self.channel_flag_threshold)
        return post_process.get_flag()
