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

class AoflaggerPlugin(AbstractParallelJoblibPlugin):
    """ Plugin to calculate RFI flags using the aoflagger algorithm and to post-process them. """

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
        self.do_store_context = do_store_context
        self.report_file_name = 'flag_report.md'

    def set_requirements(self):
        """
        Set the requirements, the scanning data `scan_data`, a path to store results and the name of the data block.
        """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')]

    def map(self,
            scan_data: TimeOrderedData,
            flag_report_writer: ReportWriter,
            output_path: str,
            block_name: str) \
            -> Generator[tuple[str, DataElement, FlagElement], None, None]:
        """
        Yield a `tuple` of the results path for one receiver, the scanning visibility data for one receiver and the
        initial flags for one receiver.
        :param scan_data: time ordered data containing the scanning part of the observation
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        :param block_name: name of the data block, not used here but for setting results
        """
        scan_data.load_visibility_flags_weights()
        initial_flags = scan_data.flags.combine(threshold=self.flag_combination_threshold)

        for i_receiver, receiver in enumerate(scan_data.receivers):
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            visibility = scan_data.visibility.get(recv=i_receiver)
            initial_flag = initial_flags.get(recv=i_receiver)
            yield receiver_path, visibility, initial_flag

    def run_job(self, anything: tuple[str, DataElement, FlagElement]) -> FlagElement:
        """
        Run the Aoflagger algorithm and post-process the result. Done for one receiver at a time.
        :param anything: `tuple` of the output path, the visibility and the initial flag
        :return: rfi mask as `FlagElement`
        """
        receiver_path, visibility, initial_flag = anything
        rfi_flag = get_rfi_mask(time_ordered=visibility,
                                mask=initial_flag,
                                mask_type=self.mask_type,
                                first_threshold=self.first_threshold,
                                threshold_scales=self.threshold_scales,
                                output_path=receiver_path,
                                smoothing_window_size=self.smoothing_kernel,
                                smoothing_sigma=self.smoothing_sigma)
        return self.post_process_flag(flag=rfi_flag, initial_flag=initial_flag)

    def gather_and_set_result(self,
                              result_list: list[FlagElement],
                              scan_data: TimeOrderedData,
                              flag_report_writer: ReportWriter,
                              output_path: str,
                              block_name: str):
        """
        Combine the `FlagElement`s in `result_list` into a new flag and set that as a result.
        :param result_list: `list` of `FlagElement`s created from the RFI flagging
        :param scan_data: `TimeOrderedData` containing the scanning part of the observation
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        :param block_name: name of the observation block
        """
        new_flag = FlagFactory().from_list_of_receiver_flags(list_=result_list)
        scan_data.flags.add_flag(flag=new_flag)

        receivers_list, flag_percent = flag_percent_recv(scan_data)
        lines = ['...........................', 'Running AoflaggerPlugin...', 'The flag fraction for each receiver: '] + [f'{x}  {y}' for x, y in zip(receivers_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        waterfall(scan_data.visibility.get(recv=0),
                  scan_data.flags.get(recv=0),
                  cmap='gist_ncar')
        plt.xlabel('time stamps')
        plt.ylabel('channels')
        plt.savefig(os.path.join(output_path, 'rfi_mitigation_result_receiver_0.png'), dpi=1000)
        plt.close()

        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data, allow_overwrite=True))
        if self.do_store_context:
            context_file_name = 'aoflagger_plugin.pickle'
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
