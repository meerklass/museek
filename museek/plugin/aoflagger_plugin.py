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
from museek.visualiser import waterfall


class AoflaggerPlugin(AbstractParallelJoblibPlugin):
    """ Plugin to calculate RFI flags using the aoflagger algorithm and to post-process them. """

    def __init__(self,
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
        self.first_threshold = first_threshold
        self.threshold_scales = threshold_scales
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_sigma = smoothing_sigma
        self.struct_size = struct_size
        self.flag_combination_threshold = flag_combination_threshold
        self.channel_flag_threshold = channel_flag_threshold
        self.time_dump_flag_threshold = time_dump_flag_threshold
        self.do_store_context = do_store_context

    def set_requirements(self):
        """ Set the requirements, the entire `data`, a path to store results and the name of the data block. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name')]

    def map(self,
            data: TimeOrderedData,
            output_path: str,
            block_name: str) \
            -> Generator[tuple[str, DataElement, FlagElement], None, None]:
        """
        Yield a `tuple` of the results path for one receiver, the visibility data for one receiver and the
        initial flags for one receiver.
        :param data: time ordered data containing the entire observation
        :param output_path: path to store results
        :param block_name: name of the data block
        """
        data.load_visibility_flags_weights()
        initial_flags = data.flags.combine(threshold=self.flag_combination_threshold)

        for i_receiver, receiver in enumerate(data.receivers):
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            visibility = data.visibility.get(recv=i_receiver)
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
                                first_threshold=self.first_threshold,
                                threshold_scales=self.threshold_scales,
                                output_path=receiver_path,
                                smoothing_window_size=self.smoothing_kernel,
                                smoothing_sigma=self.smoothing_sigma)
        return self.post_process_flag(flag=rfi_flag, initial_flag=initial_flag)

    def gather_and_set_result(self,
                              result_list: list[FlagElement],
                              data: TimeOrderedData,
                              output_path: str,
                              block_name: str):
        """
        Combine the `FlagElement`s in `result_list` into a new flag and set that as a result.
        :param result_list: `list` of `FlagElement`s created from the RFI flagging
        :param data: `TimeOrderedData` containing the entire observation
        :param output_path: path to store results
        :param block_name: name of the observation block
        """
        new_flag = FlagFactory().from_list_of_receiver_flags(list_=result_list)
        data.flags.add_flag(flag=new_flag)

        waterfall(data.visibility.get(recv=0),
                  data.flags.get(recv=0),
                  cmap='gist_ncar')
        plt.savefig(os.path.join(output_path, 'rfi_mitigation_result_receiver_0.png'), dpi=1000)
        plt.close()

        self.set_result(result=Result(location=ResultEnum.DATA, result=data, allow_overwrite=True))
        if self.do_store_context:
            context_file_name = 'aoflagger_plugin.pickle'
            context_folder = os.path.join(ROOT_DIR, 'results/')
            context_directory = os.path.join(context_folder, f'{block_name}/')
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=context_directory)

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
