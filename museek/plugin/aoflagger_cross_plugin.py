import datetime
import os
from collections.abc import Generator

import numpy as np
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from matplotlib import pyplot as plt

from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.flag_element import FlagElement
from museek.flag_factory import FlagFactory
from museek.rfi_mitigation.aoflagger_cross import get_rfi_mask_cross
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import flag_percent_recv, git_version_info
from museek.visualiser import waterfall


class AoflaggerCrossPlugin(AbstractParallelJoblibPlugin):
    """Plugin to calculate RFI flags for cross polarizations, using the aoflagger algorithm and to post-process them."""

    def __init__(
        self,
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
        **kwargs,
    ):
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
        self.report_file_name = "flag_report.md"

    def set_requirements(self):
        """
        Set the requirements, the scanning data `scan_data`, a path to store results and the name of the data block.
        """
        self.requirements = [
            Requirement(location=ResultEnum.SCAN_DATA, variable="scan_data"),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable="output_path"),
            Requirement(location=ResultEnum.BLOCK_NAME, variable="block_name"),
            Requirement(
                location=ResultEnum.FLAG_REPORT_WRITER, variable="flag_report_writer"
            ),
            Requirement(
                location=ResultEnum.POINT_SOURCE_FLAG, variable="point_source_flag"
            ),
        ]

    def map(
        self,
        scan_data: TimeOrderedData,
        flag_report_writer: ReportWriter,
        point_source_flag: np.ndarray,
        output_path: str,
        block_name: str,
    ) -> Generator[tuple[str, DataElement, np.ndarray, np.ndarray], None, None]:
        """
        Yield a `tuple` of the results path for one receiver, the scanning visibility data for one receiver and the
        initial flags for one receiver.
        :param scan_data: time ordered data containing the scanning part of the observation
        :param flag_report_writer: report of the flag
        :param point_source_flag: flag for point sources
        :param output_path: path to store results
        :param block_name: name of the data block, not used here but for setting results
        """
        scan_data.load_visibility_flags_weights(polars="cross")
        initial_flags_cross = scan_data.flags_cross.combine(
            threshold=self.flag_combination_threshold
        )
        initial_flags = scan_data.flags.combine(
            threshold=self.flag_combination_threshold
        )

        for i_receiver, receiver in enumerate(scan_data.receivers):
            if not os.path.isdir(
                receiver_path := os.path.join(output_path, receiver.name)
            ):
                # os.makedirs(receiver_path)
                pass
            visibility_cross = scan_data.visibility_cross.get(recv=i_receiver)
            i_antenna = scan_data.antenna_index_of_receiver(receiver=receiver)
            initial_flag_cross = (
                initial_flags_cross.get(recv=i_receiver).squeeze
                + initial_flags.get(recv=i_receiver).squeeze
            )
            yield (
                receiver_path,
                visibility_cross,
                initial_flag_cross,
                point_source_flag[:, :, i_antenna],
            )

    def run_job(
        self, anything: tuple[str, DataElement, np.ndarray, np.ndarray]
    ) -> FlagElement:
        """
        Run the Aoflagger algorithm and post-process the result. Done for one receiver at a time.
        :param anything: `tuple` of the output path, the visibility and the initial flag
        :return: rfi mask as `FlagElement`
        """
        receiver_path, visibility_cross, initial_flag_cross, point_source_flag_recv = (
            anything
        )
        rfi_flag = get_rfi_mask_cross(
            time_ordered=visibility_cross,
            mask=FlagElement(
                array=(initial_flag_cross + point_source_flag_recv)[:, :, np.newaxis]
            ),
            mask_type=self.mask_type,
            first_threshold=self.first_threshold,
            threshold_scales=self.threshold_scales,
            output_path=receiver_path,
            smoothing_window_size=self.smoothing_kernel,
            smoothing_sigma=self.smoothing_sigma,
        )

        rfi_flag = rfi_flag.squeeze
        # unmask the point source, set rfi_flag to initial_flag where point_source_flag_recv is True, and then run post_process_flag (to avoid the point source flag being post processed)
        rfi_flag = np.where(point_source_flag_recv, initial_flag_cross, rfi_flag)
        initial_flag_cross = FlagElement(array=initial_flag_cross[:, :, np.newaxis])
        rfi_flag = FlagElement(array=rfi_flag[:, :, np.newaxis])
        return self.post_process_flag(flag=rfi_flag, initial_flag=initial_flag_cross)

    def gather_and_set_result(
        self,
        result_list: list[FlagElement],
        scan_data: TimeOrderedData,
        point_source_flag: np.ndarray,
        flag_report_writer: ReportWriter,
        output_path: str,
        block_name: str,
    ):
        """
        Combine the `FlagElement`s in `result_list` into a new flag and set that as a result.
        :param result_list: `list` of `FlagElement`s created from the RFI flagging
        :param scan_data: `TimeOrderedData` containing the scanning part of the observation
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        :param block_name: name of the observation block
        """
        new_flag = FlagFactory().from_list_of_receiver_flags(list_=result_list)
        scan_data.flags_cross.add_flag(flag=new_flag)

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        receivers_list, flag_percent = flag_percent_recv(scan_data)
        lines = [
            "...........................",
            "Running AoflaggerCrossPlugin with "
            + f"MuSEEK version: {branch} ({commit})",
            "Finished at " + current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "The flag fraction for each receiver: ",
        ] + [f"{x}  {y}" for x, y in zip(receivers_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        waterfall(
            scan_data.visibility_cross.get(recv=0),
            scan_data.flags_cross.get(recv=0),
            cmap="gist_ncar",
        )
        plt.xlabel("time stamps")
        plt.ylabel("channels")
        plt.savefig(
            os.path.join(output_path, "rfi_cross_mitigation_result_receiver_0.png"),
            dpi=1000,
        )
        plt.close()

        self.set_result(
            result=Result(
                location=ResultEnum.SCAN_DATA, result=scan_data, allow_overwrite=True
            )
        )
        if self.do_store_context:
            context_file_name = "aoflagger_cross_plugin.pickle"
            self.store_context_to_disc(
                context_file_name=context_file_name, context_directory=output_path
            )

    def post_process_flag(
        self, flag: FlagElement, initial_flag: FlagElement
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
        post_process = RfiPostProcess(
            new_flag=flag, initial_flag=initial_flag, struct_size=self.struct_size
        )
        post_process.binary_mask_dilation()
        post_process.binary_mask_closing()
        rfi_result = post_process.get_flag()

        # operations on the entire mask
        post_process = RfiPostProcess(
            new_flag=rfi_result + initial_flag,
            initial_flag=None,
            struct_size=self.struct_size,
        )
        post_process.flag_all_channels(
            channel_flag_threshold=self.channel_flag_threshold
        )
        post_process.flag_all_time_dumps(
            time_dump_flag_threshold=self.time_dump_flag_threshold
        )
        overall_result = post_process.get_flag()
        return overall_result
