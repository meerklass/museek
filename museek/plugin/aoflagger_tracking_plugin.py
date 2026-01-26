import datetime
import os
from collections.abc import Generator

import numpy as np
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result

from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_element import FlagElement
from museek.flag_list import FlagList
from museek.rfi_mitigation.aoflagger import get_rfi_mask
from museek.rfi_mitigation.aoflagger_1d import get_rfi_mask_1d
from museek.rfi_mitigation.rfi_post_process import RfiPostProcess
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import consecutive_subsets, flag_percent_recv, git_version_info


class AoflaggerTrackingPlugin(AbstractParallelJoblibPlugin):
    """Plugin to calculate RFI flags using the aoflagger algorithm and to post-process them."""

    def __init__(
        self,
        mask_type: str,
        first_threshold: float,
        threshold_scales: list[float],
        first_threshold_flag_fraction: list[float],
        smoothing_kernel: tuple[int, int],
        smoothing_sigma: tuple[float, float],
        smoothing_kernel_flag_fraction: tuple[int, int],
        smoothing_sigma_flag_fraction: tuple[float, float],
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
        :param first_threshold_flag_fraction: initial threshold to be used for the aoflagger algorithm on the flagged fraction, after the first aoflagger on the track data
        :param threshold_scales: list of sensitivities
        :param smoothing_kernel: smoothing kernel window size tuple for axes 0 and 1
        :param smoothing_sigma: smoothing kernel sigma tuple for axes 0 and 1
        :param smoothing_kernel_flag_fraction: smoothing kernel window size
        :param smoothing_sigma_flag_fraction: smoothing kernel sigma
        :param struct_size: structure size for binary dilation, closing etc
        :param channel_flag_threshold: if the fraction of flagged channels exceeds this, all channels are flagged
        :param time_dump_flag_threshold: if the fraction of flagged time dumps exceeds this, all time dumps are flagged
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        """
        super().__init__(**kwargs)
        self.mask_type = mask_type
        self.first_threshold = first_threshold
        self.first_threshold_flag_fraction = first_threshold_flag_fraction
        self.threshold_scales = threshold_scales
        self.smoothing_kernel = smoothing_kernel
        self.smoothing_sigma = smoothing_sigma
        self.smoothing_kernel_flag_fraction = smoothing_kernel_flag_fraction
        self.smoothing_sigma_flag_fraction = smoothing_sigma_flag_fraction
        self.struct_size = struct_size
        self.flag_combination_threshold = flag_combination_threshold
        self.channel_flag_threshold = channel_flag_threshold
        self.time_dump_flag_threshold = time_dump_flag_threshold
        self.do_store_context = do_store_context
        self.report_file_name = "flag_report.md"
        self.data_element_factory = FlagElementFactory()

    def set_requirements(self):
        """
        Set the requirements, the scanning data `track_data`, a path to store results and the name of the data block.
        """
        self.requirements = [
            Requirement(location=ResultEnum.TRACK_DATA, variable="track_data"),
            Requirement(
                location=ResultEnum.CALIBRATOR_VALIDATED_PERIODS,
                variable="calibrator_validated_periods",
            ),
            Requirement(
                location=ResultEnum.CALIBRATOR_DUMP_INDICES,
                variable="calibrator_dump_indices",
            ),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable="output_path"),
            Requirement(location=ResultEnum.BLOCK_NAME, variable="block_name"),
            Requirement(
                location=ResultEnum.FLAG_REPORT_WRITER, variable="flag_report_writer"
            ),
        ]

    def map(
        self,
        track_data: TimeOrderedData,
        calibrator_validated_periods: list,
        calibrator_dump_indices: dict,
        flag_report_writer: ReportWriter,
        output_path: str,
        block_name: str,
    ) -> Generator[tuple[str, DataElement, np.ndarray, np.ndarray], None, None]:
        """
        Yield a `tuple` of the results path for one receiver, the scanning visibility data for one receiver and the
        initial flags for one receiver.
        :param track_data: time ordered data containing the scanning part of the observation
        :param calibrator_validated_periods: validated periods of single-dish calibrator scans
        :param calibrator_dump_indices: indices of validated periods of single-dish calibrator scans
        :param flag_report_writer: report of the flag
        :param point_source_flag: flag for point sources
        :param output_path: path to store results
        :param block_name: name of the data block, not used here but for setting results
        """
        track_data.load_visibility_flags_weights(polars="auto")
        initial_flags = track_data.flags.combine(
            threshold=self.flag_combination_threshold
        )
        freq = track_data.frequencies.squeeze  ####  the unit of frequencies is Hz
        dumps = np.array(
            track_data._dumps_of_scan_state()
        )  ### the dumps of track_data in original full data
        for i_receiver, receiver in enumerate(track_data.receivers):
            if not os.path.isdir(
                receiver_path := os.path.join(output_path, receiver.name)
            ):
                pass

            visibility_recv = track_data.visibility.get(recv=i_receiver)
            initial_flag_recv = initial_flags.get(recv=i_receiver)

            yield (
                receiver_path,
                calibrator_validated_periods,
                calibrator_dump_indices,
                visibility_recv,
                initial_flag_recv,
                freq,
                dumps,
            )

    def run_job(
        self, anything: tuple[str, DataElement, np.ndarray, np.ndarray]
    ) -> FlagElement:
        """
        Run the Aoflagger algorithm and post-process the result. Done for one receiver at a time.
        :param anything: `tuple` of the output path, the visibility and the initial flag
        :return: rfi mask as `FlagElement`
        """
        (
            receiver_path,
            calibrator_validated_periods,
            calibrator_dump_indices,
            visibility_recv,
            initial_flag_recv,
            freq,
            dumps,
        ) = anything

        new_flag = np.zeros(
            visibility_recv.squeeze.shape, dtype=bool
        )  ### initialize the new flag
        for periods in calibrator_validated_periods:
            calibrator_dump_indices_consecutive_subsets = consecutive_subsets(
                calibrator_dump_indices[periods]
            )  ####  split the dump indices into subsets, each subset refers to one pointing (in principle 7 pointing in total)
            for subset in calibrator_dump_indices_consecutive_subsets:
                select = np.isin(
                    dumps, subset
                )  # Boolean mask for this subset, obtain the index for subset in track_data
                vis_pointing = visibility_recv.squeeze[select]
                flag_pointing = initial_flag_recv.squeeze[select]
                rfi_flagvis = get_rfi_mask(
                    time_ordered=DataElement(array=vis_pointing[:, :, np.newaxis]),
                    mask=FlagElement(array=flag_pointing[:, :, np.newaxis]),
                    mask_type=self.mask_type,
                    first_threshold=self.first_threshold,
                    threshold_scales=self.threshold_scales,
                    output_path=receiver_path,
                    smoothing_window_size=self.smoothing_kernel,
                    smoothing_sigma=self.smoothing_sigma,
                )

                initial_flagvis = FlagElement(array=flag_pointing[:, :, np.newaxis])
                postprocess_flagvis = self.post_process_flag(
                    flag=rfi_flagvis, initial_flag=initial_flagvis
                )

                ###################    aoflagger on the flagged fraction   ############
                flag_fraction = np.mean(postprocess_flagvis.squeeze > 0, axis=0)
                flag_fraction_mask = flag_fraction > self.time_dump_flag_threshold
                rfi_flagfrac = get_rfi_mask_1d(
                    time_ordered=DataElement(
                        array=flag_fraction[:, np.newaxis, np.newaxis]
                    ),
                    mask=FlagElement(
                        array=flag_fraction_mask[:, np.newaxis, np.newaxis]
                    ),
                    mask_type="vis",
                    first_threshold=self.first_threshold_flag_fraction,
                    threshold_scales=self.threshold_scales,
                    output_path=receiver_path,
                    smoothing_window_size=self.smoothing_kernel_flag_fraction,
                    smoothing_sigma=self.smoothing_sigma_flag_fraction,
                )

                rfi_flagfrac_tile = np.tile(
                    rfi_flagfrac.squeeze, (vis_pointing.shape[0], 1)
                )
                initial_flagfrac_tile = np.tile(
                    flag_fraction_mask, (vis_pointing.shape[0], 1)
                )
                postprocess_flagfrac = self.post_process_flag(
                    flag=FlagElement(array=rfi_flagfrac_tile[:, :, np.newaxis]),
                    initial_flag=FlagElement(
                        array=initial_flagfrac_tile[:, :, np.newaxis]
                    ),
                )

                ################  combine the flags from aoflaggers on vis and flagged fraction
                new_flag[select, :] = (
                    postprocess_flagvis.squeeze + postprocess_flagfrac.squeeze
                )

        return new_flag

    def gather_and_set_result(
        self,
        result_list: list[dict],
        track_data: TimeOrderedData,
        calibrator_validated_periods: list,
        calibrator_dump_indices: dict,
        flag_report_writer: ReportWriter,
        output_path: str,
        block_name: str,
    ):
        """
        Combine the `FlagElement`s in `result_list` into a new flag and set that as a result.
        :param result_list: `list` of `FlagElement`s created from the RFI flagging
        :param track_data: `TimeOrderedData` containing the scanning part of the observation
        :param calibrator_validated_periods: validated periods of single-dish calibrator scans
        :param calibrator_dump_indices: indices of validated periods of single-dish calibrator scans
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        :param block_name: name of the observation block
        """

        result_list = np.array(result_list, dtype="bool").transpose(1, 2, 0)
        track_data.flags.add_flag(
            flag=FlagList.from_array(
                array=result_list, element_factory=self.data_element_factory
            )
        )

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        receivers_list, flag_percent = flag_percent_recv(track_data)
        lines = [
            "...........................",
            "Running AoflaggerTrackingPlugin with "
            + f"MuSEEK version: {branch} ({commit})",
            "Finished at " + current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "The flag fraction for each receiver: ",
        ] + [f"{x}  {y}" for x, y in zip(receivers_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        self.set_result(
            result=Result(
                location=ResultEnum.TRACK_DATA, result=track_data, allow_overwrite=True
            )
        )

        if self.do_store_context:
            context_file_name = "aoflagger_tracking_plugin.pickle"
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
