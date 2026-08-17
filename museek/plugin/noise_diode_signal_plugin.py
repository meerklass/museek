import datetime
import warnings
from collections.abc import Generator

import numpy as np
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result

from museek.enums.result_enum import ResultEnum
from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_list import FlagList
from museek.noise_diode import NoiseDiode
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import (
    consecutive_subsets,
    flag_percent_recv,
    git_version_info,
    remove_outliers_zscore_mad,
)


class NoiseDiodeSignalPlugin(AbstractParallelJoblibPlugin):
    """
    Plugin to compute the noise diode excess as a function of time, per frequency and per receiver,
    from the calibrator track data.

    Design (a clean rewrite of `NoiseDiodeExcessPlugin`):

    1. The firing dumps are identified directly from the `noise_diode_on` flag layer that
       `NoiseDiodeFlaggerPlugin` produced upstream.
    2. Every flag layer is combined into a single mask, and the firing dumps are then *unflagged*
       in that mask. Each firing is judged purely by its neighbouring off-dumps: if an off-dump is
       flagged (per frequency, per receiver) the firing is masked at that frequency/receiver. The
       firing dump's own flags are intentionally ignored (it was unflagged).
    3. The plugin is **pointing-aware**: the off-baseline dumps (one before and one after the firing)
       must belong to the *same* calibrator pointing. A firing at the start or end of a pointing,
       whose bracket would fall into a neighbouring pointing, is dropped (fully masked). This is the
       key improvement over `NoiseDiodeExcessPlugin`, which only guarded the track-array edges.
    4. The two-dump firing case (the diode fires across two consecutive dumps) is handled by summing
       the two on-dumps and subtracting twice the off-baseline; the firing is assigned a
       ratio-weighted time that falls between the two dumps.

    The excess for one firing is ``on - off * n_on``, where ``on`` is the sum over the firing's
    on-dump(s), ``off`` is the mean of the two bracketing off-dumps, and ``n_on`` is the number of
    on-dumps (1 or 2). Individual firings whose excess is a per-receiver outlier are additionally
    masked, and a receiver whose overall median excess is below ``noise_diode_excess_lowlim`` has all
    of its firings masked. The ``noise_diode_bad_behavior`` flag carries every mask applied to the
    excess (edge firings, flagged off-neighbours, per-firing outliers and the low-limit check),
    projected onto the firing on-dumps only -- off dumps are never flagged.

    The excess outputs (``NOISE_DIODE_EXCESS``, ``NOISE_DIODE_EXCESS_AVERAGE`` and
    ``NOISE_ON_TIMESTAMP``) are dicts keyed by calibrator period (e.g. ``before_scan``/``after_scan``),
    each carrying that period's per-firing excess, robust (masked-median) time-averaged excess, and
    firing times respectively. ``NOISE_DIODE_DUTY_CYCLE`` is a period-independent scalar
    (``duration / dump_period``); divide ``excess / gain`` by it to get the noise diode temperature.
    """

    def __init__(
        self,
        flag_combination_threshold: int,
        zscoreflag_threshold: float,
        noise_diode_excess_lowlim: float,
        max_masked_fraction: float = 0.6,
        do_store_context: bool = False,
        prefer: str = "threads",
        **kwargs,
    ):
        """
        Initialise the plugin.
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        :param zscoreflag_threshold: modified-zscore (MAD) threshold for masking individual firings
            whose frequency-median excess is far from the receiver's robust average over firings
        :param noise_diode_excess_lowlim: threshold for flagging the mean value of noise diode excess of receivers
        :param max_masked_fraction: if, after per-firing outlier removal, more than this fraction of
            a receiver's firings are masked, all of them are masked. Set to `1.0` to disable.
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        :param prefer: joblib backend preference, defaults to 'threads' to avoid process
            overhead when parallelising the pure-numpy run_job calls
        """
        super().__init__(prefer=prefer, **kwargs)
        self.flag_combination_threshold = flag_combination_threshold
        self.zscoreflag_threshold = zscoreflag_threshold
        self.noise_diode_excess_lowlim = noise_diode_excess_lowlim
        self.max_masked_fraction = max_masked_fraction
        self.do_store_context = do_store_context
        self.report_file_name = "flag_report.md"
        self.data_element_factory = FlagElementFactory()
        # receiver-independent firing structure, filled in `map()`
        self.noise_on_timestamp = None
        self.firing_on_dumps = (
            None  # list of on-dump (relative) index arrays, one per firing
        )
        self.firing_periods = (
            None  # period label per firing, aligned with the firing axis
        )
        self.noise_diode_duty = (
            None  # diode on-fraction per dump (duration / dump_period)
        )

    def set_requirements(self):
        """Set the requirements."""
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
            Requirement(
                location=ResultEnum.FLAG_REPORT_WRITER, variable="flag_report_writer"
            ),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable="output_path"),
        ]

    def map(
        self,
        track_data: TimeOrderedData,
        calibrator_validated_periods: list,
        calibrator_dump_indices: dict,
        flag_report_writer: ReportWriter,
        output_path: str,
    ) -> Generator[tuple[np.ndarray, np.ndarray, list, np.ndarray], None, None]:
        """
        Compute the receiver-independent firing structure once, then yield per-receiver visibility,
        the combined flag mask, the firing records and the noise-on boolean for each receiver.
        :param track_data: time ordered data containing the calibrator tracking part of the observation
        :param calibrator_validated_periods: validated periods of single-dish calibrator scans
        :param calibrator_dump_indices: indices of validated periods of single-dish calibrator scans
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        """
        track_data.load_visibility_flags_weights(polars="auto")

        # Combine all flag layers (incl. `noise_diode_on` and `aoflagger_tracking`) into one mask.
        initial_flags = track_data.flags.combine(
            threshold=self.flag_combination_threshold
        )

        # Firing dumps are read directly from the `noise_diode_on` flag (uniform over freq/recv).
        if "noise_diode_on" not in track_data.flags.flag_names:
            raise ValueError(
                "NoiseDiodeSignalPlugin requires a 'noise_diode_on' flag; "
                "run NoiseDiodeFlaggerPlugin first."
            )
        nd_i = track_data.flags.flag_names.index("noise_diode_on")
        # index a single freq/recv column to avoid materialising the full stacked flag array
        noise_on = (
            np.asarray(track_data.flags.get(freq=0, recv=0).array[nd_i])
            .squeeze()
            .astype(bool)
        )

        # Map each validated calibrator pointing to a contiguous block of relative track indices,
        # keeping the period each pointing belongs to.
        abs_dumps = np.array(track_data._dumps_of_scan_state())
        pointings = []
        pointing_periods = []
        for period in calibrator_validated_periods:
            subset_list = calibrator_dump_indices[period]
            if len(subset_list) == 0:
                continue
            for subset in consecutive_subsets(sorted(subset_list)):
                rel = np.where(np.isin(abs_dumps, subset))[0]
                if len(rel) > 0:
                    pointings.append(rel)
                    pointing_periods.append(period)

        # Within each pointing, split the noise-on dumps into firing groups (1 dump, or 2+ if the
        # firing straddles two dumps). Record the bracketing off-dumps and whether the firing sits
        # at a pointing edge (bracket would fall outside the pointing), plus the firing's period.
        firings = []  # list of (on_dumps: np.ndarray, off1: int, off2: int, edge: bool)
        firing_periods = []
        for rel, period in zip(pointings, pointing_periods):
            p_start, p_end = int(rel[0]), int(rel[-1])
            on_positions = [t for t in range(p_start, p_end + 1) if noise_on[t]]
            if not on_positions:
                continue
            for group in consecutive_subsets(on_positions):
                lo, hi = group[0], group[-1]
                off1, off2 = lo - 1, hi + 1
                edge = (off1 < p_start) or (off2 > p_end)
                firings.append((np.array(group), off1, off2, edge))
                firing_periods.append(period)

        # Keep, aligned with the firing axis: the on-dump indices (for projecting the excess mask
        # back onto the firing dumps) and the period label (for splitting outputs per period).
        self.firing_on_dumps = [on_dumps for on_dumps, _off1, _off2, _edge in firings]
        self.firing_periods = np.array(firing_periods)

        # Firing time vector (receiver-independent), in seconds since the start of the track data.
        # For two-dump firings the time falls between the dumps, weighted by the noise diode firing
        # ratio of each dump.
        noise_diode = NoiseDiode(
            dump_period=track_data.dump_period,
            observation_log=track_data.obs_script_log,
        )
        # Diode on-fraction per dump: the measured per-dump excess is duty * T_nd * gain, so the
        # calibrated noise diode temperature is (excess / gain) / duty.
        self.noise_diode_duty = noise_diode.duration / track_data.dump_period
        cycle_starts = noise_diode._get_noise_diode_cycle_start_times(
            timestamps=track_data.timestamps
        )
        ratios = noise_diode._get_noise_diode_ratios(
            timestamps=track_data.timestamps,
            noise_diode_cycle_starts=cycle_starts,
            dump_period=track_data.dump_period,
        )
        timestamps_arr = track_data.timestamps.squeeze
        timestamps_arr = (
            timestamps_arr - timestamps_arr.min()
        )  # epoch seconds -> seconds since start
        noise_on_timestamp = []
        for on_dumps, _off1, _off2, _edge in firings:
            if len(on_dumps) == 1:
                ts = float(timestamps_arr[on_dumps[0]])
            else:
                weights = ratios[on_dumps]
                if weights.sum() > 0:
                    ts = float(
                        np.sum(timestamps_arr[on_dumps] * weights) / np.sum(weights)
                    )
                else:  # ratios undefined at this boundary, fall back to the simple midpoint
                    ts = float(np.mean(timestamps_arr[on_dumps]))
            noise_on_timestamp.append(ts)
        self.noise_on_timestamp = np.array(noise_on_timestamp)

        visibility = track_data.visibility
        for i_receiver in range(len(track_data.receivers)):
            vis_recv = visibility.get(recv=i_receiver).squeeze
            flag_recv = initial_flags.get(recv=i_receiver).squeeze
            yield vis_recv, flag_recv, firings, noise_on

    def run_job(
        self, anything: tuple[np.ndarray, np.ndarray, list, np.ndarray]
    ) -> np.ndarray:
        """Compute the per-firing noise diode excess for one receiver."""
        vis_recv, flag_recv, firings, noise_on = anything

        # "Unflag" the firing dumps in the combined mask: a firing is judged only by its neighbours.
        combined = flag_recv.copy()
        combined[noise_on, :] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vis = np.ma.masked_array(vis_recv, mask=combined)

        n_freq = vis_recv.shape[1]
        excess = np.ma.zeros((len(firings), n_freq))
        for i_firing, (on_dumps, off1, off2, edge) in enumerate(firings):
            if edge:
                excess[i_firing, :] = np.ma.masked
                continue
            on_value = vis[on_dumps].sum(
                axis=0
            )  # firing dumps were unflagged -> never masked
            off_value = (
                vis[off1] + vis[off2]
            ) / 2.0  # masked where either off-neighbour is flagged
            excess[i_firing, :] = on_value - off_value * len(on_dumps)
        return excess

    def gather_and_set_result(
        self,
        result_list: list[np.ndarray],
        track_data: TimeOrderedData,
        calibrator_validated_periods: list,
        calibrator_dump_indices: dict,
        flag_report_writer: ReportWriter,
        output_path: str,
    ):
        """
        Stack the per-receiver excess, mask per-firing outliers, flag low-excess receivers, and set the results.
        :param result_list: `list` of per-receiver excess masked arrays of shape `(n_firings, n_freq)`
        :param track_data: `TimeOrderedData` containing the calibrator tracking part of the observation
        :param calibrator_validated_periods: validated periods of single-dish calibrator scans
        :param calibrator_dump_indices: indices of validated periods of single-dish calibrator scans
        :param flag_report_writer: report of the flag
        :param output_path: path to store the context
        """
        # (n_firings, n_freq, n_receivers)
        noise_diode_excess = np.ma.stack(result_list, axis=2)
        noise_on_timestamp = self.noise_on_timestamp

        # Per-firing outlier masking, done per period: for each receiver, mask the firings whose
        # frequency-median excess is far from that receiver's robust average over THAT period's
        # firings (modified z-score / MAD). The mask propagates into the output excess (the whole
        # firing spectrum is masked for that receiver). Doing it per period avoids mixing the
        # before/after excess levels into the robust statistics.
        n_firings, _, n_receivers = noise_diode_excess.shape
        firing_outlier = np.zeros((n_firings, n_receivers), dtype=bool)
        for period in calibrator_validated_periods:
            p_sel = self.firing_periods == period
            if not p_sel.any():
                continue
            for i_receiver in range(n_receivers):
                freqmedian = np.ma.median(
                    noise_diode_excess[p_sel, :, i_receiver], axis=1
                )
                firing_outlier[p_sel, i_receiver] = remove_outliers_zscore_mad(
                    freqmedian.data,
                    np.ma.getmaskarray(freqmedian),
                    self.zscoreflag_threshold,
                    self.max_masked_fraction,
                )
        full_mask = (
            np.ma.getmaskarray(noise_diode_excess) | firing_outlier[:, np.newaxis, :]
        )
        noise_diode_excess = np.ma.array(noise_diode_excess.data, mask=full_mask)

        # Receiver low-limit check: a receiver whose overall median excess is too low (a dead or very
        # weak noise diode) has all of its firings masked, in both the excess and the flag. The
        # median is taken after the per-firing outlier masking above so outliers don't bias it.
        bad_receivers = []
        for i_receiver, receiver in enumerate(track_data.receivers):
            median_excess = np.ma.median(noise_diode_excess[:, :, i_receiver])
            # a fully-masked median compares False against `<=`, so treat it as a low-limit failure
            lowlim_triggered = bool(
                np.ma.is_masked(median_excess)
                or median_excess <= self.noise_diode_excess_lowlim
            )
            if lowlim_triggered:
                full_mask[:, :, i_receiver] = True
                bad_receivers.append(str(receiver))
        noise_diode_excess = np.ma.array(noise_diode_excess.data, mask=full_mask)

        # The `noise_diode_bad_behavior` flag is the complete excess mask (edge firings, flagged
        # off-neighbours, per-firing outliers and the low-limit check) projected ONTO THE FIRING
        # ON-DUMPS ONLY. Off dumps (and any non-pointing dump) are never flagged.
        flag_array = np.zeros(
            track_data.visibility.shape, dtype=bool
        )  # (n_time, n_freq, n_receivers)
        for i_firing, on_dumps in enumerate(self.firing_on_dumps):
            flag_array[on_dumps, :, :] |= full_mask[i_firing][np.newaxis, :, :]

        track_data.flags.add_flag(
            flag=FlagList.from_array(
                array=flag_array, element_factory=self.data_element_factory
            ),
            name="noise_diode_bad_behavior",
        )

        # Split the per-firing excess into per-period dicts keyed by period, each with a robust
        # (masked-median) time-averaged excess spectrum.
        excess_by_period = {}
        average_by_period = {}
        timestamp_by_period = {}
        for period in calibrator_validated_periods:
            p_sel = self.firing_periods == period
            excess_by_period[period] = noise_diode_excess[
                p_sel
            ]  # (n_firings_p, n_freq, n_recv)
            average_by_period[period] = np.ma.median(
                noise_diode_excess[p_sel], axis=0
            )  # (n_freq, n_recv)
            timestamp_by_period[period] = noise_on_timestamp[p_sel]  # (n_firings_p,)

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        receivers_list, flag_percent = flag_percent_recv(track_data)
        firings_per_period = {
            p: int((self.firing_periods == p).sum())
            for p in calibrator_validated_periods
        }
        lines = (
            [
                "...........................",
                "Running NoiseDiodeSignalPlugin with "
                + f"MuSEEK version: {branch} ({commit})",
                "Finished at " + current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                f"Number of noise diode firings per period: {firings_per_period}",
                f"Receivers fully flagged for too-low noise diode excess: {bad_receivers if bad_receivers else 'none'}",
            ]
            + ["The flag fraction for each receiver: "]
            + [f"{x}  {y}" for x, y in zip(receivers_list, flag_percent)]
        )
        flag_report_writer.write_to_report(lines)

        self.set_result(
            result=Result(
                location=ResultEnum.TRACK_DATA, result=track_data, allow_overwrite=True
            )
        )
        self.set_result(
            result=Result(
                location=ResultEnum.NOISE_DIODE_EXCESS,
                result=excess_by_period,
                allow_overwrite=True,
            )
        )
        self.set_result(
            result=Result(
                location=ResultEnum.NOISE_DIODE_EXCESS_AVERAGE,
                result=average_by_period,
                allow_overwrite=True,
            )
        )
        self.set_result(
            result=Result(
                location=ResultEnum.NOISE_DIODE_DUTY_CYCLE,
                result=self.noise_diode_duty,
                allow_overwrite=True,
            )
        )
        self.set_result(
            result=Result(
                location=ResultEnum.NOISE_ON_TIMESTAMP,
                result=timestamp_by_period,
                allow_overwrite=True,
            )
        )

        if self.do_store_context:
            context_file_name = "noise_diode_signal_plugin.pickle"
            self.store_context_to_disc(
                context_file_name=context_file_name, context_directory=output_path
            )
