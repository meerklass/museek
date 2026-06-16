from typing import Generator
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.noise_diode import NoiseDiode
from museek.flag_factory import FlagFactory
from museek.flag_list import FlagList
from museek.data_element import DataElement
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import flag_percent_recv, git_version_info
from museek.util.tools import remove_outliers_zscore_mad, polynomial_flag_outlier
import numpy as np
import warnings
import datetime


class NoiseDiodeExcessPlugin(AbstractParallelJoblibPlugin):
    # For: noise diode excess / gain (from previous plugins)
    """
    Plugin to calculate a good measure of the noise diode excess signal from the
    calibrator track data.

    This is a reworked version of `NoiseDiodePlugin`. Three changes relative to
    the original:

    1. It operates on `TRACK_DATA` rather than `SCAN_DATA`, so it fits the
       point-source calibration pipeline (which discards the scan data).
    2. Flagging is leaked to the noise diode dumps. The original combined every
       flag (including ``noise_diode_on``) and then explicitly *unflagged* every
       ND-firing dump, so RFI or quality issues sitting in a firing dump were
       never accounted for. Here we combine all flags *except* the ones in
       `nd_excluded_flag_names` and leave that combined mask in place, so a
       contaminated ND dump (or a contaminated adjacent off dump used for the
       baseline) is excluded from the excess. `noise_diode_on` is excluded
       because it masks every firing by definition; `aoflagger_tracking` is excluded
       because the amplitude-based aoflagger mistakes the ND burst itself for
       RFI. Structural flags (known RFI, antenna, elevation) and the low-value
       flag are kept. A per-firing "is this firing good?" check additionally
       drops firings whose on-dump is mostly flagged, and a physicality check
       masks non-positive excess values (dead diode or a contaminated
       off-baseline), masking the whole firing if most channels go non-positive.
    3. A robust time-averaged excess is produced alongside the per-firing excess:
       the masked median over all firings, per frequency and receiver. This is
       the "good measure" intended for downstream calibration use.

    The off-dump baseline is still the average of the dumps adjacent to each
    firing (same as the original), but firings at the very start/end of the
    track data, whose bracket would fall outside the array, are dropped rather
    than being given a wrapped or one-sided baseline (the edge-firing artefact).
    A per-pointing spline baseline is a possible future refinement.
    """

    def __init__(self,
                 flag_combination_threshold: int,
                 zscoreflag_threshold: float,
                 polyflag_deg: int,
                 polyflag_threshold: float,
                 polyfit_deg: int,
                 zscore_antenaflag_threshold: float,
                 noise_diode_excess_lowlim: float,
                 nd_dump_good_fraction: float = 0.5,
                 nd_excess_failure_fraction: float = 0.5,
                 nd_excluded_flag_names: tuple[str, ...] = ('noise_diode_on', 'aoflagger_tracking'),
                 do_store_context: bool = False,
                 **kwargs):
        """
        Initialise the plugin
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        :param zscoreflag_threshold: threshold for flagging noise diode excess using modified zscore method
        :param polyflag_deg: degree of the polynomials used for fitting and flagging noise diode excess
        :param polyflag_threshold: threshold for flagging noise diode excess using polynomials fit
        :param polyfit_deg: degree of the polynomials used for fitting flagged noise diode excess
        :param zscore_antenaflag_threshold: threshold for flagging the rms of noise diode excess of receivers using modified zscore method
        :param noise_diode_excess_lowlim: threshold for flagging the mean value of noise diode excess of receivers
        :param nd_dump_good_fraction: a firing is considered bad (and fully masked) if fewer than this
                                      fraction of its frequency channels survive the leaked flagging
        :param nd_excess_failure_fraction: if more than this fraction of a firing's channels have a
                                      non-positive excess, the whole firing is masked as a diode
                                      failure (a real failure hits all channels at once; per-channel
                                      RFI does not). Remaining non-positive channels are masked too.
        :param nd_excluded_flag_names: flag layers NOT leaked onto the ND dumps. `noise_diode_on` masks
                                       every firing by definition; `aoflagger_tracking` mistakes the ND burst
                                       for RFI. All other flag layers ARE applied to the ND dumps.
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        """
        super().__init__(**kwargs)
        self.flag_combination_threshold = flag_combination_threshold
        self.zscoreflag_threshold = zscoreflag_threshold
        self.polyflag_deg = polyflag_deg
        self.polyflag_threshold = polyflag_threshold
        self.polyfit_deg = polyfit_deg
        self.zscore_antenaflag_threshold = zscore_antenaflag_threshold
        self.noise_diode_excess_lowlim = noise_diode_excess_lowlim
        self.nd_dump_good_fraction = nd_dump_good_fraction
        self.nd_excess_failure_fraction = nd_excess_failure_fraction
        self.nd_excluded_flag_names = set(nd_excluded_flag_names)
        self.do_store_context = do_store_context
        self.report_file_name = 'flag_report.md'

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]

    def map(self,
            track_data: TimeOrderedData,
            flag_report_writer: ReportWriter,
            output_path: str) \
            -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Yield a `tuple` of the track visibility data for one receiver and the leaked flags for one receiver.
        :param track_data: time ordered data containing the calibrator tracking part of the observation
        :param flag_report_writer: report of the flag
        """

        track_data.load_visibility_flags_weights(polars='auto')

        # Leak the flagging to the noise diode dumps: combine every flag layer
        # EXCEPT the ones in `nd_excluded_flag_names`. Keeping `noise_diode_on`
        # out means the ND-firing dumps are no longer auto-masked just for being
        # firings; keeping `aoflagger_tracking` out means the amplitude-based aoflagger
        # does not mask the ND burst itself. The ND dumps DO keep any real RFI /
        # antenna / elevation / low-value flags.
        result_array = np.zeros(track_data.flags.shape)
        for i, name in enumerate(track_data.flags.flag_names):
            if name not in self.nd_excluded_flag_names:
                result_array += track_data.flags._flags[i].get_array()
        result_array[result_array < self.flag_combination_threshold] = 0
        initial_flags = track_data.flags._flag_element_factory.create(
            array=np.asarray(result_array, dtype=bool)
        )

        noise_diode = NoiseDiode(dump_period=track_data.dump_period, observation_log=track_data.obs_script_log)
        noise_diode_off_dumps = noise_diode.get_noise_diode_off_scan_dumps(timestamps=track_data.timestamps)
        noise_diode_cycle_start_times = noise_diode._get_noise_diode_cycle_start_times(timestamps=track_data.timestamps)
        noise_diode_ratios = noise_diode._get_noise_diode_ratios(timestamps=track_data.timestamps,
                                                                 noise_diode_cycle_starts=noise_diode_cycle_start_times,
                                                                 dump_period=noise_diode._dump_period)

        for i_receiver, receiver in enumerate(track_data.receivers):
            visibility = track_data.visibility.get(recv=i_receiver).squeeze
            initial_flag = initial_flags.get(recv=i_receiver).squeeze
            yield visibility, initial_flag, noise_diode_off_dumps, noise_diode_ratios, track_data.timestamps.array.squeeze()

    def run_job(self, anything: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) \
            -> list[np.ndarray, np.ndarray, np.ndarray]:
        """ Run the plugin and calculate the noise diode excess signal for one receiver. """

        visibility, initial_flag, noise_diode_off_dumps, noise_diode_ratios, timestamps = anything

        noise_on = np.ones(np.shape(visibility)[0], dtype='bool')
        for i in noise_diode_off_dumps:
            noise_on[i] = False

        # Find continuous noise-on regions
        continuous_noise_on_index = self.find_continuous_noise_on_regions(noise_on)

        # calculate the noise diode firing index. Because there will be noise diode firing in two continuous timestamps, we need to re-calculate the index of noise diode firing by use of noise_diode_ratios weighted average
        noise_on_index = []
        noise_on_timestamp = []
        timestamps_shifted = timestamps - timestamps.min()
        for i_index_list, index_list in enumerate(continuous_noise_on_index):
            if len(index_list) == 1:   # if noise diode firing is in one timestamp
                noise_on_index.append(index_list[0])
                noise_on_timestamp.append(timestamps_shifted[index_list[0]])
            elif len(index_list) > 1:  # if noise diode firing is in more than one timestamps
                index_array = np.array(index_list)
                noise_on_index.append(np.sum(index_array*noise_diode_ratios[index_array]) / np.sum(noise_diode_ratios[index_array]))
                timestamps_shifted_array = np.array([timestamps_shifted[i] for i in index_list])
                noise_on_timestamp.append(np.sum(timestamps_shifted_array*noise_diode_ratios[index_array]) / np.sum(noise_diode_ratios[index_array]))

        # Apply the leaked flags. Unlike the original NoiseDiodePlugin, the ND-firing
        # dumps are NOT unflagged here: a firing dump (or an adjacent off dump) that
        # carries a real RFI / quality flag stays masked and is excluded from the excess.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            visibility = np.ma.masked_array(visibility.copy(), mask=initial_flag.copy())

        # Off-baseline neighbour dumps for each firing: the dumps immediately
        # before and after the firing region. Guard against firings at the very
        # start/end of the track data, whose bracket would fall outside the array
        # (a negative index silently wraps to the opposite end of the observation;
        # an index == n_time raises IndexError). Such "edge firings" lack a proper
        # two-sided off-baseline and are dropped: a one-sided or wrapped baseline
        # collapses the excess (the edge-firing artefact, see
        # diode_excess_onoff_source.md §14). The off dumps are bracket-adjacent in
        # both the single-dump (idx-1, idx+1) and straddled (lo-1, hi+1) cases.
        n_time = visibility.shape[0]
        off_neighbours_per_firing = []
        edge_firing = np.zeros(len(continuous_noise_on_index), dtype=bool)
        for i_index_list, index_list in enumerate(continuous_noise_on_index):
            lo, hi = np.min(index_list), np.max(index_list)
            neighbours = [i for i in range(lo - 1, hi + 2)
                          if i not in index_list and 0 <= i < n_time]
            off_neighbours_per_firing.append(neighbours)
            if len(neighbours) < 2:  # missing a bracket dump => at the track-data edge
                edge_firing[i_index_list] = True

        # calculate the noise diode firing - diode off, noise diode off is estimated by averaging two points (one before, one after) adjacent to the noise diode firing point
        noise_diode_excess = np.ma.zeros((len(continuous_noise_on_index), visibility.shape[1]))
        for i_freq in range(visibility.shape[1]):
            for i_index_list, index_list in enumerate(continuous_noise_on_index):
                neighbours = off_neighbours_per_firing[i_index_list]
                if len(index_list) == 1:
                    noise_on_value = visibility[index_list[0],i_freq]
                    noise_off_value = np.ma.mean([visibility[i,i_freq] for i in neighbours])
                    noise_diode_excess[i_index_list,i_freq] = noise_on_value - noise_off_value

                elif len(index_list) > 1:
                    noise_on_value = np.ma.sum([visibility[i,i_freq] for i in index_list])
                    noise_off_value = np.ma.mean([visibility[i,i_freq] for i in neighbours])
                    noise_diode_excess[i_index_list,i_freq] = noise_on_value - noise_off_value*len(index_list)

        noise_diode_excess.mask[np.isnan(noise_diode_excess.data)] = True

        # Drop edge firings: their bracket fell outside the track data, so the
        # off-baseline is one-sided / wrapped and the excess is untrustworthy.
        if edge_firing.any():
            noise_diode_excess.mask[edge_firing, :] = True

        # Physicality check: V_on - V_off must be positive for a working noise diode.
        # A non-positive excess means either the diode failed to fire (excess ~ 0) or the
        # off-baseline is contaminated (e.g. unflagged RFI in an adjacent off dump, which
        # drives the excess strongly negative). If most of a firing's channels are
        # non-positive, treat it as a diode failure and mask the whole firing; otherwise
        # just mask the offending channels.
        nonpositive = (noise_diode_excess.data <= 0) & ~noise_diode_excess.mask
        for i_index_list in range(noise_diode_excess.shape[0]):
            if np.mean(nonpositive[i_index_list, :]) > self.nd_excess_failure_fraction:
                noise_diode_excess.mask[i_index_list, :] = True
        noise_diode_excess.mask[noise_diode_excess.data <= 0] = True

        # Check that the noise diode dumps are good: a firing whose on-dump is mostly
        # masked (by the leaked RFI / quality flags) or whose adjacent off dumps are
        # mostly masked yields too few usable channels to trust. Mask the whole firing
        # if fewer than `nd_dump_good_fraction` of channels survived.
        for i_index_list in range(noise_diode_excess.shape[0]):
            good_fraction = np.mean(~noise_diode_excess.mask[i_index_list, :])
            if good_fraction < self.nd_dump_good_fraction:
                noise_diode_excess.mask[i_index_list, :] = True

        return noise_diode_excess, noise_on_index, noise_on_timestamp

    def gather_and_set_result(self,
                              result_list: list[np.ndarray],
                              track_data: TimeOrderedData,
                              flag_report_writer: ReportWriter,
                              output_path: str):
        """
        Combine the per-receiver noise diode excess, flag badly behaving receivers, and set results.
        :param result_list: `list` of `(noise_diode_excess, noise_on_index, noise_on_timestamp)` per receiver
        :param track_data: `TimeOrderedData` containing the calibrator tracking part of the observation
        :param flag_report_writer: report of the flag
        :param output_path: path to store the context
        """

        noise_on_index = np.array(result_list[0][1])
        noise_on_timestamp = np.array(result_list[0][2])
        noise_diode_excess = np.ma.array([result_list[i][0] for i in range(len(result_list))]).transpose(1, 2, 0)

        # fit the noise diode excess and mask the receiver if the frequency median of its noise diode excess can not be fitted well by a 2-order polynomial
        rmsnorm_poly_fit_list = []
        for i_receiver, receiver in enumerate(track_data.receivers):
            noise_excess_timemedian = np.ma.median(noise_diode_excess[:,:,i_receiver], axis=1)
            noise_excess_timemedian.mask = remove_outliers_zscore_mad(noise_excess_timemedian.data, noise_excess_timemedian.mask, self.zscoreflag_threshold)
            noise_excess_timemedian.mask, p_fit = polynomial_flag_outlier(noise_on_timestamp, noise_excess_timemedian.data, noise_excess_timemedian.mask, self.polyflag_deg, self.polyflag_threshold)
            if noise_excess_timemedian.mask.all():
                rmsnorm_poly_fit_list.append(np.nan)
            else:
                p_poly = np.polyfit(noise_on_timestamp[~noise_excess_timemedian.mask], noise_excess_timemedian.data[~noise_excess_timemedian.mask], deg=self.polyfit_deg)
                rmsnorm_poly_fit_list.append(np.std(np.polyval(p_poly, noise_on_timestamp) / np.median(np.polyval(p_poly, noise_on_timestamp))))

        rmsnorm_poly_fit_list = np.array(rmsnorm_poly_fit_list)

        # A `nan` rms means the receiver's time-median excess was mostly masked, so its
        # poly-flatness could not be assessed: that is "no opinion", NOT "bad behaviour".
        # Feeding the nan-mask straight into `remove_outliers_zscore_mad` is wrong, because
        # its `mean(mask) > 0.6 -> mask everything` rule then flags EVERY receiver (including
        # ones with a perfectly healthy excess) as soon as most receivers are nan. Assess the
        # outliers only among receivers we could actually fit; leave nan receivers for the
        # lowlim check below (a receiver with genuinely no usable data has a masked median
        # excess and is caught there).
        finite = ~np.isnan(rmsnorm_poly_fit_list)
        receiver_mask_poly_fit = np.zeros(len(rmsnorm_poly_fit_list), dtype=bool)
        if finite.any():
            receiver_mask_poly_fit[finite] = remove_outliers_zscore_mad(
                rmsnorm_poly_fit_list[finite],
                np.zeros(int(finite.sum()), dtype=bool),
                self.zscore_antenaflag_threshold,
            )

        shape = track_data.visibility.shape
        new_flag = FlagList(flags=[FlagFactory().empty_flag(shape=shape)])
        new_flag_array = np.zeros((shape[0], len(track_data.receivers)))
        bad_receivers = []
        for i_receiver, receiver in enumerate(track_data.receivers):
            median_excess = np.ma.median(noise_diode_excess[:,:,i_receiver])
            poly_triggered = bool(receiver_mask_poly_fit[i_receiver] > 0)
            # masked median compares False against <=, so treat fully-masked as a lowlim fail
            lowlim_triggered = bool(np.ma.is_masked(median_excess) or median_excess <= self.noise_diode_excess_lowlim)
            if poly_triggered or lowlim_triggered:
                outlier_antenna_flag = np.ones((shape[0]))
                bad_receivers.append(str(receiver))
            else:
                outlier_antenna_flag = new_flag_array[:,i_receiver]

            flag_array = np.repeat(outlier_antenna_flag, shape[1]).reshape((shape[0], shape[1], 1))
            new_flag.insert_receiver_flag(flag=DataElement(array=flag_array), i_receiver=i_receiver, index=0)

        track_data.flags.add_flag(flag=new_flag, name='noise_diode_bad_behavior')

        # The "good measure" of the noise diode excess: a robust time-average over
        # all firings, per frequency and receiver. Median is used so that any
        # remaining bad firings that survived the per-firing check do not bias it.
        # Shape: (n_freq, n_receivers).
        noise_diode_excess_average = np.ma.median(noise_diode_excess, axis=0)

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        receivers_list, flag_percent = flag_percent_recv(track_data)

        lines = ['...........................',
                 'Running NoiseDiodeExcessPlugin with ' + f"MuSEEK version: {branch} ({commit})",
                 'Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                 f'Number of noise diode firings: {noise_diode_excess.shape[0]}',
                 f'Receivers flagged for bad noise diode behaviour: {bad_receivers if bad_receivers else "none"}'] \
                + ['The flag fraction for each receiver: '] + [f'{x}  {y}' for x, y in zip(receivers_list, flag_percent)]
        flag_report_writer.write_to_report(lines)

        self.set_result(result=Result(location=ResultEnum.TRACK_DATA, result=track_data, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.NOISE_DIODE_EXCESS, result=noise_diode_excess, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.NOISE_DIODE_EXCESS_AVERAGE, result=noise_diode_excess_average, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.NOISE_ON_INDEX, result=noise_on_index, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.NOISE_ON_TIMESTAMP, result=noise_on_timestamp, allow_overwrite=True))

        if self.do_store_context:
            context_file_name = 'noise_diode_excess_plugin.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=output_path)

    def find_continuous_noise_on_regions(self,
                                         noise_on: np.ndarray):
        """
        Function to find continuous "noise on" sub-regions (the region where there are two timestamps sharing one noise diode firing)

        :param noise_on: bool array where noise diode firing is True, noise diode off is False
        :return: a list where noise diode is firing
        """
        regions = []
        current_region = []

        for i, val in enumerate(noise_on):
            if val == 1:
                current_region.append(i)
            else:
                if current_region:
                    regions.append(current_region)
                    current_region = []
        # Append the last region if still present
        if current_region:
            regions.append(current_region)

        return regions
