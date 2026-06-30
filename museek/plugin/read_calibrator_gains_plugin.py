import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.flag_element import FlagElement
from museek.time_ordered_data import TimeOrderedData

PERIOD_KEYS = ('before_scan', 'after_scan')


class ReadCalibratorGainsPlugin(AbstractPlugin):
    """
    Loads pickled model_components file(s) from PointSourceCalibrationPlugin, selects the requested
    calibrator period(s), averages their `gain_on_off`, matches receivers by name, and sets the gain
    solution on the target `scan_data`.

    The gain is read from the pickle and may come from a different observation block; it is applied to
    whatever `scan_data` is being processed, matching receivers by name and using the scan data's own
    frequency grid and time axis. (The pickle and the target must share the same band/channelisation.)
    """

    def __init__(self,
                 model_components_files: list,
                 periods: list | str | None = None,
                 verbose: int = 0):
        """
        :param model_components_files: list of 1 or 2 full paths to model_components pickle files
            (more than two is an error). Each file may contain one or both calibrator periods
            ('before_scan', 'after_scan').
        :param periods: which period(s) to use, as a flat list of period names (a bare string is
            accepted and treated as a one-element list). Interpreted by the number of files:
              - 1 file:  every name comes from that single file (1 or 2 names), averaged.
                         e.g. ['before_scan'] (before only), ['before_scan', 'after_scan'] (both).
              - 2 files: exactly 2 names, one per file by position — periods[0] from the first file,
                         periods[1] from the second. e.g. ['before_scan', 'after_scan'].
            Default `None`:
              - 1 file:  use all periods present (average before + after, or the single one present);
              - 2 files: error (you must specify the period for each file).
            The selected gains are averaged (masked mean) and a missing period raises an error.
        :param verbose: if non-zero, print gain statistics and save a plot after loading.
        """
        super().__init__()
        self.model_components_files = model_components_files
        self.periods = [periods] if isinstance(periods, str) else periods
        self.verbose = verbose

    def set_requirements(self):
        self.requirements = [
            Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
        ]

    def run(self, scan_data: TimeOrderedData, output_path: str):
        files = self.model_components_files
        if len(files) > 2:
            raise ValueError(f"At most two model_components files are supported, got {len(files)}.")

        # Load each file once: (path, model_components, receiver labels, periods present)
        loaded = []
        for path in files:
            with open(path, 'rb') as f:
                mc = pickle.load(f)
            if 'receivers' not in mc:
                raise ValueError(f"Pickle file {path} does not contain receiver labels. "
                                 f"Re-run PointSourceCalibrationPlugin to regenerate.")
            available = [p for p in PERIOD_KEYS if p in mc]
            loaded.append((path, mc, mc['receivers'], available))

        # Resolve the (file_index, period) selections from `periods` and the number of files.
        if self.periods is None:
            if len(files) == 2:
                raise ValueError("Two model_components files were given; specify `periods` as one "
                                 "period name per file, e.g. ['before_scan', 'after_scan'].")
            chosen = [(0, p) for p in loaded[0][3]]  # one file -> all periods present
        elif len(files) == 2:
            if len(self.periods) != 2:
                raise ValueError(f"Two files were given, so `periods` must have exactly two period "
                                 f"names (one per file); got {self.periods}.")
            chosen = [(0, self.periods[0]), (1, self.periods[1])]
        else:
            chosen = [(0, p) for p in self.periods]  # one file -> all listed periods come from it

        # Resolve each selection to its gain array, validating the period exists in that file.
        selections = []  # list of (gain_on_off (n_freq, n_recv_file), file_receivers, label)
        for i_file, period in chosen:
            path, mc, file_receivers, available = loaded[i_file]
            if period not in mc:
                raise ValueError(f"Period '{period}' not found in {path}. "
                                 f"Available periods: {available}.")
            selections.append((mc[period]['gain_on_off'], file_receivers,
                               f"{os.path.basename(path)}:{period}"))

        if not selections:
            raise ValueError("No calibrator periods selected from the provided pickle files.")

        n_freq = len(scan_data.frequencies.squeeze)
        for gain_arr, _file_receivers, label in selections:
            if gain_arr.shape[0] != n_freq:
                raise ValueError(
                    f"Gain '{label}' has {gain_arr.shape[0]} frequency channels but the target scan "
                    f"data has {n_freq}; the saved gain and the target must share the same "
                    f"band/channelisation.")
        current_receivers = [str(r) for r in scan_data.receivers]

        # Match receivers by name and average over the selected (file, period) gains.
        gain_matched = np.zeros((n_freq, len(current_receivers)))
        mask_matched = np.ones((n_freq, len(current_receivers)), dtype=bool)  # default: fully masked

        for i, recv in enumerate(current_receivers):
            recv_gains = []
            for gain_arr, file_receivers, _label in selections:
                if recv in file_receivers:
                    recv_gains.append(gain_arr[:, file_receivers.index(recv)])  # (n_freq,) masked

            if len(recv_gains) == 0:
                print(f"Warning: receiver {recv} not found in any selected gain — masking.", flush=True)
                continue  # gain=0, mask=True

            if len(recv_gains) < len(selections):
                print(f"Warning: receiver {recv} found in only {len(recv_gains)}/{len(selections)} "
                      f"selections — using partial average.", flush=True)

            recv_avg = np.ma.mean(np.ma.stack(recv_gains, axis=0), axis=0)  # (n_freq,)
            gain_matched[:, i] = recv_avg.data
            mask_matched[:, i] = recv_avg.mask if np.ma.is_masked(recv_avg) else False

        # The gain is constant in time, so pass it as a compact (n_freq, n_receivers) Result rather
        # than bolting it onto the TOD. The bad-channel mask is a per-channel data flag, so it is
        # still added to scan_data.flags; flags must match the full (n_time, n_freq, n_receivers)
        # stack, so tile the mask over the current scan-state time axis (not scan_data.shape, which
        # is the full-observation shape unchanged by the scan/track split).
        n_time = scan_data.timestamps.shape[0]
        gain_mask = np.tile(mask_matched[np.newaxis, :, :], (n_time, 1, 1))

        scan_data.flags.add_flag(flag=FlagElement(array=gain_mask), name='gain_solution_mask')
        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.CALIBRATOR_GAIN, result=gain_matched))

        if self.verbose:
            print(f"Loaded selections: {[label for _, _, label in selections]}", flush=True)
            print(f"Calibrator gain shape: {gain_matched.shape}", flush=True)
            unmasked = np.where(~mask_matched[:, 0])[0]
            freq_mid = unmasked[len(unmasked) // 2] if len(unmasked) > 0 else n_freq // 2
            for i, recv in enumerate(current_receivers):
                print(f"  {recv}: gain[mid_freq] = {gain_matched[freq_mid, i]:.4f}", flush=True)

            # Plot each selection's gain and the average for the first receiver
            freq_MHz = scan_data.frequencies.squeeze / 1e6
            recv0 = current_receivers[0]
            fig, ax = plt.subplots(figsize=(10, 4))
            for gain_arr, file_receivers, label in selections:
                if recv0 in file_receivers:
                    g = gain_arr[:, file_receivers.index(recv0)]
                    ax.plot(freq_MHz, np.ma.filled(g, np.nan), alpha=0.7, label=label)
            ax.plot(freq_MHz, gain_matched[:, 0], color='black', linewidth=1.5, label='average')
            ax.set_xlabel('Frequency [MHz]')
            ax.set_ylabel('Gain')
            ax.set_title(f'Calibrator gains — {recv0}')
            ax.legend()
            plot_path = os.path.join(output_path, f'read_calibrator_gains_{recv0}.png')
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Gain plot saved to {plot_path}", flush=True)
