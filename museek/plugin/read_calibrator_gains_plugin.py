import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.enums.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData


class ReadCalibratorGainsPlugin(AbstractPlugin):
    """
    Loads pickled model_components file(s) from PointSourceCalibrationPlugin,
    extracts gain_on_off, averages across periods if needed, matches receivers
    by name, and sets the gain solution on track_data.
    """

    def __init__(self,
                 model_components_files: list,
                 verbose: int = 0):
        """
        :param model_components_files: List of 1 or 2 full paths to model_components pickle files.
            Each file may contain one or both calibrator periods (before_scan, after_scan).
            If two files are given, each must contain a different period.
        :param verbose: If non-zero, print gain statistics after loading.
        """
        super().__init__()
        self.model_components_files = model_components_files
        self.verbose = verbose

    def set_requirements(self):
        self.requirements = [
            Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
        ]

    def run(self, track_data: TimeOrderedData, output_path: str):
        # Load all files and collect gain_on_off per period
        period_gains = {}    # {period: (n_freq, n_file_receivers) masked array}
        period_receivers = {}  # {period: receiver list from that file}

        for path in self.model_components_files:
            with open(path, 'rb') as f:
                mc = pickle.load(f)

            if 'receivers' not in mc:
                raise ValueError(f"Pickle file {path} does not contain receiver labels. "
                                 f"Re-run PointSourceCalibrationPlugin to regenerate.")
            file_receivers = mc['receivers']

            for period in ('before_scan', 'after_scan'):
                if period not in mc:
                    continue
                if period in period_gains:
                    raise ValueError(f"Period '{period}' appears in more than one pickle file.")
                period_gains[period] = mc[period]['gain_on_off']  # (n_freq, n_file_receivers)
                period_receivers[period] = file_receivers

        if not period_gains:
            raise ValueError("No calibrator periods found in the provided pickle files.")

        n_freq = len(track_data.frequencies.squeeze)
        current_receivers = [str(r) for r in track_data.receivers]

        # Match receivers by name, per receiver across periods
        gain_matched = np.zeros((n_freq, len(current_receivers)))
        mask_matched = np.ones((n_freq, len(current_receivers)), dtype=bool)  # default: fully masked

        for i, recv in enumerate(current_receivers):
            # Collect this receiver's gain from each period where it exists
            recv_gains = []
            for period, gain_period in period_gains.items():
                receivers_in_period = period_receivers[period]
                if recv in receivers_in_period:
                    j = receivers_in_period.index(recv)
                    recv_gains.append(gain_period[:, j])  # (n_freq,) masked array

            if len(recv_gains) == 0:
                print(f"Warning: receiver {recv} not found in any gain file — masking.", flush=True)
                continue  # gain=0, mask=True

            if len(recv_gains) < len(period_gains):
                print(f"Warning: receiver {recv} found in only {len(recv_gains)}/{len(period_gains)} "
                      f"periods — using partial average.", flush=True)

            recv_avg = np.ma.mean(np.ma.stack(recv_gains, axis=0), axis=0)  # (n_freq,)
            gain_matched[:, i] = recv_avg.data
            mask_matched[:, i] = recv_avg.mask if np.ma.is_masked(recv_avg) else False

        # Tile to (n_time, n_freq, n_receivers)
        n_time = track_data.shape[0]
        gain_solution = np.tile(gain_matched[np.newaxis, :, :], (n_time, 1, 1))
        gain_mask = np.tile(mask_matched[np.newaxis, :, :], (n_time, 1, 1))

        track_data.set_gain_solution(
            gain_solution_array=gain_solution,
            gain_solution_mask_array=gain_mask
        )

        if self.verbose:
            print(f"Loaded periods: {list(period_gains.keys())}", flush=True)
            print(f"Gain solution shape: {gain_solution.shape}", flush=True)
            unmasked = np.where(~mask_matched[:, 0])[0]
            freq_mid = unmasked[len(unmasked) // 2] if len(unmasked) > 0 else n_freq // 2
            for i, recv in enumerate(current_receivers):
                print(f"  {recv}: gain[mid_freq] = {gain_matched[freq_mid, i]:.4f}", flush=True)

            # Plot per-period gains and average for the first receiver
            freq_MHz = track_data.frequencies.squeeze / 1e6
            recv0 = current_receivers[0]
            i0 = 0
            fig, ax = plt.subplots(figsize=(10, 4))
            for period, gain_period in period_gains.items():
                receivers_in_period = period_receivers[period]
                if recv0 in receivers_in_period:
                    j = receivers_in_period.index(recv0)
                    g = gain_period[:, j]
                    ax.plot(freq_MHz, np.ma.filled(g, np.nan), alpha=0.7, label=period)
            ax.plot(freq_MHz, gain_matched[:, i0], color='black', linewidth=1.5, label='average')
            ax.set_xlabel('Frequency [MHz]')
            ax.set_ylabel('Gain')
            ax.set_title(f'Calibrator gains — {recv0}')
            ax.legend()
            plot_path = os.path.join(output_path, f'read_calibrator_gains_{recv0}.png')
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Gain plot saved to {plot_path}", flush=True)
