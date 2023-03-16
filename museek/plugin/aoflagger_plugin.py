from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.data_element import DataElement
from museek.enum.result_enum import ResultEnum
from museek.flag_element import FlagElement
from museek.rfi_mitigation.aoflagger import get_rfi_mask
from museek.time_ordered_data import TimeOrderedData
from museek.visualiser import waterfall
from matplotlib import pyplot as plt


class AoflaggerPlugin(AbstractPlugin):
    """ Plugin to calculate RFI flags using the aoflagger algorithm. """

    def __init__(self, flag_combination_threshold: int):
        """
        """
        self.flag_combination_threshold = flag_combination_threshold
        super().__init__()

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data')]

    def run(self, scan_data: TimeOrderedData):
        """ """
        scan_data.load_visibility_flags_weights()
        initial_flags = scan_data.flags.combine(threshold=self.flag_combination_threshold)

        chi_1 = 50  # First threshold value
        sm_kernel_m = 40  # Smoothing, kernel window size in axis=1
        sm_kernel_n = 20  # Smoothing, kernel window size in axis=0
        sm_sigma_m = 15  # Smoothing, kernel sigma in axis=1
        sm_sigma_n = 7.5  # Smoothing, kernel sigma in axis=0

        struct_size_0 = 1  # size of struct for dilation on freq direction [pixels]
        struct_size_1 = 6  # size of struct for dilation on time direction [pixels]
        di_kwargs = dict(struct_size_0=struct_size_0, struct_size_1=struct_size_1)
        sm_kwargs = dict(M=sm_kernel_m, N=sm_kernel_n, sigma_m=sm_sigma_m, sigma_n=sm_sigma_n)

        eta_i = [0.5, 0.55, 0.62, 0.75, 1]

        # to speed up for testing
        # times = range(500, 2500)
        # freqs = range(800,2000)
        times = range(scan_data.visibility.shape[0])
        freqs = range(scan_data.visibility.shape[1])

        waterfall(scan_data.visibility, scan_data.flags, cmap='gist_ncar', norm='log')
        plt.show()

        for i_receiver, receiver in enumerate(scan_data.receivers):
            visibility = scan_data.visibility.get(recv=i_receiver, time=times, freq=freqs)
            initial_flag = initial_flags.get(recv=i_receiver, time=times, freq=freqs)
            rfi_flag = get_rfi_mask(tod=visibility,
                                    mask=initial_flag,
                                    first_threshold=chi_1,
                                    threshold_scales=eta_i,
                                    do_plot=False,
                                    smoothing_kwargs=sm_kwargs,
                                    dilation_kwargs=di_kwargs)
            # scan_data.flags.add_flag(rfi_flag)

            # new_flags = FlagElement(flags=[initial_flag])
            # new_flags.add_flag(rfi_flag)

            new_flags = FlagElement(flags=[rfi_flag])

            waterfall(scan_data.visibility.get(recv=i_receiver, time=times, freq=freqs), new_flags, cmap='gist_ncar', norm='log')
            plt.show()
