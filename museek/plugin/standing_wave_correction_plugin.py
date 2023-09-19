import os
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from definitions import MEGA
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from museek.data_element import DataElement
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from scipy.optimize import curve_fit
from scipy.integrate import simps
from sklearn.utils import resample
import csv


class StandingWaveCorrectionPlugin(AbstractPlugin):
    """ Experimental plugin to apply the standing wave correction to the data"""

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [
            Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
            Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
            Requirement(location=ResultEnum.STANDING_WAVE_CHANNELS, variable='target_channels'),
            Requirement(location=ResultEnum.STANDING_WAVE_EPSILON_FUNCTION_DICT, variable='epsilon_function_dict')
        ]

    def run(self,
            scan_data: TimeOrderedData,
            track_data: TimeOrderedData,
            output_path: str,
            target_channels: range | list[int],
            epsilon_function_dict: dict[dict[Callable]]):
        """ Run the plugin, i.e. apply the standing wave correction. """
        before_or_after = 'before_scan'   # can change this to after_scan as well.

        for i_receiver, receiver in enumerate(scan_data.receivers):
            print(f'Working on {receiver}...')
            if not os.path.isdir(receiver_path := os.path.join(output_path, receiver.name)):
                os.makedirs(receiver_path)
            antenna_index = receiver.antenna_index(receivers=scan_data.receivers)
            frequencies = scan_data.frequencies.get(freq=target_channels)
            epsilon = epsilon_function_dict[receiver.name][before_or_after](frequencies)
            self.plot_individual_swings(scan_data=scan_data,
                                        antenna_index=antenna_index,
                                        i_receiver=i_receiver,
                                        before_or_after=before_or_after,
                                        target_channels=target_channels,
                                        epsilon=epsilon,
                                        receiver_path=receiver_path)

            self.plot_azimuth_bins(scan_data=scan_data,
                                   i_receiver=i_receiver,
                                   antenna_index=antenna_index,
                                   target_channels=target_channels,
                                   epsilon=epsilon,
                                   frequencies=frequencies,
                                   receiver_path=receiver_path)

    @staticmethod
    def swing_turnaround_dumps(azimuth: DataElement) -> list[int]:
        """ DOC """
        sign = np.sign(np.diff(azimuth.squeeze))  #tells you where the telescope switches from inc azimuth to dec azimuth.
        sign_change = ((np.roll(sign, 1) - sign) != 0).astype(bool) # indices where the telescope turns around. 1 specefies how much the array is shifted by
        return np.where(sign_change)[0]

    @staticmethod
    def azimuth_digitizer(azimuth: DataElement) -> tuple[np.ndarray, np.ndarray]:
        """ DOC """
        bins = np.linspace(azimuth.min(axis=0).squeeze, azimuth.max(axis=0).squeeze, 50)
        return np.digitize(azimuth.squeeze, bins=bins), bins


    def plot_individual_swings(self,
                               scan_data,
                               antenna_index,
                               i_receiver,
                               before_or_after,
                               target_channels,
                               epsilon,
                               receiver_path):
        swing_turnaround_dumps = self.swing_turnaround_dumps(
            azimuth=scan_data.azimuth.get(recv=antenna_index)
        )
        fig = plt.figure(figsize=(8, 12))

        ax = fig.subplots(2, 1)
        ax[1].plot(scan_data.timestamp_dates.squeeze, scan_data.temperature.squeeze)

        for i, dump in enumerate(swing_turnaround_dumps):
            ax[1].text(scan_data.timestamp_dates.get(time=dump).squeeze,
                       scan_data.temperature.get(time=dump).squeeze,
                       f'{i}',
                       fontsize='x-small')
        if before_or_after == 'after_scan':
            correction_dump = scan_data.timestamps.shape[0] - 1
        else:
            correction_dump = 0
        ax[1].text(scan_data.timestamp_dates.get(time=correction_dump).squeeze,
                   scan_data.temperature.get(time=correction_dump).squeeze,
                   'model fit',
                   fontsize='small')

        ax[1].set_xlabel('time')
        ax[1].set_ylabel('temperature')

        for i in range(len(swing_turnaround_dumps) - 1):
            times = range(swing_turnaround_dumps[i], swing_turnaround_dumps[i + 1])
            flags = scan_data.flags.get(time=times,
                                        freq=target_channels,
                                        recv=i_receiver)
            mean_bandpass = scan_data.visibility.get(time=times,
                                                     freq=target_channels,
                                                     recv=i_receiver).mean(axis=0, flags=flags)
            # stanndard_deviation_bandpass = scan_data.visibility.get(time=times,
            #                                                         freq=target_channels,
            #                                                         recv=i_receiver).standard_deviation(axis=0,flags=flags)
            frequencies = scan_data.frequencies.get(freq=target_channels)

            corrected = mean_bandpass.squeeze / (1 + epsilon)
            # corrected_stddev = stanndard_deviation_bandpass.squeeze / (1+epsilon)

            ax[0].plot(frequencies.squeeze / MEGA,
                       corrected,
                       label=f'swing {i}')
            # ax[0].errorbar(frequencies.squeeze / MEGA,corrected,stanndard_deviation_bandpass.squeeze)
            if i % 5 == 0:
                ax[0].text(frequencies.squeeze[0] / MEGA,
                           corrected[0],
                           f'{i}',
                           fontsize='x-small')

        ax[0].set_xlabel('frequency [MHz]')
        ax[0].set_ylabel('intensity')
        plot_name = 'standing_wave_correction_scanning_swings.png'
        plt.savefig(os.path.join(receiver_path, plot_name))
        plt.close()
        
    def fit_linear(self, x, m, b):
        """ Linear fit function. """
        return m * x + b    

    # def plot_azimuth_bins(self,
    #                       scan_data,
    #                       i_receiver,
    #                       antenna_index,
    #                       target_channels,
    #                       epsilon,
    #                       frequencies,
    #                       receiver_path):

    #     azimuth_digitized, azimuth_bins = self.azimuth_digitizer(azimuth=scan_data.azimuth.get(recv=antenna_index))
    #     corrected_azimuth_binned_bandpasses = []
    #     for index in range(len(azimuth_bins)):
    #         time_dumps = np.where(azimuth_digitized == index)[0]
    #         visibility = scan_data.visibility.get(time=time_dumps,
    #                                               freq=target_channels,
    #                                               recv=i_receiver)
    #         flag = scan_data.flags.get(time=time_dumps,
    #                                    freq=target_channels,
    #                                    recv=i_receiver)
    #         corrected_azimuth_binned_bandpasses.append(visibility.mean(axis=0, flags=flag).squeeze / (1 + epsilon))

    #     plt.figure(figsize=(8, 6))
    #     for i, bandpass in enumerate(corrected_azimuth_binned_bandpasses):
    #         label = ''
    #         if i % 7 == 0:
    #             label = f'azimuth {azimuth_bins[i]:.1f}'
    #         plt.plot(frequencies.squeeze / MEGA, bandpass, label=label)

    #     plt.legend()
    #     plt.xlabel('frequency [MHz]')
    #     plt.ylabel('intensity')
    #     plot_name = 'standing_wave_correction_scanning_azimuth_bins.png'
    #     plt.savefig(os.path.join(receiver_path, plot_name))
    #     plt.close()
        
    # def read_reduced_chi_squared_values(self,file_path):
    #     values = []
    #     with open(file_path, "r") as file:
    #         for line in file:
    #             values.append(float(line.strip()))
    #     return values
    
    def read_reduced_chi_squared_values(self, file_path):
        values = []
        with open(file_path, "r") as file:
            for line in file:
                row_values = line.strip().split()
                values.extend([float(value) for value in row_values])
        return values

    def store_reduced_chi_squared_values(self, file_path, values):
        with open(file_path, "a") as file:
            row_values = " ".join(map(str, values))
            file.write(f"{row_values}\n")
            
    def append_to_csv(self, reduced_chi_squared, frequencies, upper_bounds, corrected_bandpass, fit_curve):
        with open("append_data_1631732038.csv", "w", newline='') as file:
            writer = csv.writer(file)
            
            # Write reduced_chi_squared as the first line
            writer.writerow([reduced_chi_squared])
            
            for i in range(len((corrected_bandpass.data))):
                row_data = [
                    frequencies.squeeze[i] / MEGA,
                    upper_bounds[i],
                    corrected_bandpass[i],
                    fit_curve[i]
                ]
                writer.writerow(row_data)
                
    def append_chi2_to_csv(self, reduced_chi_squared):
        with open("append_chi2_1631732038.csv", "a", newline='') as file:
            writer = csv.writer(file)
            
            # Write reduced_chi_squared as the first line
            writer.writerow([reduced_chi_squared])
            

    def read_csv_data(self):
        frequencies = []
        upper_bounds = []
        corrected_bandpass = []
        fit_curve = []
        reduced_chi_squared = None
        
        try:
            with open("append_data_1631732038.csv", "r", newline='') as file:
                reader = csv.reader(file)
                # Read the first line which is the reduced_chi_squared value
                reduced_chi_squared = float(next(reader)[0])
                for row in reader:
                    frequencies.append(float(row[0]))
                    upper_bounds.append(float(row[1]))
                    corrected_bandpass.append(float(row[2]))
                    fit_curve.append(float(row[3]))
        except FileNotFoundError:
            print("CSV file not found. Continuing without loading data.")
        
        return reduced_chi_squared, frequencies, upper_bounds, corrected_bandpass, fit_curve
    
    def read_chi2_csv_data(self):

        reduced_chi_squared = []
        
        try:
            with open("append_chi2_1631732038.csv", "r", newline='') as file:
                reader = csv.reader(file)
                # Read the first line which is the reduced_chi_squared value
                for row in reader:
                    reduced_chi_squared.append(float(row[0]))
                
        except FileNotFoundError:
            print("CSV file not found. Continuing without loading data.")
        
        return reduced_chi_squared

    def plot_selected_reduced_chi_squared(self, scan_data, receiver_indices, frequencies, MEGA):
        stored_values = self.read_reduced_chi_squared_values("reduced_chi_squared_values.txt")

        selected_values = [stored_values[i] for i in receiver_indices]

        plt.figure(figsize=(8, 6))
        for value in selected_values:
            plt.plot(frequencies.squeeze / MEGA, value, label=f"Receiver {receiver_indices[selected_values.index(value)]}")

        plt.grid()
        plt.xlabel("Frequencies [MHz]")
        plt.ylabel("Reduced Chi Squared Values")
        plt.title("Reduced Chi-Squared vs Frequencies")
        plt.legend()
        plt.show()
        plt.savefig("test_reading_chi2.png")

    def plot_azimuth_bins(self,
                          scan_data,
                          i_receiver,
                          antenna_index,
                          target_channels,
                          epsilon,
                          frequencies,
                          receiver_path):

        azimuth_digitized, azimuth_bins = self.azimuth_digitizer(azimuth=scan_data.azimuth.get(recv=antenna_index))
        corrected_azimuth_binned_bandpasses = []
        azimuth_fits = []  # Store linear fit parameters (m, b) for each azimuth curve
        azimuth_chi_squared = []  # Store chi-squared values for each azimuth curve
        standard_deviation_bandpass_list = []
        n_bootstrap = 1000
        integrated_areas = []
        upper_bounds = []
        lower_bounds = []
        # for a single azimuth combined bin
        visibility = scan_data.visibility.get(freq=target_channels, 
                                                recv=i_receiver)
        flag = scan_data.flags.get( freq=target_channels,
                                    recv=i_receiver)
        # corrected_visibility = visibility / (1 + epsilon)
        corrected_visibility = visibility.squeeze / (1+epsilon)
        corrected_bandpass = visibility.mean(axis=0, flags=flag).squeeze / (1 + epsilon)
        # corrected_azimuth_binned_bandpasses.append(corrected_bandpass)



        standard_deviation_bandpass = visibility.standard_deviation(axis=0,flags=flag).squeeze / (1+epsilon)
        antenna_name_list = scan_data._antenna_name_list
        updated_antenna_name_list = [antenna + polarization for antenna in antenna_name_list for polarization in ['h', 'v']]

        # Fit a linear curve to the azimuth curve using scipy.optimize.curve_fit
        fit_params, _ = curve_fit(self.fit_linear, frequencies.squeeze, corrected_bandpass)



        # Calculate chi-squared value for the linear fit
        expected = self.fit_linear(frequencies.squeeze, *fit_params)
        residuals = corrected_visibility - expected
        reduced_chi_squared_time = np.sum(residuals**2, axis=0)/((standard_deviation_bandpass)**2 * (len(frequencies.squeeze)-2) * 2929 )
        reduced_chi_squared = np.mean(reduced_chi_squared_time)
        reduced_array = reduced_chi_squared_time.data

        absolute_residuals = np.abs(residuals)  # Take the absolute value
        area = simps(absolute_residuals, frequencies.squeeze)


        upper_bound = standard_deviation_bandpass.data
        

        fit_curve = self.fit_linear(frequencies.squeeze, *fit_params)
        # self.append_to_csv(reduced_chi_squared, frequencies, upper_bound, corrected_bandpass, fit_curve)
        
        #do it only when appending the chi2 values
        # self.append_chi2_to_csv(reduced_chi_squared)
        
        # reduced_chi2_values_all = self.read_chi2_csv_data()
        
        # plt.figure(figsize=(8, 6))
        
        # plt.scatter(np.array(updated_antenna_name_list),np.array(reduced_chi2_values_all) , s = 5, color = 'red')

        # plt.grid()
        # plt.xlabel("Receiver names")
        # plt.ylabel("Reduced Chi Squared Values")
        # plt.title("Reduced Chi-Squared vs Receivers")
        # plt.show()
        # plt.savefig("test_reading_chi2_for_receivers.png")
        
        
        reduced_chi_squared_after, frequencies_after, upper_bounds_after, corrected_bandpass_after, fit_curve_after = self.read_csv_data()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original Points - Set 1 and Set 2
        ax1 = axes[0]

        ax1.scatter(frequencies.squeeze / MEGA, corrected_bandpass, label = "bandpass")
        ax1.plot(frequencies.squeeze / MEGA, fit_curve, color='black', linewidth=0.8, label = "linear fit")
        ax1.errorbar(
                                x=frequencies.squeeze / MEGA,
                                y=corrected_bandpass,
                                yerr=[upper_bound],
                                fmt='none',
                                label = "errorbar",
                                color='green',
                                alpha = 0.4
                            )
        ax1.set_xlabel('Frequency [MHz]')
        ax1.set_ylabel('Intensity')

        ax1.legend(title='Markers', loc='upper right')
        ax1.grid()

        # Updated Points - Set 1 and Set 2
        ax2 = axes[1]
        ax2.scatter(frequencies.squeeze / MEGA, reduced_chi_squared_time.data, label = "reduced chi squared", s = 2)
        ax2.set_xlabel('Frequency [MHz]')
        ax2.set_ylabel('Reduced chi squared')
        ax2.legend(title='Markers', loc='lower right')

        ax2.grid()


        # ax2.set_ylim(-10,0)


        plt.suptitle('Singular azimuth binned bandpass plot')
        plt.tight_layout()
        plt.savefig("combined_azimuth_bins_with_chi2_1631732038.png")
        plt.show()
        plt.close() 
        
        #code to plot before and after (you must run either one before the other.)

        plt.scatter(frequencies.squeeze / MEGA, corrected_bandpass, label = "SW removed bandpass using before scan fit", s = 2)
        plt.plot(frequencies.squeeze / MEGA, fit_curve, color='black', linewidth=0.8, label = "linear fit before")
        plt.errorbar(
                                x=frequencies.squeeze / MEGA,
                                y=corrected_bandpass,
                                yerr=[upper_bound],
                                fmt='none',
                                label = "errorbar before",
                                color='blue',
                                alpha = 0.3
                            )
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Intensity')


        plt.grid()


        plt.scatter(frequencies.squeeze / MEGA, corrected_bandpass_after, label = "SW removed bandpass using after scan fit", s = 2)
        plt.plot(frequencies.squeeze / MEGA , fit_curve_after, label = "linear fit after" ,  color='black', linewidth=0.8)
        plt.errorbar(
                                x=frequencies.squeeze / MEGA,
                                y=corrected_bandpass_after,
                                yerr=[upper_bounds_after],
                                fmt='none',
                                label = "errorbar after",
                                color='green',
                                alpha = 0.3
                            )


        plt.legend(title='Markers', bbox_to_anchor=(-0.1, 1))

        plt.title(f'reduced chi2 for before: {reduced_chi_squared:.5f}, and after: {reduced_chi_squared_after:.5f}')


        plt.savefig("before_and_after_fits_1631732038_m010v.png", bbox_inches='tight')
        plt.show()
        plt.close() 
        print(reduced_chi_squared_time.data)

        # with open("append_data_1631732038.csv", "a", newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([reduced_chi_squared])
        #     for i in range(len(corrected_bandpass)):
        #         row_data = [
        #             frequencies.squeeze[i] / MEGA,
        #             upper_bound[i],
        #             corrected_bandpass[i],
        #             fit_curve[i]
        #         ]
        #         writer.writerow(row_data)
        # stored_values = self.read_reduced_chi_squared_values("reduced_chi_squared_values.txt")
        # print("Stored Reduced Chi-Squared Values:")
        # for i, value in enumerate(stored_values):
        #     print(f"Row {i}: {value}")
        
        # selected_indices = input("Enter the row indices you want to use (comma-separated): ")
        # selected_indices = [int(i) for i in selected_indices.split(',')]
        
        # Plot selected reduced chi-squared values
        # self.plot_selected_reduced_chi_squared(scan_data, selected_indices, frequencies, MEGA)
        # for index in range(len(azimuth_bins)):
        #     time_dumps = np.where(azimuth_digitized == index)[0]
        #     if len(time_dumps) == 0:
        #         continue
        #     visibility = scan_data.visibility.get(time=time_dumps, # for entire scan, do not specify time dumps, and so the same thing for flag
        #                                             freq=target_channels,
        #                                             recv=i_receiver)
        #     flag = scan_data.flags.get(time=time_dumps,
        #                                 freq=target_channels,
        #                                 recv=i_receiver)
        #     # corrected_visibility = visibility / (1 + epsilon)
        #     corrected_visibility = visibility.squeeze / (1+epsilon)
        #     corrected_bandpass = visibility.mean(axis=0, flags=flag).squeeze / (1 + epsilon)
        #     # corrected_azimuth_binned_bandpasses.append(corrected_bandpass)



        #     standard_deviation_bandpass = visibility.standard_deviation(axis=0,flags=flag).squeeze / (1+epsilon)


        #     # Fit a linear curve to the azimuth curve using scipy.optimize.curve_fit
        #     fit_params, _ = curve_fit(self.fit_linear, frequencies.squeeze, corrected_bandpass)



        #     # Calculate chi-squared value for the linear fit
        #     expected = self.fit_linear(frequencies.squeeze, *fit_params)
        #     residuals = corrected_visibility - expected
        #     reduced_chi_squared = np.sum(residuals**2, axis=0)/((standard_deviation_bandpass)**2 * (len(time_dumps)) )


        #     absolute_residuals = np.abs(residuals)  # Take the absolute value
        #     area = simps(absolute_residuals, frequencies.squeeze)


        #     fit_param_std_i = standard_deviation_bandpass
        #     upper_bound =  fit_param_std_i
        #     lower_bound =  -fit_param_std_i
            
        #     if not corrected_bandpass.mask[0]:
        #         corrected_azimuth_binned_bandpasses.append(corrected_bandpass)
        #         azimuth_chi_squared.append(reduced_chi_squared)
        #         standard_deviation_bandpass_list.append(standard_deviation_bandpass)
        #         integrated_areas.append(area)
        #         azimuth_fits.append(fit_params)
        #         upper_bounds.append(upper_bound)
        #         lower_bounds.append(lower_bound)
                
        # with open("reduced_chi_squared_values_.txt", "w") as file:
        #     for value in azimuth_chi_squared:
        #         file.write(f"{value}\n")
        # print("lists obtained")
        # stored_values = self.read_reduced_chi_squared_values("reduced_chi_squared_values.txt")
        
        # plt.figure(figsize=(8, 6))
        # plt.plot(frequencies.squeeze / MEGA , (stored_values), label = "before scan")
        # plt.plot(frequencies.squeeze / MEGA , (reduced_chi_squared.data), label = "after scan")
        # plt.plot(frequencies.squeeze / MEGA , (reduced_chi_squared.data - stored_values), label = "absolute value of residuals")
        # plt.grid()
        # plt.xlabel("Frequencies [MHz]")
        # plt.ylabel("Log Reduced Chi Squared Values")
        # plt.title("Comparision of before and after scan values of m010h, obs block: 1630519596")
        # plt.legend()
        # plt.savefig("chi_comparision_before_after.png")
        # plt.close()
        plt.figure(figsize=(8, 6))
        for i, bandpass in enumerate(corrected_azimuth_binned_bandpasses):
            label = ''
            if i % 12 == 0:
                label = f'azimuth {azimuth_bins[i]:.1f}'
            plt.plot(frequencies.squeeze / MEGA, bandpass, label=label)
            
            # Plot linear fit curve
            fit_params = azimuth_fits[i]
            fit_curve = self.fit_linear(frequencies.squeeze, *fit_params)
            plt.plot(frequencies.squeeze / MEGA, fit_curve, color='black', linestyle='--', linewidth=0.8)
            label_err = ''
            if i % 12 == 0:
                label_err = f'azimuth {azimuth_bins[i]:.1f} Error bar'
            lower_bound = lower_bounds[i]
            upper_bound = upper_bounds[i]
            print("bounds added.........")
            plt.errorbar(
                x=frequencies.squeeze / MEGA,
                y=fit_curve,
                yerr=[upper_bound],
                fmt='none',
                label=label_err,
                color='cyan',
                alpha = 0.02
            )
            # for lower_bound in lower_bounds:
            #     for upper_bound in upper_bounds:
            #         plt.fill_between(frequencies.squeeze / MEGA, lower_bound, upper_bound, color='green', alpha=0.01)
        
        plt.legend()
        plt.xlabel('frequency [MHz]')
        # plt.ylim(325,360)
        plt.ylabel('intensity')
        plot_name = 'standing_wave_correction_scanning_azimuth_bins.png'
        plt.savefig(os.path.join(receiver_path, plot_name))
        plt.close()
        


        # # Print chi-squared values for each azimuth fit
        # print("Chi-squared values for azimuth fits:", azimuth_chi_squared)
        # print("Integrated areas covered by residuals:", integrated_areas)
        
                # Bootstrap method to estimate uncertainties
        # bootstrap_fit_params = []
        # for bandpass in corrected_azimuth_binned_bandpasses:
        #     boot_fit_params = []
        #     for _ in range(n_bootstrap):
        #         # Resample with replacement
        #         boot_bandpass = resample(bandpass, replace=True)
        #         boot_params, _ = curve_fit(self.fit_linear, frequencies.squeeze, boot_bandpass)
        #         boot_fit_params.append(boot_params)
        #     bootstrap_fit_params.append(boot_fit_params)

        # # Calculate standard deviations of bootstrap fit parameters
        # bootstrap_fit_params = np.array(bootstrap_fit_params)
        # fit_param_std = np.std(bootstrap_fit_params, axis=0)
        
    #     plt.figure(figsize=(8, 6))
    #     for i, bandpass in enumerate(corrected_azimuth_binned_bandpasses):
    # # ...

    # # Plot linear fit curve
    #         label = ''
    #         if i % 7 == 0:
    #             label = f'azimuth {azimuth_bins[i]:.1f}'
    #         fit_params = azimuth_fits[i]
    #         fit_curve = self.fit_linear(frequencies.squeeze, *fit_params)
    #         plt.plot(frequencies.squeeze / MEGA, bandpass, label=label)
    #         plt.plot(frequencies.squeeze / MEGA, fit_curve, color='black', linestyle='--', linewidth=0.8,label=label)
    #         plt.ylim(280,400)

            # Calculate upper and lower bounds of the fit curve using fit_params and fit_param_std


            # # Plot error bars on the fit curve
            # label_err = ''
            # if i % 7 == 0:
            #     label_err = f'azimuth {azimuth_bins[i]:.1f} Error bar'
            # plt.errorbar(
            #     x=frequencies.squeeze / MEGA,
            #     y=fit_curve,
            #     yerr=[fit_curve - lower_bound, upper_bound - fit_curve],
            #     fmt='none',
            #     label=label_err,
            #     color='black',
            # )
            

        # plt.legend()
        # plt.xlabel('frequency [MHz]')
        # plt.ylabel('intensity')
        # # plt.ylim(300,315)
        # plot_name = 'standing_wave_correction_scanning_azimuth_bins.png'
        # plt.savefig(os.path.join(receiver_path, plot_name))
        # plt.close()

        # Print chi-squared values for each azimuth fit
        print("Chi-squared values for azimuth fits:", azimuth_chi_squared)
        
        m010h_residual_areas_1631552188 = np.array([ 9160515.48862845,  8927956.11246097,  9025844.69040565,
        9421534.05297259,  9921329.70199034, 10342557.47477862,
        9296040.60758734,  9718184.05405944,  9746523.85061968,
        9768973.20765154,  8652713.59009258,  8727622.45208029,
        9120297.97623731,  9582788.78580422, 10010767.59749657,
       10181317.66236299, 10093581.347679  ,  8778359.2048842 ,
        9232024.58185859,  8943113.04368305,  8916263.60529438,
        8653027.12780839,  8321424.41861824,  9051333.72268278,
        8868645.80987151,  8796879.01849444,  8542086.38516399,
        8619701.38115387,  8923193.06904417,  8761626.03962359,
        8702922.68185715,  9333745.33190565,  9431508.27737902,
        9494459.22848812,  9760337.4360182 ,  9744499.72182532,
        9864671.62721698,  9846499.82149825,  9672970.89006116,
        9931864.47184641,  9787439.0071212 ,  9905149.91127782,
       10107237.07102066, 10084621.56891663,  9744471.13139952,
       10574239.6162585 , 10423415.337349  , 10936404.83952651,
       10885687.35953212]) 
        m010h_chi2_1631552188 = np.array([0.0061985 , 0.0053679 , 0.0062715 , 0.00542227, 0.00510782,
       0.00650572, 0.0038141 , 0.00426888, 0.00371859, 0.00376694,
       0.00316021, 0.00315374, 0.00344323, 0.00376321, 0.00413506,
       0.00464121, 0.00507143, 0.0040521 , 0.0051564 , 0.00450354,
       0.00499759, 0.0043199 , 0.00454216, 0.00545451, 0.0047436 ,
       0.00596318, 0.00534594, 0.00599456, 0.00708648, 0.00746431,
       0.00730218, 0.00930878, 0.00963906, 0.00979807, 0.00913544,
       0.00958807, 0.00994203, 0.01011681, 0.01169483, 0.01129488,
       0.01210922, 0.01146448, 0.01226025, 0.01232862, 0.012179  ,
       0.01336331, 0.01584785, 0.01594496, 0.01932916])
        m010v_chi2_1631552188 = np.array([0.00961698, 0.0068553 , 0.00671801, 0.00558737, 0.00372762,
       0.00425359, 0.00311786, 0.00329967, 0.00271374, 0.00407738,
       0.00359569, 0.00309392, 0.0037998 , 0.00316597, 0.0027671 ,
       0.00244338, 0.00355916, 0.00378028, 0.00489601, 0.00447045,
       0.00484702, 0.00389999, 0.00564971, 0.00531699, 0.00442988,
       0.00476294, 0.00450174, 0.00450074, 0.00569752, 0.00659028,
       0.00809771, 0.01018342, 0.01062118, 0.00948811, 0.00870918,
       0.0078532 , 0.00831004, 0.00889625, 0.01168733, 0.01029861,
       0.01141115, 0.01074737, 0.01066417, 0.01077132, 0.00916835,
       0.00960586, 0.00967038, 0.00991591, 0.01290529])
        m010v_residual_areas_1631552188 = np.array([5488209.51783557, 5313785.25954837, 5163524.60676379,
       5020820.02004368, 4550217.29497527, 4712601.79520412,
       4861933.02779889, 4806452.63633632, 4826005.39063941,
       5218648.78145517, 5195864.5319379 , 4659789.40854539,
       5181079.41967481, 4825856.05327855, 4760604.13905325,
       4516978.72142541, 4739726.19517506, 4727752.85019114,
       4730769.21060908, 4913995.49702127, 4936438.91458621,
       4903145.83974787, 5169954.62816498, 4833266.94539268,
       4829129.73208362, 4494933.46121966, 4359615.62327611,
       4090318.84696129, 4592603.03592917, 4552735.16211393,
       4899142.80086269, 4835889.00238181, 4858323.58054427,
       4770376.8761648 , 4686425.44274632, 4585682.32605812,
       4651168.64969916, 4670019.86751842, 4920297.3828828 ,
       4805510.5407752 , 4955837.78590556, 5113888.00987978,
       4801874.25662246, 4992959.88433645, 4452839.25144907,
       4712346.68563318, 4277212.53135217, 4419954.13341464,
       4910992.82332742])
        m040h_residual_areas_1631552188 = np.array([12355346.15693632, 11955222.97108265, 12124596.06903041,
       11839577.8205317 , 11740043.22379675, 11726830.31534059,
       11058986.51300687, 11163851.64659864, 11546780.82609729,
       11429422.79445242, 11434857.99004299, 11557035.5781376 ,
       11086259.04663388, 10808825.38470115, 10614644.73113672,
       11062886.0976388 , 10484067.04538073,  9982808.55706811,
       10876387.84807691, 10958535.33468965, 11285829.79989643,
       11216779.31170222, 11369818.33695806, 11387295.95471189,
       10926549.64599662, 10979913.22801339, 11487565.36018488,
       11341912.16537985, 11429571.48194746, 11632590.57366701,
       12251995.56379883, 12005475.31623814, 12111429.64677826,
       12158205.13254864, 11777906.83146614, 11469844.4314661 ,
       11309955.07934154, 11275343.31165793, 11517352.7329532 ,
       11800731.86064504, 11638864.5779322 , 12356062.05810032,
       11616547.49156867, 11331604.58614347, 11427729.95912414,
       11163212.12346748, 10939716.38961225, 10952982.11186103,
       11055236.72685738])
        m040v_residual_areas_1631552188 = np.array([ 8610760.61395122,  8435825.39650999,  8443006.33435473,
        8576337.18770553,  8516430.12170449,  9005160.29566733,
        8883291.30192319,  8497035.46814062,  8975119.3275819 ,
        8622504.02910834,  8284476.15363811,  8563152.36073259,
        8407979.92408241,  8744513.39794916,  8581005.17569258,
        8736142.78496522,  8529403.47382181,  8683919.8725752 ,
        8727940.49176712,  8889895.83399616,  9470035.93883482,
        9813343.24976479,  9918490.14030305,  9566347.29290569,
        9687789.62283918,  9731513.55597351,  9789103.58595463,
       10144056.50893451,  9743198.39533701, 10226838.76052894,
       10551805.14244366, 10373269.55149337,  9965227.00465316,
       10029107.74864754, 10109457.61787493, 10245987.94892028,
       10254302.46222742, 10068161.37383018, 10104878.14524079,
        9815968.89169731,  9994219.55863244, 10071848.9514405 ,
        9943805.29997154,  9476373.56215658, 10023175.1787307 ,
       10211872.51692263, 10389010.21062997,  9751823.14476575,
       10172624.69092654])
        m040h_chi2_1631552188 = np.array([0.03704786, 0.03144968, 0.03045505, 0.02620373, 0.01824167,
       0.0195549 , 0.0137769 , 0.01243706, 0.01168423, 0.01347749,
       0.01466393, 0.0134169 , 0.01189428, 0.01199045, 0.0123677 ,
       0.01210335, 0.01531985, 0.01358679, 0.02065788, 0.0187194 ,
       0.02006273, 0.02024684, 0.0203231 , 0.02054352, 0.01956279,
       0.0210426 , 0.02516051, 0.02472077, 0.03363619, 0.03074612,
       0.04173869, 0.04704594, 0.05916049, 0.05602753, 0.04896074,
       0.04910365, 0.05577227, 0.04410778, 0.04486297, 0.05177606,
       0.05633767, 0.05867914, 0.0510947 , 0.06054604, 0.05000937,
       0.04835139, 0.05462037, 0.04657461, 0.06296614])
        m040v_chi2_1631552188 = np.array([0.00609264, 0.00589773, 0.00569939, 0.00550888, 0.00467622,
       0.0051266 , 0.0040557 , 0.0035598 , 0.00360499, 0.00375103,
       0.00336386, 0.0037747 , 0.00364994, 0.00362075, 0.00359709,
       0.00382525, 0.00398138, 0.00405424, 0.00520309, 0.00505597,
       0.00557383, 0.00646295, 0.00684594, 0.00704355, 0.00623557,
       0.00755005, 0.0087318 , 0.00865255, 0.00895922, 0.0096593 ,
       0.01111332, 0.01140899, 0.01154462, 0.01217797, 0.01231184,
       0.01292489, 0.01369508, 0.01261454, 0.01496642, 0.01550139,
       0.01691187, 0.01707956, 0.01613513, 0.01574439, 0.01695797,
       0.01796195, 0.02236333, 0.01868864, 0.02317695])
        m046h_chi2_1631552188 = np.array([0.00110197, 0.0010667 , 0.00082294, 0.00062779, 0.00063123,
       0.00080299, 0.00073986, 0.00085119, 0.00070048, 0.00078139,
       0.00066982, 0.00052814, 0.000631  , 0.00048467, 0.00059825,
       0.00058186, 0.00067166, 0.00058276, 0.00080018, 0.00077356,
       0.00095633, 0.00077495, 0.00099615, 0.00100767, 0.00088144,
       0.00113784, 0.00091634, 0.00118032, 0.00121927, 0.00116582,
       0.00126109, 0.0011747 , 0.00109075, 0.00100309, 0.00092954,
       0.00120743, 0.00111603, 0.00106437, 0.00147121, 0.00148178,
       0.00136595, 0.00101649, 0.0009823 , 0.00080932, 0.00085048,
       0.00090281, 0.00102903, 0.00119796, 0.00215163])
        m046h_residual_areas_1631552188 = np.array([5214888.82799563, 5158801.74657272, 4637175.30895219,
       4370457.42266357, 4320921.9050896 , 4873131.94105905,
       5312444.15261921, 5375558.04650274, 5181028.0731301 ,
       5415614.22959278, 4975536.51526321, 4467604.34994517,
       4783073.60571714, 4203942.85931927, 5027304.44050634,
       4726386.10408392, 4904974.46201705, 4628540.66930255,
       4908000.32471544, 5110046.36998261, 5651300.41513681,
       5055486.09199602, 5477914.3377226 , 5347612.74492662,
       5156294.92169505, 5769148.19110039, 5092265.86139801,
       5906660.44797142, 5728933.8692983 , 5437559.19726801,
       5582445.76792393, 5281721.25862863, 5248499.19613176,
       4796536.80579785, 5015372.32739659, 5283946.6476417 ,
       5338807.58346011, 4858136.45789547, 5682388.69441451,
       5570121.00966593, 4911955.68492807, 4522218.25350822,
       4585920.33793297, 3954041.25030668, 4153141.78500711,
       4140795.77447221, 4032954.19895881, 4429571.82020828,
       5510092.21714108])
        
        m010h_chi2_1634252028 = np.array([0.14596002, 0.1038269 , 0.10223009, 0.10301323, 0.10202008,
       0.12127511, 0.10474606, 0.08546199, 0.07781141, 0.0701343 ,
       0.05801853, 0.05474356, 0.06067166, 0.05099787, 0.04516449,
       0.04399006, 0.05209853, 0.05169473, 0.04842863, 0.04169588,
       0.03455117, 0.02900304, 0.03176205, 0.03027298, 0.02938964,
       0.02929592, 0.03159839, 0.03163645, 0.03376532, 0.03013343,
       0.02957869, 0.02833454, 0.02547714, 0.02441941, 0.02528344,
       0.01881165, 0.01788447, 0.01780898, 0.01721931, 0.0171239 ,
       0.01959558, 0.02177978, 0.02465891, 0.02380727, 0.0252588 ,
       0.02730045, 0.02786632, 0.02955694, 0.0334196 ])
        m010h_residual_areas_1634252028 = np.array([17395104.18647366, 17928910.91464433, 17547717.21448001,
       17561630.27965948, 17342527.53059829, 16754366.84339206,
       16526777.98928882, 16576898.78781796, 16252560.59555835,
       16298861.25447189, 16822040.89662511, 16579084.4361109 ,
       16824207.3628158 , 16595987.00630992, 16605059.06261946,
       16496675.8981383 , 16907630.11131738, 17059964.67935136,
       17198526.68801132, 17335268.08809962, 16937041.2620685 ,
       16976298.02149621, 17038784.984951  , 17382885.40405053,
       17053415.89888017, 16718588.91664141, 16866587.49496975,
       16792829.42878332, 17013922.01578563, 16819735.01175905,
       17044721.96179239, 17221909.12258203, 17428338.3635155 ,
       17267347.07212583, 17489298.24904904, 17690688.52549502,
       17341714.17031175, 16918158.53586618, 17004170.57967002,
       16482348.79079189, 17138578.70318184, 17382007.21291405,
       17583753.38973947, 17699741.12657089, 17616490.91815812,
       17257940.35503786, 17441862.11493286, 17083388.9040428 ,
       17696694.79128801])
        m010v_chi2_1634252028 = np.array([0.1363017 , 0.08584491, 0.07751642, 0.0819909 , 0.08326793,
       0.09560591, 0.08444721, 0.0692917 , 0.06460909, 0.049197  ,
       0.04588344, 0.04144925, 0.04556864, 0.04148209, 0.03706398,
       0.03759348, 0.04944491, 0.04543283, 0.04175182, 0.03695776,
       0.02732769, 0.02306641, 0.02390894, 0.02444357, 0.02245153,
       0.02444639, 0.02528056, 0.02439036, 0.02490942, 0.02447193,
       0.02500049, 0.024032  , 0.02124769, 0.02105336, 0.02189919,
       0.01711193, 0.01645507, 0.01665462, 0.01693216, 0.01779569,
       0.02043424, 0.02029388, 0.0240671 , 0.02225512, 0.02540034,
       0.02729821, 0.0252727 , 0.0267759 , 0.02841419])
        m010v_residual_areas_1634252028 = np.array([10407260.04040186,  9923080.57322741,  9960515.91734216,
        9985076.14703874, 10305381.04404947, 10132490.2554045 ,
       10299615.91195438, 10027721.75127126, 10207729.62247026,
       10031551.26517773, 10109765.73558269, 10112737.85247754,
       10095770.64829674, 10311842.51947742, 10085382.18774761,
       10367395.20353269, 10644735.13873839, 10536563.06162861,
       10210946.86299404, 10413507.6385445 ,  9856144.00543549,
        9841131.70369554,  9954126.72617999, 10334040.07813578,
       10101132.95639485, 10318508.18208953, 10169619.37063908,
       10206768.39861694, 10251281.9474805 , 10194738.35395216,
       10289141.19887438, 10639342.23782296, 10295720.82457381,
       10440421.9887722 , 10401158.34390301, 10838517.52687074,
       10538260.70058996, 10924107.83995327, 11050278.10370287,
       11048981.37699686, 11438102.43636309, 11243059.12223878,
       11454978.25211116, 11567908.14646957, 11359476.42165717,
       11136584.31141684, 11188694.5792147 , 11174327.11018449,
       11173292.95707849])
