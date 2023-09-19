from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.result import Result
from museek.flag_list import FlagList
from ivory.utils.requirement import Requirement
from museek.enum.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.data_element import DataElement
from museek.flag_list import FlagList
from museek.time_ordered_data_mapper import TimeOrderedDataMapper
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from scipy.stats import norm, lognorm
import numpy as np
# import pickle
import json
from scipy.stats import norm
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 24


# Load the JSON data from the file

class ParameterDistributionPlugin(AbstractPlugin):
      
    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='data'),
                            Requirement(location=ResultEnum.TRACK_DATA, variable='track_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path')]
        
    def run(self, data: TimeOrderedData,track_data: TimeOrderedData, output_path: str):
        
        # Load the JSON data from the file
        with open("/users/pranav/Project/museek/results/1631732038/parameters_dict_frequency_975_to_1016_MHz.json", "r") as json_file:
            data = json.load(json_file)

        set1 = []
        set2 = []

        # Iterate through each receiver
        for receiver, receiver_data in data.items():
            before_scan_data = receiver_data.get("before_scan", {})
            after_scan_data = receiver_data.get("after_scan", {})
            
            for key, value in before_scan_data.items():
                if "wavelength" in key:
                    wavelength = float(key.split("_")[1])
                    phase = before_scan_data.get(f"wavelength_{wavelength}_phase", None)
                    amplitude = before_scan_data.get(f"wavelength_{wavelength}_amplitude", None)
                    if phase is not None and amplitude is not None:
                        set1.append((phase, amplitude, wavelength))
                        
            for key, value in after_scan_data.items():
                if "wavelength" in key:
                    wavelength = float(key.split("_")[1])
                    phase = after_scan_data.get(f"wavelength_{wavelength}_phase", None)
                    amplitude = after_scan_data.get(f"wavelength_{wavelength}_amplitude", None)
                    if phase is not None and amplitude is not None:
                        set2.append((phase, amplitude, wavelength))
                        
                        
            # Convert data to numpy arrays for easier manipulation
        set1 = np.array(set1)
        set2 = np.array(set2)
        
        subset1 = set1[:16]
        subset2 = set2[:16]
        
        real_parts_set1_m010h = set1[:, 0]
        imag_parts_set1_m010h = set1[:, 1]
        wavelengths_set1_m010h = set1[:, 2]

        real_parts_set2_m010h = set2[:, 0]
        imag_parts_set2_m010h = set2[:, 1]
        wavelengths_set2_m010h = set2[:, 2]
        
        real_parts_set1 = real_parts_set1_m010h[16:32]
        imag_parts_set1 = imag_parts_set1_m010h[16:32]
        wavelengths_set1 = wavelengths_set1_m010h[16:32]
        real_parts_set2 = real_parts_set2_m010h[16:32]
        imag_parts_set2 = imag_parts_set2_m010h[16:32]
        wavelengths_set2 = wavelengths_set2_m010h[16:32]

        # Extract real, imaginary parts, and wavelengths


        # Create a 3D subplot
        
        # Create unique colors for each wavelength
        colors_set1 = plt.cm.gist_rainbow(wavelengths_set1 / max(wavelengths_set1))
        colors_set2 = plt.cm.gist_rainbow(wavelengths_set2 / max(wavelengths_set2))

        # Create scatter plots with different markers and colors
        # plt.figure(figsize=(10, 6))

        # scatter_set1 = plt.scatter(real_parts_set1, imag_parts_set1, c=colors_set1, marker='o', alpha=0.7, label='Set 1')
        # scatter_set2 = plt.scatter(real_parts_set2, imag_parts_set2, c=colors_set2, marker='s', alpha=0.7, label='Set 2')


    # # Create a custom legend for wavelengths only
    #         # wavelengths_arr = [29.4, 26.8, 32.4, 35.8, 24.8, 39.2, 11.6]
        fig=plt.figure(figsize=(10, 6))

        
        ax=fig.add_subplot(111,polar=True)
        
        unique_wavelengths = [29.4, 26.8, 32.4, 35.8, 24.8, 39.2, 11.6]
        
        legend_entries = []
        
        for wavelength in unique_wavelengths:
            mask_set1 = wavelengths_set1 == wavelength
            mask_set2 = wavelengths_set2 == wavelength

            ax.scatter(real_parts_set1[mask_set1], imag_parts_set1[mask_set1], c=colors_set1[mask_set1], marker='o', alpha=0.7, label=f'Wavelength: {wavelength}')
            ax.scatter(real_parts_set2[mask_set2], imag_parts_set2[mask_set2], c=colors_set2[mask_set2], marker='d', alpha=0.7)
            
            

        
        existing_handles, existing_labels = ax.get_legend_handles_labels()

        # Create custom legend entries for the marker types
        legend_entries = [
            Line2D([0], [0], marker='o', color='k', markersize=5, label='Before Scan'),
            Line2D([0], [0], marker='d', color='k', markersize=5, label='After Scan')
        ]

        # Combine existing legend entries with custom marker entries
        all_handles = existing_handles + legend_entries
        all_labels = existing_labels + ['Before Scan'] + ['After Scan']

        # Add custom legend entries to the existing legend
        ax.legend(handles=all_handles, labels=all_labels, title='Markers', loc='upper right')
        ax.grid()
        plt.show()
        plt.savefig("what_is_happening_polar_1631732038_m010v.png", bbox_inches = 'tight')
        plt.close()


        unique_wavelengths = [29.4, 26.8, 32.4, 35.8, 24.8, 39.2, 11.6]
        legend_entries = []
        for wavelength in unique_wavelengths:
            mask_set1 = wavelengths_set1 == wavelength
            mask_set2 = wavelengths_set2 == wavelength

            plt.scatter(real_parts_set1[mask_set1], imag_parts_set1[mask_set1], c=colors_set1[mask_set1], marker='o', alpha=0.7, label=f'Wavelength: {wavelength}')
            plt.scatter(real_parts_set2[mask_set2], imag_parts_set2[mask_set2], c=colors_set2[mask_set2], marker='d', alpha=0.7)
            
            

        plt.xlabel('Phase')
        plt.ylabel('Amplitude')
        plt.title('Parameters with different Standing Wave Wavelengths')
        existing_handles, existing_labels = plt.gca().get_legend_handles_labels()

        # Create custom legend entries for the marker types
        legend_entries = [
            Line2D([0], [0], marker='o', color='k', markersize=5, label='Before Scan'),
            Line2D([0], [0], marker='d', color='k', markersize=5, label='After Scan')
        ]

        # Combine existing legend entries with custom marker entries
        all_handles = existing_handles + legend_entries
        all_labels = existing_labels + ['Before Scan'] + ['After Scan']

        # Add custom legend entries to the existing legend
        plt.legend(handles=all_handles, labels=all_labels, title='Markers', loc='upper right')
        plt.grid()
        plt.show()
        plt.savefig("what_is_happening_1631732038_m010v.png")
        plt.close()
    #         # Print the extracted sets
    #         # print("set1 =", set1)
    #         # print("set2 =", set2)

    #         # Combine all points
    #         all_real_parts = np.concatenate((real_parts_set1, real_parts_set2))
    #         all_imag_parts = np.concatenate((imag_parts_set1, imag_parts_set2))

    #         # Fit Gaussian distribution
    #         mean_real = np.mean(all_real_parts)
    #         std_real = np.std(all_real_parts)

    #         mean_imag = np.mean(all_imag_parts)
    #         stddev_lognormal = np.std(np.log(all_imag_parts / (1 - all_imag_parts)))  # Calculate standard deviation in log-space
    #         mean_lognormal = np.log(mean_imag / (1 - mean_imag))  # Calculate mean in log-space

    #         # Create a range for plotting the distributions
    #         x_range_real = np.linspace(mean_real - 3 * std_real, mean_real + 3 * std_real, 100)
    #         x_range_imag = np.linspace(0, 1, 100)  # Amplitude is restricted between 0 and 1
    #         x_range_lognormal = np.linspace(0.001, 0.999, 100)  # Avoid zero and one for log-normal

    #         # Evaluate Gaussian and log-normal probability density functions
    #         pdf_real = norm.pdf(x_range_real, mean_real, std_real)
    #         pdf_imag = lognorm.pdf(x_range_imag, stddev_lognormal, scale=np.exp(mean_lognormal))

    #         # Plot distributions side by side
    #         plt.figure(figsize=(12, 6))

    #         # Plot for Phase
    #         plt.subplot(1, 2, 1)
    #         plt.plot(x_range_real, pdf_real)
    #         plt.xlabel('Phase Value')
    #         plt.ylabel('Probability Density')
    #         plt.title('Phase Distribution')
    #         plt.grid()

    #         # Plot for Log-Normal Amplitude
    #         plt.subplot(1, 2, 2)
    #         plt.plot(x_range_imag, pdf_imag, color='orange')
    #         plt.xlabel('Amplitude Value')
    #         plt.ylabel('Probability Density')
    #         plt.title('Log-Normal Amplitude Distribution')
    #         plt.grid()

    #         plt.tight_layout()
    #         plt.show()
    #         plt.savefig("side_by_side_param_lot.png")
    #         plt.close()

        valid_indices_set1 = imag_parts_set1 > 0.0
        valid_indices_set2 = imag_parts_set2 > 0.0

        # Apply the filter to extract valid data
        filtered_set1 = subset1[valid_indices_set1]
        filtered_set2 = subset2[valid_indices_set2]

        # Extract real, imaginary parts, and wavelengths for filtered data
        filtered_real_parts_set1 = filtered_set1[:, 0]
        filtered_imag_parts_set1 = filtered_set1[:, 1]
        filtered_wavelengths_set1 = filtered_set1[:, 2]

        filtered_real_parts_set2 = filtered_set2[:, 0]
        filtered_imag_parts_set2 = filtered_set2[:, 1]
        filtered_wavelengths_set2 = filtered_set2[:, 2]
        
        updated_colors_set1 = plt.cm.gist_rainbow(filtered_wavelengths_set1 / max(filtered_wavelengths_set1))
        updated_colors_set2 = plt.cm.gist_rainbow(filtered_wavelengths_set2 / max(filtered_wavelengths_set2))

        # Create a list of all unique wavelengths
        unique_wavelengths = np.unique(np.concatenate((filtered_wavelengths_set1, filtered_wavelengths_set2)))

        # Create subplots for each wavelength
        num_wavelengths = len(unique_wavelengths)
        num_columns = 2  # Two subplots per row
        num_rows = int(np.ceil(num_wavelengths / num_columns))
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 6 * num_rows))

        # Create subplots for each wavelength
        for i, wavelength in enumerate(unique_wavelengths):
            row = i // num_columns
            col = i % num_columns
            ax = axes[row, col]

            mask_set1 = filtered_wavelengths_set1 == wavelength
            mask_set2 = filtered_wavelengths_set2 == wavelength

            imag_parts_set1_wl = filtered_imag_parts_set1[mask_set1]
            imag_parts_set2_wl = filtered_imag_parts_set2[mask_set2]

            # Check if there are valid data points for this wavelength
            if len(imag_parts_set2_wl) > 0:
                # Calculate parameters for log-normal distribution
                mean_imag_wl = np.mean(imag_parts_set2_wl)
                stddev_lognormal_wl = np.std(np.log(imag_parts_set2_wl / (1 - imag_parts_set2_wl) + 1e-9))
                mean_lognormal_wl = np.log(mean_imag_wl / (1 - mean_imag_wl) + 1e-9)

                # Create a range for plotting the distributions
                x_range_wl = np.linspace(0.001, 0.999, 100)

                # Evaluate log-normal probability density function
                pdf_wl = lognorm.pdf(x_range_wl, stddev_lognormal_wl, scale=np.exp(mean_lognormal_wl))

                # Plot log-normal distribution for the wavelength
                ax.plot(x_range_wl, pdf_wl, color='blue', label=f'Wavelength: {wavelength:.2f}')

                # Add parameter statistics as a legend
                legend_text = f'Mean: {mean_imag_wl:.2f}, StdDev: {stddev_lognormal_wl:.2f}'
                ax.legend([legend_text], loc='upper right')

            ax.set_xlabel('Amplitude Value')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Log-Normal Amplitude Distribution')
            ax.grid()

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig("wavelength_distributions_1631732038_m010v.png")
        plt.show()
        plt.close()

        # Create a side-by-side plot for the updated set of points
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original Points - Set 1 and Set 2
        ax1 = axes[0]
        for wavelength in unique_wavelengths:
            mask_set1 = wavelengths_set1 == wavelength
            mask_set2 = wavelengths_set2 == wavelength

            ax1.scatter(real_parts_set1[mask_set1]-real_parts_set2[mask_set1], imag_parts_set1[mask_set1]-imag_parts_set2[mask_set2], c=colors_set1[mask_set1], marker='o', alpha=0.7, label=f'Wavelength: {wavelength:.2f}')
                # ax1.scatter(real_parts_set2[mask_set2], imag_parts_set2[mask_set2], c=colors_set2[mask_set2], marker='d', alpha=0.7)

        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Original Parameters - Difference between before and after scan')
        ax1.legend(title='Markers', loc='upper right')
        ax1.grid()

        # Updated Points - Set 1 and Set 2
        ax2 = axes[1]
        for wavelength in unique_wavelengths:
            mask_set1 = filtered_wavelengths_set1 == wavelength
            mask_set2 = filtered_wavelengths_set2 == wavelength

            ax2.scatter(filtered_real_parts_set1[mask_set1] - filtered_real_parts_set2[mask_set2], np.log(filtered_imag_parts_set1[mask_set1] - filtered_imag_parts_set2[mask_set2]), c=updated_colors_set1[mask_set1], marker='o', alpha=0.7, label=f'Wavelength: {wavelength:.2f}')
            # ax2.scatter(filtered_real_parts_set2[mask_set2], np.log(filtered_imag_parts_set2[mask_set2]), c=updated_colors_set2[mask_set2], marker='d', alpha=0.7)

        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Updated Parameters (Log Plot ) - Differences between before and after scan')
        ax2.legend(title='Markers', loc='lower right')
        
        ax2.grid()
        



        # ax2.set_ylim(-10,0)



        plt.tight_layout()
        plt.savefig("updated_subplots_v_2_1631732038_m010v.png")
        plt.show()
        plt.close()
            
            
    

        fig, ax1 = plt.subplots(figsize=(16, 8), subplot_kw={'projection': 'polar'})

        for wavelength in unique_wavelengths:
            mask_set1 = wavelengths_set1 == wavelength
            mask_set2 = wavelengths_set2 == wavelength
            ax1.scatter(real_parts_set1[mask_set1], imag_parts_set1[mask_set1], c=colors_set1[mask_set1], marker='o', alpha=0.7, label=f'Wavelength: {wavelength:.2f}')
            ax1.scatter(real_parts_set2[mask_set2], imag_parts_set2[mask_set2], c=colors_set2[mask_set2], marker='d', alpha=0.7)
                    
            set1_real_parts = real_parts_set1[mask_set1]
            set1_imag_parts = imag_parts_set1[mask_set1]
            set2_real_parts = real_parts_set2[mask_set2]
            set2_imag_parts = imag_parts_set2[mask_set2]
                    
            for r1, i1, r2, i2, color in zip(set1_real_parts, set1_imag_parts, set2_real_parts, set2_imag_parts, colors_set1[mask_set1]):
                ax1.plot([r1, r2], [i1, i2], color=color, alpha=0.3)

        ax1.set_xlabel('Phase', labelpad=20)
        ax1.set_ylabel('Amplitude', labelpad=40)
        ax1.set_title('Comparision of standing wave parameters for the two fitted models in m010v')
        ax1.legend(title='Markers', bbox_to_anchor=(-0.2, 1))  # Adjust the bbox_to_anchor values

        ax1.grid(True)

        # Increase the number of radial grid lines
        ax1.locator_params(axis='y', nbins=8)  # You can adjust the nbins value as needed
        plt.savefig("original_updated_subplots_joined_1631732038_m010v_zoom.png", bbox_inches='tight')
        plt.show()
        plt.close()