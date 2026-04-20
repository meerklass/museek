import gc
import pickle
import warnings
from collections.abc import Generator

import astropy
import astropy.coordinates as ac
import numpy as np
import pysm3.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from scipy import ndimage

from museek.enums.result_enum import ResultEnum
from museek.time_ordered_data import TimeOrderedData
from museek.util.tools import (
    fill_masked_regions_polyfit,
    find_continuous_mask_regions,
    polynomial_flag_outlier,
    project_2d,
)


class InpaintingMapmakingPlugin(AbstractParallelJoblibPlugin):
    """Plugin to inapinting and mapmaking, using post calibrated data. This plugin is used for 
    inpainting and map making for a single block, after runing this for each block, the outputs 
    from different blocks can be combined using an external notebook."""

    def __init__(
        self,
        threshold_MHz: float,
        inpainting_window: float,
        inpainting_polydeg: int,
        mask_antnum_threshold: float,
        pix_reso: float,
        x_crval: float,
        y_crval: float,
        x_range: float,
        y_range: float,
        do_store_context: bool,
        **kwargs,
    ):
        """
        Initialise the plugin
        :param threshold_MHz: if a long continuous frequency region is masked, this timestamp will be totally masked [MHz]
        :param inpainting_window: inpainting the masked regions by fitting a polynomial using +-inpainting_window of the unmasked data around the masked regions [MHz]
        :param inpainting_polydeg: the degree of polynomials fit in inpainting
        :param mask_antnum_threshold: masking the pixels where <=mask_antnum_threshold antennas contributes
        :param pix_reso: map resolution [deg]
        :param x_crval: map center x(ra) [deg]
        :param y_crval: map center y(dec) [deg]
        :param x_range: map range (+-x_range) in x(ra) [deg]
        :param y_range: map range (+-y_range) in y(dec) [deg]
        :param do_store_context: save the output or not
        """
        super().__init__(**kwargs)
        self.threshold_MHz = threshold_MHz
        self.inpainting_window = inpainting_window
        self.inpainting_polydeg = inpainting_polydeg
        self.mask_antnum_threshold = mask_antnum_threshold
        self.pix_reso = pix_reso
        self.x_crval = x_crval
        self.y_crval = y_crval
        self.x_range = x_range
        self.y_range = y_range
        self.do_store_context = do_store_context

    def set_requirements(self):
        """
        Set the requirements, the scanning data `scan_data`, a path to store results and the name of the data block.
        """
        self.requirements = [
            Requirement(location=ResultEnum.SCAN_DATA, variable="scan_data"),
            Requirement(location=ResultEnum.CALIBRATED_VIS, variable="calibrated_data"),
            Requirement(location=ResultEnum.FREQ_SELECT, variable="freq_select"),
            Requirement(location=ResultEnum.OUTPUT_PATH, variable="output_path"),
            Requirement(location=ResultEnum.BLOCK_NAME, variable="block_name"),
            Requirement(location=ResultEnum.KNOWN_RFI_LIST, variable="rfi_list"),
        ]

    def map(
        self,
        scan_data: TimeOrderedData,
        calibrated_data: np.ma.MaskedArray,
        freq_select: np.ndarray,
        output_path: str,
        block_name: str,
        rfi_list: list[tuple[float, float]],
    ) -> Generator[
        tuple[
            astropy.wcs.WCS,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ma.MaskedArray,
            np.ndarray,
        ],
        None,
        None,
    ]:
        """
        Yield a `tuple` of the results path for one antenna, the scanning calibrated data for one antenna and the flag for one antenna.
        :param scan_data: time ordered data containing the scanning part of the observation
        :param calibrated_data: calibrated data containing the scanning part of the observation
        :param freq_select: frequency for calibrated data, in [Hz]
        :param output_path: path to store results
        :param block_name: name of the data block
        :param rfi_list: list of known rfi
        """

        #########  delete the visibility, flags, and weights of raw data
        scan_data.delete_visibility_flags_weights(polars="auto")

        ##########  mask known RFI
        # Start with all False (unmasked)
        mask_freq = np.zeros_like(freq_select, dtype=bool)
        # Set mask to True within each interval (inclusive bounds)
        for low, high in rfi_list:
            mask_freq |= (freq_select / 10**6 >= low) & (freq_select / 10**6 <= high)

        calibrated_data.mask[:, mask_freq, :] = True

        ################# define the wcs for map
        # Create a new WCS object with two dimensions
        wcs_map = WCS(naxis=2)

        x_min = self.x_crval - self.x_range  # map region x(ra)
        x_max = self.x_crval + self.x_range  # map region x(ra)
        y_min = self.y_crval - self.y_range  # map region y(dec)
        y_max = self.y_crval + self.y_range  # map region y(dec)

        crpix_x = int((x_max - x_min) / self.pix_reso / 2.0)
        crpix_y = int((y_max - y_min) / self.pix_reso / 2.0)

        # Assuming the RA ad Dec axis are the first and second axis respectively
        # Define the reference pixel (center of the array or some other reference)
        wcs_map.wcs.crpix = [crpix_x, crpix_y]  # Middle of the array
        wcs_map.wcs.cdelt = np.array(
            [self.pix_reso, self.pix_reso]
        )  # degrees/pixel for RA and Dec
        wcs_map.wcs.crval = [(x_max + x_min) / 2.0, (y_max + y_min) / 2.0]
        wcs_map.wcs.ctype = [
            "RA---ZEA",
            "DEC--ZEA",
        ]  # Use ZEA projection for RA and Dec
        # Optionally set unit types
        wcs_map.wcs.cunit = ["deg", "deg"]

        # set the shape of map
        map_shape = (
            np.round((x_max - x_min) / self.pix_reso).astype(int) + 1,
            np.round((y_max - y_min) / self.pix_reso).astype(int) + 1,
        )

        # Assume image_shape = [ra_length, dec_length]
        ra_len, dec_len = map_shape[0], map_shape[1]

        ########################################
        for i_antenna, antenna in enumerate(scan_data.antennas):
            right_ascension = scan_data.right_ascension.get(recv=i_antenna).squeeze
            declination = scan_data.declination.get(recv=i_antenna).squeeze
            yield (
                wcs_map,
                map_shape,
                right_ascension,
                declination,
                calibrated_data[:, :, i_antenna],
                freq_select,
                mask_freq,
            )

    def run_job(
        self,
        anything: tuple[
            astropy.wcs.WCS,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ma.MaskedArray,
            np.ndarray,
            np.ndarray,
        ],
    ) -> tuple[np.ma.MaskedArray,np.ndarray,np.ndarray,np.ndarray,astropy.wcs.WCS]:
        """
        Run the inpainting and mapmaking. Done for one antenna at a time.
        :param anything: `tuple` of the wcs_map, map_shape, ra_ant, dec_ant, calibrated_data_ant, freq_select, mask_freq
        :return: map making data for each antenna, hit map, hit map without considering mask, known rfi flag, wcs map
        """
        (
            wcs_map,
            map_shape,
            ra_ant,
            dec_ant,
            calibrated_data_ant,
            freq_select,
            mask_freq,
        ) = anything

        calibrated_data_ant = calibrated_data_ant / 10**6.0  ### convert to K
        ##### if a long continuous frequency region is masked, this timestamp will be totally masked,
        ##### because it is hard to do the inpainting
        # Frequency resolution (assuming evenly spaced)
        freq_resolution = np.abs(np.diff(freq_select).mean()) / 10**6.0
        assert freq_resolution < 100, (
            "freq_resolution too large, check unit, it must be in MHz unit"
        )
        # Threshold for what counts as a "long" masked region (e.g., 30 MHz)
        threshold_length = self.threshold_MHz / freq_resolution
        for i_time in np.arange(calibrated_data_ant.shape[0]):
            mask_check = calibrated_data_ant.mask[i_time, :].copy()
            mask_check[mask_freq] = False
            # find continuous masked regions
            continuous_mask_regions = find_continuous_mask_regions(mask_check)
            # Check if any continuous region exceeds threshold
            if any(
                len(region) >= threshold_length for region in continuous_mask_regions
            ):
                # Mask entire timestamp for this antenna
                calibrated_data_ant.mask[i_time, :] = True

        ###   removing time average
        calibrated_data_ant = calibrated_data_ant - np.ma.mean(
            calibrated_data_ant, axis=0, keepdims=True
        )
        # masking the antennas which are heavily masked (more than 60% is masked)
        if np.mean(calibrated_data_ant.mask) >= 0.6:
            calibrated_data_ant.mask = True

        calibrated_data_ant_interp = np.ma.ones_like(calibrated_data_ant)
        # Calculate the ABBA T[n] + T[n+1] - T[n-1] - T[n+2] for weight estimation
        delta = (
            calibrated_data_ant[:, 1:-2]
            + calibrated_data_ant[:, 2:-1]
            - calibrated_data_ant[:, :-3]
            - calibrated_data_ant[:, 3:]
        ) / 2.0
        weight_diff = 1.0 / np.ma.std(delta, axis=1)
        weight_diff = weight_diff.filled(0)

        # inpainting
        delta_sigma_ant = np.ma.std(delta, axis=1) / np.sqrt(2)

        if calibrated_data_ant.mask.all():
            pass
        else:
            # define the sigma of estimated noise for each antenna
            delta_sigma_ant = delta_sigma_ant.filled(np.ma.median(delta_sigma_ant))
            for i_time in np.arange(calibrated_data_ant.shape[0]):
                if calibrated_data_ant[i_time, :].mask.all():
                    pass
                else:
                    mask = calibrated_data_ant[i_time, :].mask
                    # inpainting the masked regions by fitting a polynomial using +-inpainting_window of the unmasked data around the masked regions
                    calibrated_data_ant_interp.data[i_time, :] = (
                        fill_masked_regions_polyfit(
                            freq_select / 10**6,
                            calibrated_data_ant[i_time, :],
                            polyfit_window=self.inpainting_window,
                            polydeg=self.inpainting_polydeg,
                        )
                    )
                    calibrated_data_ant_interp.mask[i_time, :] = np.zeros(
                        len(freq_select), dtype="bool"
                    )
                    n_mask = np.count_nonzero(mask)
                    # filling the white noise onto the polynomial
                    if n_mask > 0:
                        noise = np.random.default_rng().normal(
                            0.0, delta_sigma_ant[i_time], n_mask
                        )
                        calibrated_data_ant_interp.data[i_time, mask] += noise

        # apply the known RFI masking
        calibrated_data_ant_interp.mask[:, mask_freq] = True

        # in case there are some misfitting points, mask them
        lo = np.ma.min(calibrated_data_ant) - 0.5
        hi = np.ma.max(calibrated_data_ant) + 0.5
        misfit = (calibrated_data_ant_interp <= lo) | (calibrated_data_ant_interp >= hi)
        misfit_time = np.sum(misfit, axis=1) > 0
        calibrated_data_ant_interp.mask[misfit_time, :] = True

        ##################  project the data into maps #####################
        hit_data = np.zeros((map_shape[0], map_shape[1], len(freq_select)))
        hit_data_nomask = np.zeros((map_shape[0], map_shape[1], len(freq_select)))

        map_making_data_ant = np.zeros((map_shape[0], map_shape[1], len(freq_select)))
        weight_data_ant = np.zeros((map_shape[0], map_shape[1], len(freq_select)))

        sky_sc = ac.SkyCoord(
            ra=ra_ant.flatten() * u.deg, dec=dec_ant.flatten() * u.deg
        )  # pointings in observation
        pix_coords = skycoord_to_pixel(sky_sc, wcs_map)

        for i_freq, _ in enumerate(freq_select):
            data = calibrated_data_ant_interp[:, i_freq]
            output, weight, hit_nomask, xedges, yedges = project_2d(
                pix_coords[1],
                pix_coords[0],
                data.flatten(),
                map_shape,
                weights=weight_diff.flatten(),
            )
            _, _, hit, _, _ = project_2d(
                pix_coords[1],
                pix_coords[0],
                calibrated_data_ant[:, i_freq].flatten(),
                map_shape,
                weights=weight_diff.flatten(),
            )

            hit_data[:, :, i_freq] += hit
            hit_data_nomask[:, :, i_freq] += hit_nomask
            map_making_data_ant[:, :, i_freq] = output
            weight_data_ant[:, :, i_freq] = weight

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            map_making_data_ant = map_making_data_ant / weight_data_ant
        map_making_data_ant = np.ma.masked_array(
            map_making_data_ant, mask=~np.isfinite(map_making_data_ant)
        )

        del weight_data_ant, calibrated_data_ant, calibrated_data_ant_interp
        gc.collect()

        return map_making_data_ant, hit_data, hit_data_nomask, mask_freq, wcs_map

    def gather_and_set_result(
        self,
        result_list: list[np.ndarray],
        scan_data: TimeOrderedData,
        calibrated_data: np.ma.MaskedArray,
        freq_select: np.ndarray,
        output_path: str,
        block_name: str,
        rfi_list: list[tuple[float, float]],
    ):
        """
        Combine the `np.ma.MaskedArray`s in `result_list` into a new data set, and mask the frequencies that flag fraction is high (taking all antennas into consideration)
        :param result_list: `list` of `np.ndarray`s created from the RFI flagging
        :param scan_data: time ordered data containing the scanning part of the observation
        :param calibrated_data: calibrated data containing the scanning part of the observation
        :param freq_select: frequency for calibrated data, in [Hz]
        :param output_path: path to store results
        :param block_name: name of the observation block
        :param rfi_list: list of known rfi
        """

        map_making_data = []
        hit_data = []
        hit_data_nomask = []

        for i in range(len(result_list)):
            map_making_data.append(result_list[i][0])
            hit_data.append(result_list[i][1])
            hit_data_nomask.append(result_list[i][2])
            mask_freq = result_list[i][3]
            wcs_map = result_list[i][4]
            result_list[i] = None  # free memory progressively

        map_making_data = np.ma.masked_array(map_making_data)
        hit_data = np.sum(hit_data, axis=0)
        hit_data_nomask = np.sum(hit_data_nomask, axis=0)

        # Calculate the expression T[n] + T[n+1] - T[n-1] - T[n+2] for valid indices
        delta = (
            map_making_data[:, :, :, 1:-2]
            + map_making_data[:, :, :, 2:-1]
            - map_making_data[:, :, :, :-3]
            - map_making_data[:, :, :, 3:]
        ) / 2.0
        weight = 1.0 / np.ma.std(delta, axis=-1)
        weight_ex = np.repeat(
            weight[:, :, :, np.newaxis], map_making_data.shape[-1], axis=3
        )
        map_antennacombine = np.ma.average(map_making_data, axis=0, weights=weight_ex)
        hit_data[~np.isfinite(map_antennacombine)] = 0.0
        del weight_ex, weight
        gc.collect()

        # masking the pixels where <=self.mask_antnum_threshold antennas contributes
        mask_antennacombine_sum = np.sum(~map_making_data.mask, axis=0)
        mask_antennacombine_sum_freqmedian = np.median(mask_antennacombine_sum, axis=-1)
        map_antennacombine.mask[
            mask_antennacombine_sum_freqmedian <= self.mask_antnum_threshold
        ] = True
        hit_data[mask_antennacombine_sum_freqmedian <= self.mask_antnum_threshold] = 0

        #  iteratively running the polynomial_flag to flag the outliers
        freq_mhz = freq_select / 10.0**6
        mask_polyfit = np.ones(map_antennacombine.shape, dtype="bool")
        for pixel_i in range(map_antennacombine.shape[0]):
            for pixel_j in range(map_antennacombine.shape[1]):
                if map_antennacombine[pixel_i, pixel_j].mask.all():
                    pass
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.exceptions.RankWarning)

                        polyfit_schedule = [
                            (18, 6),
                            (10, 6),
                            (8, 6.5),
                            (6, 6.5),
                            (6, 6.5),
                            (6, 6.5),
                            (6, 6.5),
                        ]
                        initial_mask = map_antennacombine[pixel_i, pixel_j].mask
                        for degree, threshold in polyfit_schedule:
                            initial_mask, _ = polynomial_flag_outlier(
                                freq_mhz,
                                map_antennacombine.data[pixel_i, pixel_j],
                                mask=initial_mask,
                                degree=degree,
                                threshold=threshold,
                            )

                    ###  mask dilation
                    struct = np.ones(3, dtype=bool)
                    initial_mask = ndimage.binary_dilation(
                        initial_mask, structure=struct, iterations=2
                    )

                    ###    update the mask
                    mask_polyfit[pixel_i, pixel_j] = initial_mask

        map_antennacombine = np.ma.masked_array(
            map_antennacombine.data, mask=mask_polyfit
        )
        hit_data[mask_polyfit] = 0.0

        ################## do second inpainting to fill the regions masked by polynomial_flag
        map_antennacombine_interp = np.ma.ones_like(map_antennacombine)

        # Calculate the expression T[n] + T[n+1] - T[n-1] - T[n+2] for valid indices
        delta = (
            map_antennacombine[:, :, 1:-2]
            + map_antennacombine[:, :, 2:-1]
            - map_antennacombine[:, :, :-3]
            - map_antennacombine[:, :, 3:]
        ) / 2.0

        # inpainting
        delta_sigma = np.ma.std(delta, axis=2) / np.sqrt(2)

        for pixel_i in range(map_antennacombine.shape[0]):
            for pixel_j in range(map_antennacombine.shape[1]):
                if map_antennacombine[pixel_i, pixel_j].mask.all():
                    pass
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.exceptions.RankWarning)
                        delta_sigma_pixel = delta_sigma[pixel_i, pixel_j]

                        mask = map_antennacombine[pixel_i, pixel_j].mask
                        map_antennacombine_interp.data[pixel_i, pixel_j] = (
                            fill_masked_regions_polyfit(
                                freq_mhz,
                                map_antennacombine[pixel_i, pixel_j],
                                polyfit_window=self.inpainting_window,
                                polydeg=self.inpainting_polydeg,
                            )
                        )
                        map_antennacombine_interp.mask[pixel_i, pixel_j] = np.zeros(
                            len(freq_mhz), dtype="bool"
                        )
                        n_mask = np.count_nonzero(mask)
                        ###  add white noise to masked regions
                        if n_mask > 0:
                            noise = np.random.default_rng().normal(
                                0.0, delta_sigma_pixel, n_mask
                            )
                            map_antennacombine_interp.data[pixel_i, pixel_j, mask] += (
                                noise
                            )

        ##########  mask known RFI
        map_antennacombine_interp.mask[:, :, mask_freq] = True

        # in case there are some misfitting points, mask them
        lo = np.ma.min(map_antennacombine) - 0.5
        hi = np.ma.max(map_antennacombine) + 0.5
        misfit = (map_antennacombine_interp <= lo) | (map_antennacombine_interp >= hi)

        misfit_time = np.sum(misfit, axis=2) > 0
        map_antennacombine_interp.mask[misfit_time] = True
        hit_data[misfit_time] = 0.0

        #########   save results
        if self.do_store_context:
            arrays_dict = {
                "map": map_antennacombine_interp,
                "hit": hit_data,
                "hit_nomask": hit_data_nomask,
                "wcs": wcs_map,
                "freq": freq_mhz,
                "antenna_list": scan_data._antenna_name_list,
            }

            with open(
                output_path + "/map_making_" + block_name + "_inpainting.pkl", "wb"
            ) as f:
                pickle.dump(arrays_dict, f)
