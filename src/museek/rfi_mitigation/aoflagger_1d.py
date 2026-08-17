from copy import copy

import numpy as np

from museek.data_element import DataElement
from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_element import FlagElement

"""
A collection of functions for RFI flagging using the AOflagger algorithm.
"""


def get_rfi_mask_1d(
    time_ordered: DataElement,
    mask: FlagElement,
    mask_type: str,
    first_threshold: float,
    threshold_scales: list[float],
    smoothing_window_size: int,
    smoothing_sigma: float,
    output_path: str | None = None,
) -> FlagElement:
    """
    Computes a mask to cover the RFI in a 1-dimensional data set.

    :param time_ordered: `DataElement` with RFI to be masked
    :param mask: the initial mask
    :param mask_type: the data to which the flagger will be applied
    :param first_threshold: initial threshold to be used for the aoflagger algorithm
    :param threshold_scales: list of sensitivities
    :param smoothing_window_size: smoothing kernel window size
    :param smoothing_sigma: smoothing kernel sigma
    :param output_path: if not `None`, statistics plots are stored at that location
    :return: the mask covering the identified RFI
    """
    if mask_type == "vis":
        data = time_ordered.squeeze

    elif mask_type == "inverse":
        data = (1.0 / np.ma.masked_array(time_ordered.squeeze, mask=mask.squeeze)).data

    else:
        raise ValueError(f"Unknown mask_type {mask_type}")

    max_pixels = 8  # Maximum neighbourhood size
    pixel_arange = np.arange(1, max_pixels)
    scaling_base = 1.5
    n_iterations = 2 ** (pixel_arange - 1)
    thresholds = first_threshold / scaling_base ** np.log2(pixel_arange)

    sum_threshold_mask = mask.squeeze
    for threshold_scale in threshold_scales:
        sum_threshold_mask = _run_sumthreshold_1d(
            data=data,
            initial_mask=sum_threshold_mask,
            threshold_scale=threshold_scale,
            n_iterations=n_iterations,
            thresholds=thresholds,
            output_path=output_path,
            smoothing_window_size=smoothing_window_size,
            smoothing_sigma=smoothing_sigma,
        )

    return FlagElementFactory().create(
        array=sum_threshold_mask[:, np.newaxis, np.newaxis]
    )


def gaussian_filter_1d(
    array: np.ndarray, mask: np.ndarray, window_size: int = 40, sigma: float = 0.5
) -> np.ndarray[float | None]:
    """
    Apply a gaussian filter (smoothing) to the given positive definite array taking into account masked values,
    any result entries that are zero are replaced by `None`
    :param array: the array to be smoothed
    :param mask: boolean array defining masked values
    :param window_size: kernel window size
    :param sigma: kernel sigma tuple
    :return: filtered array with entries >0 or `None`
    """

    def exponential_window_1d(x, sigma_x):
        return np.exp(-(x**2) / (2 * sigma_x**2))

    window_ranges = np.arange(-window_size / 2, window_size / 2 + 1)
    kernel = exponential_window_1d(x=window_ranges, sigma_x=sigma)

    result = _apply_kernel_1d(
        array=array, mask=mask, kernel=kernel, even_window_size=window_size
    )
    result[result == 0] = None
    return result


def _apply_kernel_1d(
    array: np.ndarray, mask: np.ndarray, kernel: np.ndarray, even_window_size: int
) -> np.ndarray:
    """
    Apply smoothing with `kernel` to `array` taking into account values masked by `mask` and return the result.
    :param array: `numpy` `array` to be smoothed
    :param mask: boolean mask `array`
    :param kernel: `np.ndarray` defining the smoothing kernel
    :param even_window_size: smoothing window size, must be divisible by 2
    :return: smoothed array
    """
    array_larger = np.zeros(array.shape[0] + even_window_size)
    array_larger[even_window_size // 2 : -even_window_size // 2] = array[:]

    mask_larger = np.zeros_like(array_larger)
    mask_larger[even_window_size // 2 : -even_window_size // 2] = ~mask[:]

    window_size_half = even_window_size // 2

    tmp_array = np.zeros_like(array_larger)
    result = np.zeros_like(array_larger)
    for i in range(window_size_half, array.shape[0] + window_size_half):
        if mask[i - window_size_half]:
            tmp_array[i] = 0
        else:
            value = np.sum(
                mask_larger[i - window_size_half : i + window_size_half + 1]
                * array_larger[i - window_size_half : i + window_size_half + 1]
                * kernel
            )
            tmp_array[i] = value / np.sum(
                mask_larger[i - window_size_half : i + window_size_half + 1] * kernel
            )

    # remove window on either end
    result = tmp_array[window_size_half:-window_size_half]
    result[mask] = array[mask]
    return result


def _run_sumthreshold_1d(
    data: np.ndarray,
    initial_mask: np.ndarray,
    threshold_scale: float,
    n_iterations: list[int],
    thresholds: list[float],
    smoothing_window_size: int,
    smoothing_sigma: float,
    output_path: str | None = None,
) -> np.ndarray:
    """
    Perform one SumThreshold operation: sum the un-masked data after
    subtracting a smooth background and threshold it.
    :param data: visibility data
    :param initial_mask: initial mask, already obtained from sum thresholding
    :param threshold_scale: number that scales the threshold value for each iteration
    :param n_iterations: number of iterations
    :param thresholds: thresholding criteria
    :param smoothing_window_size: smoothing kernel window size
    :param smoothing_sigma: smoothing kernel sigma
    :param output_path: if not `None`, statistics plots are stored at that location
    :return: SumThreshold mask
    """

    smoothed_data = gaussian_filter_1d(
        data, initial_mask, window_size=smoothing_window_size, sigma=smoothing_sigma
    )
    residual = (data - smoothed_data) / smoothed_data

    sum_threshold_mask = initial_mask.copy()

    for n_iteration, threshold in zip(n_iterations, thresholds):
        threshold = threshold / threshold_scale
        if n_iteration == 1:
            sum_threshold_mask = sum_threshold_mask | (threshold <= residual)
        else:
            sum_threshold_mask = _sum_threshold_mask_1d(
                data=residual,
                mask=sum_threshold_mask,
                n_iteration=n_iteration,
                threshold=threshold,
            )

    return sum_threshold_mask


def _sum_threshold_mask_1d(
    data: np.ndarray, mask: np.ndarray[bool], n_iteration: int, threshold: float
) -> np.ndarray[bool]:
    """
    Return the boolean mask obtained from summing and thresholding.
    :param data: visibility data
    :param mask: mask of visibility data
    :param n_iteration: number of iterations
    :param threshold: initial threshold
    :return: boolean `numpy` array
    """
    result = copy(mask)
    sum_ = 0
    count = 0

    max_index_axis = min(n_iteration, data.shape[0])
    indices_to_sum = np.where(np.logical_not(mask[:max_index_axis]))[0]
    sum_ += np.sum(data[indices_to_sum])
    count += len(indices_to_sum)

    for index_axis in range(n_iteration, data.shape[0]):
        if sum_ > threshold * count:
            result[index_axis - n_iteration : index_axis - 1] = True

        if not mask[index_axis]:
            sum_ += data[index_axis]
            count += 1

        if not mask[index_axis - n_iteration]:
            sum_ -= data[index_axis - n_iteration]
            count -= 1
    return result
