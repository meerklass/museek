import os
from copy import copy

import numpy as np
from matplotlib import pyplot as plt

from museek.data_element import DataElement
from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_element import FlagElement

"""
A collection of functions for RFI flagging using the AOflagger algorithm.
"""


def get_rfi_mask(
        time_ordered: DataElement,
        mask: FlagElement,
        mask_type: str,
        first_threshold: float,
        threshold_scales: list[float],
        smoothing_window_size: tuple[int, int],
        smoothing_sigma: tuple[float, float],
        output_path: str | None = None
) -> FlagElement:
    """
    Computes a mask to cover the RFI in a data set.

    :param time_ordered: `DataElement` with RFI to be masked
    :param mask: the initial mask
    :param mask_type: the data to which the flagger will be applied
    :param first_threshold: initial threshold to be used for the aoflagger algorithm
    :param threshold_scales: list of sensitivities
    :param smoothing_window_size: smoothing kernel window size tuple for axes 0 and 1
    :param smoothing_sigma: smoothing kernel sigma tuple for axes 0 and 1
    :param output_path: if not `None`, statistics plots are stored at that location
    :return: the mask covering the identified RFI
    """
    if mask_type == 'vis':
        data = time_ordered.squeeze
    elif mask_type == 'flag_fraction':
        flag_fraction = np.mean(mask.squeeze, axis=0)
        data = np.tile(flag_fraction, (np.shape(mask.squeeze)[0], 1))
    else:
        raise ValueError("Unknown mask_type {}".format(mask_type))

    if output_path is not None:
        plot_moments(data, output_path)

    max_pixels = 8  # Maximum neighbourhood size
    pixel_arange = np.arange(1, max_pixels)
    scaling_base = 1.5
    n_iterations = 2 ** (pixel_arange - 1)
    thresholds = first_threshold / scaling_base ** np.log2(pixel_arange)

    sum_threshold_mask = mask.squeeze
    for threshold_scale in threshold_scales:
        sum_threshold_mask = _run_sumthreshold(data=data,
                                               initial_mask=sum_threshold_mask,
                                               threshold_scale=threshold_scale,
                                               n_iterations=n_iterations,
                                               thresholds=thresholds,
                                               output_path=output_path,
                                               smoothing_window_size=smoothing_window_size,
                                               smoothing_sigma=smoothing_sigma)

    return FlagElementFactory().create(array=sum_threshold_mask[:, :, np.newaxis])


def gaussian_filter(array: np.ndarray,
                    mask: np.ndarray,
                    window_size: tuple[int, int] = (20, 40),
                    sigma: tuple[float, float] = (0.5, 0.5)) -> np.ndarray[float | None]:
    """
    Apply a gaussian filter (smoothing) to the given positive definite array taking into account masked values,
    any result entries that are zero are replaced by `None`
    :param array: the array to be smoothed
    :param mask: boolean array defining masked values
    :param window_size: kernel window size tuple for axes 0 and 1
    :param sigma: kernel sigma tuple for axes 0 and 1
    :return: filtered array with entries >0 or `None`
    """

    def exponential_window(x, y, sigma_x, sigma_y):
        return np.exp(-x ** 2 / (2 * sigma_x ** 2) - y ** 2 / (2 * sigma_y ** 2))

    window_ranges = [np.arange(-size / 2, size / 2 + 1) for size in window_size]
    kernel_0 = exponential_window(x=window_ranges[0], y=0, sigma_x=sigma[0], sigma_y=sigma[1]).T
    kernel_1 = exponential_window(x=0, y=window_ranges[1], sigma_x=sigma[0], sigma_y=sigma[1]).T

    result = _apply_kernel(array=array,
                           mask=mask,
                           kernel=(kernel_0, kernel_1),
                           even_window_size=window_size)
    result[result == 0] = None
    return result


def _apply_kernel(array: np.ndarray,
                  mask: np.ndarray,
                  kernel: tuple[np.ndarray, np.ndarray],
                  even_window_size: tuple[int, int]) -> np.ndarray:
    """
    Apply smoothing with `kernel` to `array` taking into account values masked by `mask` and return the result.
    :param array: `numpy` `array` to be smoothed
    :param mask: boolean mask `array`
    :param kernel: `tuple` of two `np.ndarray`s defining the smoothing kernel in axes 0 and 1
    :param even_window_size: smoothing window size in axes 0 and 1, must be divisible by 2
    :return: smoothed array
    """
    array_larger = np.zeros((array.shape[0] + even_window_size[0], array.shape[1] + even_window_size[1]))
    array_larger[even_window_size[0] // 2:-even_window_size[0] // 2,
    even_window_size[1] // 2:-even_window_size[1] // 2] = array[:]

    mask_larger = np.zeros_like(array_larger)
    mask_larger[even_window_size[0] // 2:-even_window_size[0] // 2,
    even_window_size[1] // 2:-even_window_size[1] // 2] = ~mask[:]

    window_size_half = [size // 2 for size in even_window_size]

    tmp_array = np.zeros_like(array_larger)
    result = np.zeros_like(array_larger)
    for i in range(window_size_half[0], array.shape[0] + window_size_half[0]):
        for j in range(window_size_half[1], array.shape[1] + window_size_half[1]):
            if mask[i - window_size_half[0], j - window_size_half[1]]:
                tmp_array[i, j] = 0
            else:
                value = np.sum(
                    (mask_larger[i - window_size_half[0]:i + window_size_half[0] + 1, j]
                     * array_larger[i - window_size_half[0]:i + window_size_half[0] + 1, j]
                     * kernel[0])
                )
                tmp_array[i, j] = value / np.sum(
                    mask_larger[i - window_size_half[0]:i + window_size_half[0] + 1, j] * kernel[0]
                )

    for j2 in range(window_size_half[1], array.shape[1] + window_size_half[1]):
        for i2 in range(window_size_half[0], array.shape[0] + window_size_half[0]):
            if mask[i2 - window_size_half[0], j2 - window_size_half[1]]:
                result[i2, j2] = 0
            else:
                value = np.sum(
                    (mask_larger[i2, j2 - window_size_half[1]:j2 + window_size_half[1] + 1]
                     * tmp_array[i2, j2 - window_size_half[1]:j2 + window_size_half[1] + 1]
                     * kernel[1])
                )
                result[i2, j2] = value / np.sum(
                    mask_larger[i2, j2 - window_size_half[1]:j2 + window_size_half[1] + 1] * kernel[1]
                )

    # remove window on either end
    result = result[window_size_half[0]:-window_size_half[0], window_size_half[1]:-window_size_half[1]]
    result[mask] = array[mask]
    return result


def _run_sumthreshold(data: np.ndarray,
                      initial_mask: np.ndarray,
                      threshold_scale: float,
                      n_iterations: list[int],
                      thresholds: list[float],
                      smoothing_window_size: tuple[int, int],
                      smoothing_sigma: tuple[float, float],
                      output_path: str | None = None) \
        -> np.ndarray:
    """
    Perform one SumThreshold operation: sum the un-masked data after
    subtracting a smooth background and threshold it.
    :param data: visibility data
    :param initial_mask: initial mask, already obtained from sum thresholding
    :param threshold_scale: number that scales the threshold value for each iteration
    :param n_iterations: number of iterations
    :param thresholds: thresholding criteria
    :param smoothing_window_size: smoothing kernel window size tuple for axes 0 and 1
    :param smoothing_sigma: smoothing kernel sigma tuple for axes 0 and 1
    :param output_path: if not `None`, statistics plots are stored at that location
    :return: SumThreshold mask
    """

    smoothed_data = gaussian_filter(data, initial_mask, window_size=smoothing_window_size, sigma=smoothing_sigma)
    residual = (data - smoothed_data) / smoothed_data

    sum_threshold_mask = initial_mask.copy()

    for n_iteration, threshold in zip(n_iterations, thresholds):
        threshold = threshold / threshold_scale
        if n_iteration == 1:
            sum_threshold_mask = sum_threshold_mask | (threshold <= residual)
        else:
            sum_threshold_mask = _sum_threshold_mask(data=residual,
                                                     mask=sum_threshold_mask,
                                                     n_iteration=n_iteration,
                                                     threshold=threshold)
            sum_threshold_mask = _sum_threshold_mask(data=residual.T,
                                                     mask=sum_threshold_mask.T,
                                                     n_iteration=n_iteration,
                                                     threshold=threshold).T

    if output_path is not None:
        plot_step(data,
                  sum_threshold_mask,
                  smoothed_data,
                  residual,
                  title=f'Tresholds: {threshold_scale} {thresholds}',
                  plot_name=f'sum_threshold_step_at_threshold_scale_{threshold_scale}.png',
                  output_path=output_path)

    return sum_threshold_mask


def _sum_threshold_mask(
        data: np.ndarray,
        mask: np.ndarray[bool],
        n_iteration: int,
        threshold: float
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
    for index_axis_0 in range(data.shape[0]):
        sum_ = 0
        count = 0

        max_index_axis_1 = min(n_iteration, data.shape[1])
        indices_to_sum = np.where(np.logical_not(mask[index_axis_0, :max_index_axis_1]))[0]
        sum_ += np.sum(data[index_axis_0, indices_to_sum])
        count += len(indices_to_sum)

        for index_axis_1 in range(n_iteration, data.shape[1]):
            if sum_ > threshold * count:
                result[index_axis_0, index_axis_1 - n_iteration:index_axis_1 - 1] = True

            if not mask[index_axis_0, index_axis_1]:
                sum_ += data[index_axis_0, index_axis_1]
                count += 1

            if not mask[index_axis_0, index_axis_1 - n_iteration]:
                sum_ -= data[index_axis_0, index_axis_1 - n_iteration]
                count -= 1
    return result


def plot_moments(data, output_path: str):
    """
    Plot standard divation and mean of data.
    """

    std_time = np.std(data, axis=0)
    mean_time = np.mean(data, axis=0)
    std_freuqency = np.std(data, axis=1)
    mean_freuqency = np.mean(data, axis=1)
    plt.subplot(221)
    plt.plot(mean_time)
    plt.xlabel('time')
    plt.ylabel('mean')
    plt.subplot(222)
    plt.plot(std_time)
    plt.xlabel('time')
    plt.ylabel('std')
    plt.subplot(223)
    plt.plot(mean_freuqency)
    plt.xlabel('freuqency')
    plt.ylabel('mean')
    plt.subplot(224)
    plt.plot(std_freuqency)
    plt.xlabel('freuqency')
    plt.ylabel('std')
    plt.tight_layout()
    plot_name = 'standard_deviation_and_mean.png'
    plt.savefig(os.path.join(output_path, plot_name))
    plt.close()


def plot_data(data, ax, title, vmin=None, vmax=None, cb=True, norm=None, extent=None, cmap=None):
    """
    Plot `data`.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax.set_title(title)
    im = ax.imshow(data,
                   aspect='auto',
                   origin='lower',
                   norm=norm,
                   extent=extent,
                   cmap=cmap,
                   interpolation='nearest', vmin=vmin, vmax=vmax)

    if cb:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='20%', pad=0.05)
        plt.colorbar(im, cax=cax)


def plot_step(data, mask, smoothed_data, residual, title, plot_name: str, output_path: str):
    """
    Plot individual step of SumThreshold.
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(title)
    plot_data(data, ax[0, 0], 'data')
    plot_data(mask, ax[1, 0], f'mask {mask.sum()}', 0, 1)
    smoothed = np.ma.MaskedArray(smoothed_data, mask)
    plot_data(smoothed, ax[0, 1], 'smooth')
    plot_data(residual, ax[1, 1], 'residual')
    fig.savefig(os.path.join(output_path, plot_name))
    plt.close()
