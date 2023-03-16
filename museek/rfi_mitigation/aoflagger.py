import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from museek.data_element import DataElement


def plot_data(data, ax, title, vmin=None, vmax=None, cb=True, norm=None, extent=None, cmap=None):
    """
    Plot TOD.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax.set_title(title)
    im = ax.imshow(data,
                   aspect="auto",
                   origin="lower",
                   norm=norm,
                   extent=extent,
                   cmap=cmap,
                   interpolation="nearest", vmin=vmin, vmax=vmax)

    if cb:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="20%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)


def plot_steps(data, st_mask, smoothed_data, res, eta):
    """
    Plot individual steps of SumThreshold.
    """
    f, ax = plt.subplots(2, 2, figsize=(15, 8))
    f.suptitle("Eta: %s" % eta)
    plot_data(data, ax[0, 0], "data")
    plot_data(st_mask, ax[1, 0], "mask (%s)" % (st_mask.sum()), 0, 1)
    smoothed = np.ma.MaskedArray(smoothed_data, st_mask)
    plot_data(smoothed, ax[0, 1], "_smooth")
    plot_data(res, ax[1, 1], "residuals")
    f.show()


# @numba.jit
def _sum_threshold_mask(
        data: np.ndarray,
        mask: np.ndarray[bool],
        n_iteration: int,
        threshold: float
) -> np.ndarray[bool]:
    """
    Returns the boolean mask obtained from summing and thresholding.
    :param data: visibility data
    :param mask: mask of visibility data
    :param n_iteration: number of iterations
    :param threshold: thresholding criteria
    :return: boolean `numpy` array
    """
    # result = copy(mask)
    result = mask[:]
    for index_axis_0 in range(data.shape[0]):
        sum_ = 0
        count = 0

        max_index_axis_1 = min(n_iteration, data.shape[1])
        indices_to_sum = np.where(np.logical_not(mask[index_axis_0, :max_index_axis_1]))[0]
        sum_ += np.sum(data[index_axis_0, indices_to_sum])
        count += len(indices_to_sum)

        # for index_axis_1 in range(min(n_iteration, data.shape[1])):
        #     if not mask[index_axis_0, index_axis_1]:
        #         sum_ += data[index_axis_0, index_axis_1]
        #         count += 1

        for index_axis_1 in range(n_iteration, data.shape[1]):
            if sum_ > threshold * count:
                # for frequency_index_subtractor in range(n_iteration):
                #     result[index_axis_0, index_axis_1 - frequency_index_subtractor - 1] = True
                result[index_axis_0, index_axis_1 - n_iteration:index_axis_1 - 1] = True

            if not mask[index_axis_0, index_axis_1]:
                sum_ += data[index_axis_0, index_axis_1]
                count += 1

            if not mask[index_axis_0, index_axis_1 - n_iteration]:
                sum_ -= data[index_axis_0, index_axis_1 - n_iteration]
                count -= 1
    return result


def _gaussian_filter(Vp, vs0, vs1, Wfp, mask, Vh, Vh2, kernel_0, kernel_1, M, N):
    n2 = N // 2
    m2 = M // 2
    for i in range((N // 2), vs0 + (N // 2)):
        for j in range((M // 2), vs1 + (M // 2)):
            if mask[i - n2, j - m2]:
                Vh[i, j] = 0  # V[i-n2, j-m2]
            else:
                val = np.sum((Wfp[i - n2:i + n2 + 1, j] * Vp[i - n2:i + n2 + 1, j] * kernel_0))
                Vh[i, j] = val / np.sum(Wfp[i - n2:i + n2 + 1, j] * kernel_0)

    for j2 in range((M // 2), vs1 + (M // 2)):
        for i2 in range((N // 2), vs0 + (N // 2)):
            if mask[i2 - n2, j2 - m2]:
                Vh2[i2, j2] = 0  # V[i2-n2, j2-m2]
            else:
                val = np.sum((Wfp[i2, j2 - m2:j2 + m2 + 1] * Vh[i2, j2 - m2:j2 + m2 + 1] * kernel_1))
                Vh2[i2, j2] = val / np.sum(Wfp[i2, j2 - m2:j2 + m2 + 1] * kernel_1)
    return Vh2


def gaussian_filter(V, mask, M=40, N=20, sigma_m=0.5, sigma_n=0.5):
    """
    Applies a gaussian filter (smoothing) to the given array taking into account masked values
    :param V: the value array to be smoothed
    :param mask: boolean array defining masked values
    :param M: kernel window size in axis=1
    :param N: kernel window size in axis=0
    :param sigma_m: kernel sigma in axis=1
    :param sigma_n: kernel sigma in axis=0

    :returns vs: the filtered array
    """

    def wd(n, m, sigma_n, sigma_m):
        return np.exp(-n ** 2 / (2 * sigma_n ** 2) - m ** 2 / (2 * sigma_m ** 2))

    Vp = np.zeros((V.shape[0] + N, V.shape[1] + M))
    Vp[N // 2:-N // 2, M // 2:-M // 2] = V[:]

    Wfp = np.zeros((V.shape[0] + N, V.shape[1] + M))
    Wfp[N // 2:-N // 2, M // 2:-M // 2] = ~mask[:]
    Vh = np.zeros((V.shape[0] + N, V.shape[1] + M))
    Vh2 = np.zeros((V.shape[0] + N, V.shape[1] + M))

    n = np.arange(-N / 2, N / 2 + 1)
    m = np.arange(-M / 2, M / 2 + 1)
    kernel_0 = wd(n, 0, sigma_n=sigma_n, sigma_m=sigma_m).T
    kernel_1 = wd(0, m, sigma_n=sigma_n, sigma_m=sigma_m).T

    Vh = _gaussian_filter(Vp, V.shape[0], V.shape[1], Wfp, mask, Vh, Vh2, kernel_0, kernel_1, M, N)
    Vh = Vh[N // 2:-N // 2, M // 2:-M // 2]
    Vh[mask] = V[mask]
    return Vh


def _run_sumthreshold(data, initial_mask, threshold_scale, n_iterations, thresholds, smoothing_kwargs, do_plot=True):
    """
    Perform one SumThreshold operation: sum the un-masked data after
    subtracting a smooth background and threshold it.

    :param data: data
    :param initial_mask: initial mask, already obtained from sum thresholding
    :param threshold_scale: number that scales the threshold value for each iteration
    :param n_iterations: number of iterations
    :param thresholds: thresholding criteria
    :param smoothing_kwargs: smoothing keyword
    :param do_plot: whether to plot

    :return: SumThreshold mask

    """

    smoothed_data = gaussian_filter(data, initial_mask, **smoothing_kwargs)
    residual = data - smoothed_data

    sum_threshold_mask = initial_mask.copy()

    for n_iteration, threshold in zip(n_iterations, thresholds):
        threshold = threshold / threshold_scale
        if n_iteration == 1:
            sum_threshold_mask = sum_threshold_mask | (threshold <= residual)
        else:
            sum_threshold_mask = _sum_threshold_mask(residual, sum_threshold_mask, n_iteration, threshold)
            sum_threshold_mask = _sum_threshold_mask(residual.T, sum_threshold_mask.T, n_iteration, threshold).T

    if do_plot:
        plot_steps(data, sum_threshold_mask, smoothed_data, residual, "%s (%s)" % (threshold_scale, thresholds))

    return sum_threshold_mask


def binary_mask_dilation(mask, struct_size_0, struct_size_1):
    """
    Dilates the mask.

    :param mask: original mask
    :param struct_size_0: dilation parameter
    :param struct_size_1: dilation parameter

    :return: dilated mask
    """
    struct = np.ones((struct_size_0, struct_size_1), np.bool)
    return ndimage.binary_dilation(mask, structure=struct, iterations=2)


def normalize(data, mask):
    """
    Simple normalization of standing waves: subtracting the median over time
    for each frequency.

    :param data: data
    :param mask: mask

    :return: normalized data
    """
    median = np.ma.median(np.ma.MaskedArray(data, mask), axis=1).reshape(data.shape[0], -1)
    data = np.abs(data - median)
    return data.data


def plot_moments(data):
    """
    Plot standard divation and mean of data.
    """

    std_time = np.std(data, axis=0)
    mean_time = np.mean(data, axis=0)
    std_freuqency = np.std(data, axis=1)
    mean_freuqency = np.mean(data, axis=1)
    plt.subplot(121)
    plt.plot(mean_time)
    plt.xlabel("time")
    plt.ylabel("mean")
    plt.subplot(122)
    plt.plot(std_time)
    plt.xlabel("time")
    plt.ylabel("std")
    plt.show()
    plt.subplot(121)
    plt.plot(mean_freuqency)
    plt.xlabel("freuqency")
    plt.ylabel("mean")
    plt.subplot(122)
    plt.plot(std_freuqency)
    plt.xlabel("freuqency")
    plt.ylabel("std")
    plt.tight_layout()
    plt.show()


def plot_dilation(st_mask, mask, dilated_mask):
    """
    Plot mask and dilation.
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle("Mask analysis")
    plot_data(mask, ax[0, 0], "Original mask")
    plot_data(st_mask.astype(np.bool) - mask, ax[0, 1], "Sum threshold mask", 0, 1)
    plot_data(dilated_mask, ax[1, 0], "dilated mask")
    plot_data(dilated_mask + mask, ax[1, 1], "New mask")
    fig.show()


def get_rfi_mask(tod: DataElement,
                 mask: DataElement,
                 first_threshold,
                 threshold_scales,
                 normalize_standing_waves=True,
                 suppress_dilation=False,
                 do_plot: bool = True,
                 smoothing_kwargs=None,
                 dilation_kwargs=None):
    """
    Computes a mask to cover the RFI in a data set.

    :param tod:
    :param mask: the initial mask
    :param first_threshold:
    :param threshold_scales: List of sensitivities
    :param normalize_standing_waves: whether to normalize standing waves
    :param suppress_dilation: if true, mask dilation is suppressed
    :param do_plot: True if statistics plot should be displayed
    :param smoothing_kwargs: smoothing key words
    :param dilation_kwargs: dilation key words

    :return mask: the mask covering the identified RFI
    """
    data = tod.squeeze

    if do_plot:
        plot_moments(data)

    if normalize_standing_waves:
        data = normalize(data, mask.squeeze)

        if do_plot:
            plot_moments(data)

    # Maximum neighbourhood size
    max_pixels = 8
    p = 1.5
    m = np.arange(1, max_pixels)
    n_iterations = 2 ** (m - 1)
    thresholds = first_threshold / p ** np.log2(m)

    sum_threshold_mask = mask.squeeze  # TODO: investigate wether or not it is better to start from scratch
    for threshold_scale in threshold_scales:
        sum_threshold_mask = _run_sumthreshold(data=data,
                                               initial_mask=sum_threshold_mask,
                                               threshold_scale=threshold_scale,
                                               n_iterations=n_iterations,
                                               thresholds=thresholds,
                                               smoothing_kwargs=smoothing_kwargs,
                                               do_plot=do_plot)

    if not suppress_dilation:
        sum_threshold_mask = binary_mask_dilation(sum_threshold_mask.astype(np.float) - mask.squeeze.astype(np.float),
                                                  **dilation_kwargs)

        if do_plot: plot_dilation(sum_threshold_mask, mask, sum_threshold_mask)

    return DataElement(sum_threshold_mask[:, :, np.newaxis])
