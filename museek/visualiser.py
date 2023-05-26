import numpy as np
from matplotlib import pyplot as plt

from museek.data_element import DataElement
from museek.flag_list import FlagList
from museek.time_ordered_data_mapper import TimeOrderedDataMapper


def plot_time_ordered_data_map(right_ascension: DataElement,
                               declination: DataElement,
                               visibility: DataElement,
                               flags: FlagList | None = None,
                               flag_threshold: int = 1,
                               grid_size: tuple[int, int] = (60, 60),
                               cmap: str = 'jet',
                               norm: str = 'linear',
                               vmin: float | None = None,
                               vmax: float | None = None,
                               do_mask: bool = True,
                               interpolation_method: str = 'linear'):
    """
    Function to plot the gridded and masked visibility data.
    :param right_ascension: coordinate
    :param declination: coordinate
    :param visibility: visibility data to grid and plot
    :param flags: flags to mask the visibility
    :param flag_threshold: flags are ignored if they sum to less than this value
    :param grid_size: size of the grid
    :param cmap: color map for `imshow`
    :param norm: color normalisation for `imshow`
    :param vmin: color mapping minimum value for `imshow`
    :param vmax: color mapping maximum value for `imshow`
    :param do_mask: wether or not to mask the flagged pixels in the map
    :param interpolation_method: passed on to `grid()`
    """
    maps, masks = TimeOrderedDataMapper(right_ascension=right_ascension,
                                        declination=declination,
                                        to_map=visibility,
                                        flags=flags,
                                        flag_threshold=flag_threshold).grid(grid_size=grid_size,
                                                                            method=interpolation_method)
    if (n_channels := len(maps)) > 1:
        print(f'Only one channel can be plotted at a time, got {n_channels}.')
        return
    if maps[0] is None:
        print('Flag covers the entire visibility data.')
        maps[0] = np.zeros(grid_size)
        do_mask = True
    if do_mask:
        maps = [np.ma.array(map_, mask=mask) for map_, mask in zip(maps, masks)]

    plt.imshow(maps[0][::-1, :],
               extent=(right_ascension.squeeze.min(),
                       right_ascension.squeeze.max(),
                       declination.squeeze.min(),
                       declination.squeeze.max()),
               cmap=plt.get_cmap(cmap, 255),
               aspect='auto',
               norm=norm,
               vmin=vmin,
               vmax=vmax)
    plt.colorbar()
    plt.xlim(right_ascension.squeeze.min(), right_ascension.squeeze.max())
    plt.ylim(declination.squeeze.min(), declination.squeeze.max())


def waterfall(visibility: DataElement, flags: FlagList | None, flag_threshold: int = 1, **imshow_kwargs):
    """
    Function to create the waterfall plot.
    :param visibility: visibility data to be plotted
    :param flags: optional boolean flags
    :param flag_threshold: flags are only used if they overlap more than this value
    :param imshow_kwargs: keyword arguments for `plt.imshow()`
    """
    if flags:
        all_flags = flags.combine(threshold=flag_threshold).squeeze
    else:
        all_flags = None
    masked = np.ma.array(visibility.squeeze, mask=all_flags)
    image = plt.imshow(masked.T, aspect='auto', **imshow_kwargs)
    plt.colorbar(image)
