import numpy as np
from matplotlib import pyplot as plt

from museek.data_element import DataElement
from museek.flag_element import FlagElement
from museek.time_ordered_data_mapper import TimeOrderedDataMapper


def plot_time_ordered_data_map(right_ascension: DataElement,
                               declination: DataElement,
                               visibility: DataElement,
                               flags: FlagElement | None = None,
                               grid_size: tuple[int, int] = (60, 60),
                               cmap: str = 'jet',
                               norm: str = 'linear',
                               vmin: float | None = None,
                               vmax: float | None = None):
    """
    Function to plot the gridded and masked visibility data.
    :param right_ascension: coordinate
    :param declination: coordinate
    :param visibility: visibility data to grid and plot
    :param flags: flags to mask the visibility
    :param grid_size: size of the grid
    :param cmap: color map for `imshow`
    :param norm: color normalisation for `imshow`
    :param vmin: color mapping minimum value for `imshow`
    :param vmax: color mapping maximum value for `imshow`
    """
    maps, masks = TimeOrderedDataMapper(right_ascension=right_ascension,
                                        declination=declination,
                                        to_map=visibility,
                                        flags=flags).grid(grid_size=grid_size)
    masked_maps = [np.ma.array(map_, mask=mask) for map_, mask in zip(maps, masks)]

    plt.imshow(masked_maps[0][::-1, :],
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


def waterfall(visibility: DataElement, flags: FlagElement | None, flag_threshold: int = 1, **imshow_kwargs):
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
