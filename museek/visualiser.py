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
               extent=(right_ascension.min(), right_ascension.max(), declination.min(), declination.max()),
               cmap=plt.get_cmap(cmap, 255),
               aspect='auto',
               norm=norm,
               vmin=vmin,
               vmax=vmax)
    plt.colorbar()
    plt.xlim(right_ascension.min(), right_ascension.max())
    plt.ylim(declination.min(), declination.max())
