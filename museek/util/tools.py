import numpy as np
from museek.time_ordered_data import TimeOrderedData
import pysm3
import pysm3.units as u
import healpy as hp
from astropy.coordinates import SkyCoord
import matplotlib.colors as colors
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import warnings

def flag_percent_recv(data: TimeOrderedData):
    """
    return the flag percent for each receiver
    """
    flag_percent = []
    receivers_list = []
    for i_receiver, receiver in enumerate(data.receivers):
        flag_recv = data.flags.get(recv=i_receiver)
        flag_recv_combine = flag_recv.combine(threshold=1)
        flag_percent.append(round(np.sum(flag_recv_combine.array>=1)/len(flag_recv_combine.array.flatten()), 4))
        receivers_list.append(str(receiver))

    return receivers_list, flag_percent


def Synch_model_sm(data: TimeOrderedData, synch_model, nside, beamsize, beam_frequency):

    """
    return the beam smoothed Synch model that occupies the same frequency and spatial region as the scan data
    param synch_model: the model used to create the synchrotron sky [str]
    param beamsize: the beam fwhm used to smooth the Synch model [arcmin]
    param beam_frequency: reference frequencies at which the beam fwhm are defined [MHz]
    param nside: resolution parameter at which the synchrotron model is to be calculated
    """

    sky = pysm3.Sky(nside=nside, preset_strings=synch_model)

    ###########    frequency should be in Hz unit   ###########
    freq = data.frequencies.squeeze * u.Hz

    synch_model = np.zeros(data.visibility.array.shape)

    for i_freq in np.arange(synch_model.shape[1]):

        map_reference = sky.get_emission(freq[i_freq]).value
        map_reference_smoothed = pysm3.apply_smoothing_and_coord_transform(map_reference, fwhm=beamsize*u.arcmin * ((beam_frequency*u.MHz)/(freq[i_freq])).decompose().value )


        for i_receiver, receiver in enumerate(data.receivers):
            i_antenna = data.antenna_index_of_receiver(receiver=receiver)
            right_ascension = data.right_ascension.get(recv=i_antenna).squeeze
            declination = data.declination.get(recv=i_antenna).squeeze

            c = SkyCoord(ra=right_ascension * u.degree, dec=declination * u.degree, frame='icrs')
            theta = 90. - (c.galactic.b / u.degree).value
            phi = (c.galactic.l / u.degree).value

            synch_I = hp.pixelfunc.get_interp_val(map_reference_smoothed[0], theta / 180. * np.pi, phi / 180. * np.pi)

            synch_model[:,i_freq,i_receiver] = synch_I

    return synch_model

def plot_data(x, y, z, gsize=30, levels=15, grid_method='linear', scatter=False, cmap='jet', vmin=None, vmax=None):
    """
    Plotting function
    This plots a rasterscan as an intensity image
    x,y,z must be the same length and z is the amplitude
    
    param x, y: the position of the data points
    param z: the intensity map
    param gsize: the number of the regrid pixels
    param levels: level for the contour
    param grid_method: the method to interpolate
    param scatter: adds markers to indicate the data points
    param cmap: color map used for imshow 
    param vmin, vmax: define the data range that the colormap covers
    """

    if type(z) == np.ma.core.MaskedArray:
        m = ~z.mask
        x = x[m]
        y = y[m]            
        z = z[m]
    else:
        print ('type error, z should be masked array')

    # define grid.
    npts = z.shape[0]
    xi = np.linspace(x.min(), x.max(), gsize)
    yi = np.linspace(y.min(), y.max(), gsize)
    # grid the data.
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method = grid_method)
    IM = plt.imshow(zi[::-1, :], vmin=vmin, vmax=vmax, extent=(x.min(),x.max(),y.min(),y.max()), cmap=plt.get_cmap(cmap,255), aspect='auto')
    CT = plt.contour(xi, yi, zi, levels, linewidths=0.5, colors='k')

    if scatter:
        plt.scatter(x,y)

    return IM


def plot_mdata(x, y, z, gsize=90, levels=6, x_mask=1, y_mask=1, grid_method='linear', cmap='jet', scatter=False, vmin=None, vmax=None):
    """
    Plotting function
    This plots a rasterscan as an intensity image, masked area set to NaN
    x,y,z must be the same length and z is the amplitude

    param x, y: the position of the data points
    param z: the intensity map
    param gsize: the number of the regrid pixels
    param levels: level for the contour
    param x_mask, y_mask: the x, y region beyond the masked point that set to NaN
    param grid_method: the method to interpolate
    param scatter: adds markers to indicate the data points
    param cmap: color map used for imshow
    param vmin, vmax: define the data range that the colormap covers
    param vmax:
    """

    if type(z) == np.ma.core.MaskedArray:
        if np.ndim(z) == 1:
            m = ~z.mask
            #masked area
            mx= x[z.mask]
            my= y[z.mask]
            #plot area
            xx = x[m]
            yy = y[m]
            zz = z[m]
        else:
            print ('shape error')
    # define grid.
    xi = np.linspace(x.min(), x.max(), gsize)
    yi = np.linspace(y.min(), y.max(), gsize)

    x_nan_list=[]
    y_nan_list=[]
    for i in range(len(mx)):
        ii=np.where(abs(xi-mx[i])==np.min(abs(xi-mx[i])))[0][0]
        jj=np.where(abs(yi-my[i])==np.min(abs(yi-my[i])))[0][0]
        x_nan_list.append(ii)
        y_nan_list.append(jj)

    # grid the data.
    zi = griddata((xx, yy), zz, (xi[None, :], yi[:, None]), method = grid_method)

    for i in range(len(x_nan_list)):
        zi[y_nan_list[i]-y_mask:y_nan_list[i]+y_mask+1,x_nan_list[i]-x_mask:x_nan_list[i]+x_mask+1]=np.NaN #imshow, (y,x) is confirmed by plot#
    # contour the gridded data, plotting dots at the randomly spaced data points.
    IM = plt.imshow(zi[::-1, :], vmin=vmin, vmax=vmax, extent=(x.min(),x.max(),y.min(),y.max()), cmap=cmap, aspect='auto' )
    CT = plt.contour(xi, yi, zi, levels, linewidths=0.5, colors='k')

    if scatter:
        plt.scatter(xx,yy)

    return IM


# define the function to project the calibrated visibility on a 2D grid

def project_2d(x, y, data, shape, weights=None):
    """Project x,y, data TOIs on a 2D grid.

    Parameters
    ----------
    x, y : array_like
        input pixel indexes, 0 indexed convention
    data : array_like
        input data to project
    shape : int or tuple of int
        the shape of the output projected map
    weights : array_like
        weights to be use to sum the data (by default, ones)

    Returns
    -------
    data, weight, hits : ndarray
        the projected data set and corresponding weights and hits

    Notes
    -----
    The pixel index must follow the 0 indexed convention, i.e. use `origin=0` in `*_worl2pix` methods from `~astropy.wcs.WCS`.

    >>> data, weight, hits = project([0], [0], [1], 2)
    >>> data
    array([[ 1., nan],
           [nan, nan]])
    >>> weight
    array([[1., 0.],
           [0., 0.]])
    >>> hits
    array([[1, 0],
           [0, 0]]))

    >>> data, _, _ = project([-0.4], [0], [1], 2)
    >>> data
    array([[ 1., nan],
           [nan, nan]])

    There is no test for out of shape data

    >>> data, _, _ = project([-0.6, 1.6], [0, 0], [1, 1], 2)
    >>> data
    array([[nan, nan],
           [nan, nan]])

    Weighted means are also possible :

    >>> data, weight, hits = project([-0.4, 0.4], [0, 0], [0.5, 2], 2, weights=[2, 1])
    >>> data
    array([[ 1., nan],
           [nan, nan]))
    >>> weight
    array([[3., 0.],
           [0., 0.]])
    >>> hits
    array([[2, 0],
           [0, 0]])
    """
    if isinstance(shape, (int, np.integer)):
        shape = (shape, shape)

    assert len(shape) == 2, "shape must be a int or have a length of 2"

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if weights is None:
        weights = np.ones(data.shape)

    if isinstance(data, np.ma.MaskedArray):
        # Put weights as 0 for masked data
        hit_weights = ~data.mask
        weights = weights * hit_weights
    else:
        hit_weights = np.ones_like(data, dtype=bool)

    kwargs = {"bins": shape, "range": tuple((-0.5, size - 0.5) for size in shape)}

    _hits, xedges, yedges = np.histogram2d(y, x, weights=hit_weights, **kwargs)
    _weights = np.histogram2d(y, x, weights=weights, **kwargs)[0]
    _data = np.histogram2d(y, x, weights=weights * data, **kwargs)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = _data / _weights

    return output, _weights, _hits.astype(int), xedges, yedges


def project_3d(x, y, z, data, shape, weights=None, weighted_output=False):
    """Project x,y, data TOIs on a 2D grid.

    Parameters
    ----------
    x, y, z : array_like
        input pixel indexes, 0 indexed convention
    data : array_like
        input data to project
    shape : int or tuple of int
        the shape of the output projected map
    weights : array_like
        weights to be use to sum the data (by default, ones)
    weighted_output : bool
        if True, return the weighted output instead of normalized output, by default False

    Returns
    -------
    data, weight, hits : ndarray
        the projected data set and corresponding weights and hits

    Notes
    -----
    The pixel index must follow the 0 indexed convention, i.e. use `origin=0` in `*_worl2pix` methods from `~astropy.wcs.WCS`.

    """
    if isinstance(shape, (int, np.integer)):
        shape = (shape, shape, shape)
    if len(shape) == 2:
        shape = (shape[0], shape[1], shape[1])

    assert len(shape) == 3, "shape must be a int or have a length of 2 or 3"

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if weights is None:
        weights = np.ones_like(data)

    if isinstance(data, np.ma.MaskedArray):
        # Put weights as 0 for masked data
        hit_weights = ~data.mask
        weights = weights * hit_weights
    else:
        hit_weights = np.ones_like(data, dtype=bool)

    kwargs = {
        "bins": shape,
        "range": tuple((-0.5, float(size) - 0.5) for size in shape),
    }

    sample = (z, y, x)
    _hits = np.histogramdd(sample, weights=hit_weights, **kwargs)[0]
    _weights = np.histogramdd(sample, weights=weights, **kwargs)[0]
    _data = np.histogramdd(sample, weights=weights * data, **kwargs)[0]

    if not weighted_output:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _data /= _weights

    return _data, _weights, _hits.astype(int)
