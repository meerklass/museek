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


def Synch_model_sm(data: TimeOrderedData, nside, beamsize, beam_frequency):

    """
    return the beam smoothed Synch model that occupies the same frequency and spatial region as the scan data
    param beamsize: the beam fwhm used to smooth the Synch model [arcmin]
    param beam_frequency: reference frequencies at which the beam fwhm are defined [MHz]
    param nside: resolution parameter at which the synchrotron model is to be calculated
    """

    sky = pysm3.Sky(nside=nside, preset_strings=["s1"])

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


