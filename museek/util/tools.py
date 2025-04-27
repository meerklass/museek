import numpy as np
from museek.time_ordered_data import TimeOrderedData
import pysm3
import pysm3.units as u
import healpy as hp
from astropy.coordinates import SkyCoord
import matplotlib.colors as colors
import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import warnings
import scipy
import subprocess
import os

def git_version_info(directory=None):
    """
    Get git branch and commit information.

    Parameters:
    directory: str - Optional path to the git repository. Uses current directory if None.

    Returns:
    tuple: (branch_name, commit_hash) or (None, None) if not in a git repository
    """
    # Store original directory to return to it later
    original_dir = os.getcwd()
    
    # Use current directory if no repo_dir provided
    if directory is None:
        directory = os.environ.get('MUSEEK_REPO_DIR')
        if directory is None:
            # No explicit directory provided - use current directory
            directory = os.getcwd()

    try:
        # Change to the specified directory
        os.chdir(directory)

        # Check if we're in a git repository first
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )

        if result.returncode != 0:
            return None, None

        # Get branch name
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode().strip()

        # Get short commit hash
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode().strip()

        return branch, commit

    except Exception as e:
        print(f"Error getting git info: {e}")
        return None, None

    finally:
        # Return to original directory
        os.chdir(original_dir)


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


def point_sources_coordinate(point_source_file_path, right_ascension_scan, declination_scan, point_sources_match_flux, point_sources_match_raregion, point_sources_match_decregion):
    """
    Return coordinates of selected point sources

    Parameters
    ----------
    point_source_file_path: str
    path where point source catalog is located
    right_ascension_scan: float
    the median of the right ascension of a scan [deg]
    declination_scan: float
    the median of the declination of a scan [deg]
    point_sources_match_flux: float 
    flux threshold for selecting the point source for masking [Jy]
    point_sources_match_raregion: float
    the ra distance to the median of observed ra to select the point sources [deg]
    point_sources_match_decregion: float
    the dec region to the median of observed dec to select the point sources [deg]

    Returns
    -------
    ra, dec, flux : ndarray
    the ra, dec, and flux of the selected point sources
    """

    ps_info = np.genfromtxt(point_source_file_path+'/1jy_scat-V0_new.txt', delimiter='|')
    ra_sources = ps_info[:,1]
    dec_sources = ps_info[:,2]
    flux_sources = ps_info[:,3]

    ra_selection = (ra_sources>=right_ascension_scan-point_sources_match_raregion) & (ra_sources<=right_ascension_scan+point_sources_match_raregion)
    dec_selection = (dec_sources>=declination_scan-point_sources_match_decregion) & (dec_sources<=declination_scan+point_sources_match_decregion)
    flux_selection = (flux_sources>=point_sources_match_flux)

    select_sources = ra_selection & dec_selection & flux_selection

    ra_sources_select = ra_sources[select_sources]
    dec_sources_select = dec_sources[select_sources]
    flux_sources_select = flux_sources[select_sources]

    return ra_sources_select, dec_sources_select, flux_sources_select


def point_source_flag(ra_point_source, dec_point_source, ra_scan, dec_scan, frequency, beam_threshold, beamsize, beam_frequency):
    """
    Return TOD mask for point sources, as a funtion of frequency

    Parameters
    ----------
    ra_point_source, dec_point_source: ndarray
        position of the point sources to be masked
    ra_scan, dec_scan: ndarray
        pointing of the TOD
    frequency, ndarray
        the frequency coverage, in unit of Hz
    beam_threshold: float
        times of the beam size around the point source to be masked
    beamsize: float
        the beam fwhm [arcmin]
    beam_frequency float
        reference frequency at which the beam fwhm are defined [MHz]

    Returns
    -------
    mask_point_source : ndarray
    the TOD mask for point sources
    """

    shape = (len(ra_scan), len(frequency))
    mask_point_source = np.zeros(shape, dtype=bool)

    # create a `list` of `SkyCoord` coordinates of scan pointings
    skycoord_scan = SkyCoord(ra_scan * u.deg, dec_scan * u.deg, frame='icrs')

    # create a `list` of `SkyCoord` coordinates of point sources that will be masked
    skycoord_point_source = [SkyCoord(ra_ps * u.deg, dec_ps * u.deg, frame='icrs') for ra_ps, dec_ps in zip(ra_point_source, dec_point_source)]
    
    for i_freq, freq in enumerate(frequency):
        fwhm = beamsize / 60. * ((beam_frequency*u.MHz)/(freq * u.Hz)).decompose().value  ## in deg

        mask_point_source_dump_list = []
        for mask_coord in skycoord_point_source:
            separation = (mask_coord.separation(skycoord_scan) / u.deg)
            mask_point_source_dump_list.extend( np.where(separation < (beam_threshold * fwhm) )[0])
        mask_point_source_dump_list = list(set(mask_point_source_dump_list))
        mask_point_source_dump_list.sort()

        mask_point_source[mask_point_source_dump_list, i_freq] = True

    return mask_point_source


def remove_outliers_zscore_mad(data, mask, threshold=3.5):
    """
    Removes outliers from the data based on the Median Absolute Deviation (MAD) method.

    Parameters:
    data (array-like): The input signal.
    mask (array-like): Boolean mask for the sample points, where True represents masked points.
    threshold (float): The threshold in terms of modified Z-score based on MAD. 
                       Data points with a modified Z-score greater than this threshold 
                       will be considered outliers.

    Returns:
    array-like: Updated mask where outliers are also masked.
    """
    # Make a copy of the mask to avoid modifying the original
    initial_mask = mask.copy()

    # Mask the entire data if more than 60% of data points are already masked
    if np.mean(mask) > 0.6:
        initial_mask[:] = True
    else:
        # Calculate the median of the unmasked data
        median = np.median(data[~mask])

        # Calculate the MAD of the unmasked data
        mad = np.median(np.abs(data[~mask] - median))

        # Avoid division by zero; if MAD is zero, use a small constant
        if mad == 0:
            mad = 1e-10

        # Compute modified Z-scores using MAD
        modified_z_scores = np.abs((data[~mask] - median) / mad)

        # Mask data points where modified Z-score exceeds the threshold
        initial_mask[~mask] |= modified_z_scores > threshold

    return initial_mask


def polynomial_flag_outlier(x, y, mask, degree, threshold):
    """
    fitting a powerlaw to the input data and mask outliers which deviate the powerlaw larger than `threshold' times median absolute deviation

    Parameters:
    x: x-coordinates of the sample points [array]
    y: y-coordinates of the sample points [array]
    mask: mask for the sample points [bool]
    degree: Degree of the fitting polynomial [int]
    threshold: threshold of the masking [float]

    Returns:
    bool array: updated mask
    1darray: fitting polynomial coefficients
    """
    initial_mask = mask.copy()
    ########  mask the whole data if the flagged fraction is larger than 0.6
    if np.mean(mask>0) > 0.6:
        initial_mask[:] = True
        p_fit = None
    else:
        fit_x = x[~mask]
        fit_y = y[~mask]

        # Fit a polynomial
        p_fit = np.polyfit(fit_x, fit_y, degree)
        y_fit = np.polyval(p_fit, fit_x)

        # Calculate residuals
        residuals = fit_y - y_fit
        mad_residuals = scipy.stats.median_abs_deviation(residuals)

        # Identify outliers
        initial_mask[~mask] |= np.abs(residuals) >= threshold * mad_residuals
    return initial_mask, p_fit


def moving_median_masked(data, window_size=20):
    """
    Calculate the moving median for a masked array.

    Parameters:
    -----------
    data : numpy.ma.MaskedArray
        The input masked array
    window_size : int, optional
        The size of the window (default: 20)

    Returns:
    --------
    numpy.ma.MaskedArray
        Moving median values with the same shape as the input array
    """
    # Ensure the input is a masked array
    if not isinstance(data, np.ma.MaskedArray):
        data = np.ma.array(data, mask=np.isnan(data))

    # Get the shape of the input array
    n = len(data)

    # Initialize output array
    result = np.ma.zeros(n)
    result.mask = np.ones(n, dtype=bool)  # Start with all values masked

    # Calculate the moving median
    half_window = window_size // 2

    for i in range(n):
        # Calculate the window boundaries
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        # Extract the window
        window = data[start:end]

        # Skip if all values in the window are masked
        if window.count() == 0:
            continue

        # Calculate median of unmasked values
        result[i] = np.ma.median(window)

    return result


def gaussian_filter_masked(masked_array, sigma, **kwargs):
    """
    Apply a Gaussian filter to a masked array.

    Parameters:
    -----------
    masked_array : numpy.ma.MaskedArray
        The input masked array to filter
    sigma : float or sequence of floats
        Standard deviation for Gaussian kernel
    **kwargs :
        Additional arguments to pass to scipy.ndimage.gaussian_filter

    Returns:
    --------
    numpy.ma.MaskedArray
        The filtered array with the same mask as the input
    """
    # Get the data and mask
    data = masked_array.filled(0)  # Fill masked values with 0
    mask = masked_array.mask

    if mask is np.ma.nomask:
        # If there's no mask, just apply the filter to the data
        filtered_data = scipy.ndimage.gaussian_filter(data, sigma, **kwargs)
        return np.ma.array(filtered_data)

    # Create an array of weights (0 for masked, 1 for unmasked)
    weights = np.logical_not(mask).astype(float)

    # Apply the filter to the data and weights
    filtered_data = scipy.ndimage.gaussian_filter(data * weights, sigma, **kwargs)
    filtered_weights = scipy.ndimage.gaussian_filter(weights, sigma, **kwargs)

    # Avoid division by zero
    filtered_weights = np.where(filtered_weights > 0, filtered_weights, 1)

    # Normalize the filtered data by the filtered weights
    result = filtered_data / filtered_weights

    # Apply the original mask to the result
    return np.ma.array(result, mask=mask)



def interpolate_1d_masked_array(masked_array, kind='linear'):
    """
    Interpolate unmasked regions and extrapolate masked regions in a 1D masked array
    using specified interpolation method.
    
    Parameters:
    -----------
    masked_array : numpy.ma.MaskedArray
        The 1D masked array to process
    kind : str, optional (default='linear')
        Specifies the kind of interpolation as a string or integer
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.)
        
    Returns:
    --------
    numpy.ndarray
        1D array with interpolated and extrapolated values replacing masked areas
    """
    # Check if input is 1D
    if masked_array.ndim != 1:
        raise ValueError("Function only works with 1D masked arrays")
    
    # Make a copy to avoid modifying the original
    result = masked_array.copy()
    
    # Get indices for all points
    indices = np.arange(len(masked_array))
    
    # Get unmasked indices and values
    valid_indices = indices[~masked_array.mask]
    
    # If all values are masked or no values are masked, return the original data
    if len(valid_indices) == 0:
        return result.data
    if len(valid_indices) == len(indices):
        return result.data
        
    valid_values = masked_array.data[valid_indices]
    
    # Check if we have enough points for the requested interpolation method
    min_points = {
        'linear': 2,
        'nearest': 1,
        'zero': 2,
        'slinear': 2,
        'quadratic': 3,
        'cubic': 4,
        'polynomial': 5
    }
    
    # Default to linear if not in our dictionary
    required_points = min_points.get(kind, 2)
    
    # Fall back to simpler interpolation if we don't have enough points
    if len(valid_indices) < required_points:
        if len(valid_indices) >= 2:
            kind = 'linear'
        else:
            kind = 'nearest'
    
    # Create interpolation function
    f = interpolate.interp1d(valid_indices, valid_values, 
                            kind=kind, 
                            bounds_error=False, 
                            fill_value='extrapolate')
    
    # Apply to masked values only
    masked_indices = indices[masked_array.mask]
    if len(masked_indices) > 0:
        result.data[masked_array.mask] = f(masked_indices)
    
    return result.data



