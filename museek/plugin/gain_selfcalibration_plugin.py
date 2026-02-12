import os
from typing import Generator
from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.data_element import DataElement
from museek.enums.result_enum import ResultEnum
from museek.flag_element import FlagElement
from museek.flag_factory import FlagFactory
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import git_version_info
from museek.util.tools import remove_outliers_zscore_mad, polynomial_flag_outlier
from museek.util.tools import moving_median_masked, gaussian_filter_masked, interpolate_1d_masked_array
import pysm3.units as u
import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
import astropy.coordinates as ac
import pickle
import gc
import scipy
import warnings
import datetime
from numpy.polynomial.legendre import Legendre, legval
from scipy.optimize import least_squares

class GainSelfCalibrationPlugin(AbstractParallelJoblibPlugin):
    """ Plugin to calibrtion the gain using input map """

    def __init__(self,
                 frequency_high: float,
                 frequency_low: float,
                 flag_combination_threshold: int,
                 do_store_context: bool,
                 nd_zscoreflag_threshold: float,
                 nd_polyflag_deg: int,
                 nd_polyflag_threshold: float,
                 nd_polyfit_deg: int,
                 nd_window_movingmedian: int,
                 nd_gausm_sigma: int,
                 do_delete_auto_data: bool,
                 map_path: str,
                 map_name: str,
                 baseline_polyfit_deg: int,
                 gain_polyfit_deg: int,
                 model_polyfit_deg: int,
                 deg_range: int,
                 deg_step: int,
                 **kwargs):
        """
        Initialise the plugin
        :param frequency_high: high frequency cut 
        :param frequency_low: low frequency cut
        :param flag_combination_threshold: for combining sets of flags, usually `1`
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        :param nd_zscoreflag_threshold: threshold for flagging noise diode excess using modified zscore method
        :param nd_polyflag_deg: degree of the polynomials used for fitting and flagging noise diode excess  
        :param nd_polyflag_threshold: threshold for flagging noise diode excess using polynomials fit
        :param nd_polyfit_deg: degree of the polynomials used for fitting flagged noise diode excess
        :param nd_window_movingmedian: The size of the window for the moving median calculation for frequency spectrum of noise diode signal
        :param nd_gausm_sigma: The size of the window for the Gaussian Smooth of Noise Diode Excess frequency spectrum
        :param do_delete_auto_data: switch that determines wether the raw auto data should be deleted after self-calibration
        :para map_path: directory of the input map
        :param map_name: name of the input map
        :param baseline_polyfit_deg: degree of the polynomials used for fitting time baseline in raw data 
        :param gain_polyfit_deg: degree of the polynomials used for fitting gain
        :param model_polyfit_deg: degree of the polynomials used for fitting input model
        :param deg_range: Half-width of the search window in polynomial degree space
        :param deg_step: Step size used when sampling degrees within the ±deg_range window

        """
        super().__init__(**kwargs)
        self.frequency_high = frequency_high
        self.frequency_low = frequency_low
        self.flag_combination_threshold = flag_combination_threshold
        self.do_store_context = do_store_context
        self.nd_zscoreflag_threshold = nd_zscoreflag_threshold
        self.nd_polyflag_deg = nd_polyflag_deg
        self.nd_polyflag_threshold = nd_polyflag_threshold
        self.nd_polyfit_deg = nd_polyfit_deg
        self.nd_window_movingmedian = nd_window_movingmedian
        self.nd_gausm_sigma = nd_gausm_sigma
        self.do_delete_auto_data = do_delete_auto_data
        self.map_path = map_path
        self.map_name = map_name
        self.baseline_polyfit_deg = baseline_polyfit_deg
        self.gain_polyfit_deg = gain_polyfit_deg
        self.model_polyfit_deg = model_polyfit_deg
        self.deg_range = deg_range
        self.deg_step = deg_step

    def set_requirements(self):
        """
        Set the requirements, the scanning data `scan_data`, a path to store results and the name of the data block.
        """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data'),
                             Requirement(location=ResultEnum.OUTPUT_PATH, variable='output_path'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
                             Requirement(location=ResultEnum.POINT_SOURCE_FLAG, variable='point_source_flag'),
                             Requirement(location=ResultEnum.NOISE_DIODE_EXCESS, variable='nd_excess'),
                             Requirement(location=ResultEnum.NOISE_ON_INDEX, variable='nd_on_index'),
                             Requirement(location=ResultEnum.FLAG_REPORT_WRITER, variable='flag_report_writer')]

    def map(self, 
            scan_data: TimeOrderedData, 
            point_source_flag: np.ndarray, 
            nd_excess: np.ndarray, 
            nd_on_index: np.ndarray, 
            output_path: str, 
            block_name: str, 
            flag_report_writer: ReportWriter) \
            -> Generator[tuple[np.ndarray, np.ndarray, np.ma.MaskedArray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, astropy.wcs.WCS, np.ma.MaskedArray, np.ndarray], None, None]:
        """
        Run the gain calibration
        :return: the calibrated scan_data
        :param data: the time ordered scan data
        :param point_source_flag: mask for point sources
        :param nd_excess: noise diode excess signal
        :param nd_on_index: the index of the noise firing timestamps
        :param output_path: path to store results
        :param block_name: name of the observation block
        :param flag_report_writer: report of the flag info
        """

        
        ########  load the visibility  ###########
        scan_data.load_visibility_flags_weights(polars='auto')
        initial_flags = scan_data.flags.combine(threshold=self.flag_combination_threshold)
        freq = scan_data.frequencies.squeeze    ####  the unit of scan_data.frequencies is Hz 
        timestamps = scan_data.timestamps.array.squeeze()

        #########  select the frequency region we want to use  #######
        freqlow_index = np.argmin(np.abs(freq/10.**6 - self.frequency_low))
        freqhigh_index = np.argmin(np.abs(freq/10.**6 - self.frequency_high))
        freq_select = freq[freqlow_index:freqhigh_index]

        #########  load the map and fit a model using polynomial   ########
        data_read = pickle.load(open(self.map_path+'/'+self.map_name,'rb'))
        map_model = data_read['map']
        map_shape = map_model.shape[:2]
        wcs_map = data_read['wcs']
        freq_map = data_read['freq']      ####   should be in MHz
        assert not np.any(freq_map > 10000), "Frequency values from saved map must not exceed 10000, because they should be in MHz"

        #######  loop for each receiver   ########
        for i_receiver, receiver in enumerate(scan_data.receivers):

            i_antenna = scan_data.antenna_index_of_receiver(receiver=receiver)
            right_ascension = scan_data.right_ascension.get(recv=i_antenna).squeeze
            declination = scan_data.declination.get(recv=i_antenna).squeeze

            visibility_recv = scan_data.visibility.get(recv=i_receiver).squeeze
            initial_flag = initial_flags.get(recv=i_receiver).squeeze
            visibility_recv = visibility_recv[:,freqlow_index:freqhigh_index]
            initial_flag = initial_flag[:,freqlow_index:freqhigh_index]
            nd_excess_recv = nd_excess[:,freqlow_index:freqhigh_index,i_receiver]

            yield  receiver.name, visibility_recv, initial_flag, nd_excess_recv, nd_on_index, point_source_flag[:,:,i_antenna], right_ascension, declination, timestamps, wcs_map, map_model, map_shape, freq_select, output_path, block_name


    def run_job(self, anything: tuple[str, np.ndarray, np.ndarray, np.ma.MaskedArray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, astropy.wcs.WCS, np.ma.MaskedArray, np.ndarray, np.ndarray, str, str]) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray]:
        """
        Run the nd_excess smooth and selfcalibration. Done for one receiver at a time.
        :param anything: `tuple` of the raw data, initial flag, noise diode excess, point source flag, right ascension, declination, wcs of input map, model for the input map
        :return: self-calibrated map, gain from self-calibration, and baseline from self-calibration 
        """

        receiver_name, visibility_recv, initial_flag, nd_excess_recv, nd_on_index,  point_source_flag, right_ascension, declination, timestamps, wcs_map, map_model, map_shape, freq_select, output_path, block_name = anything

        ########  calibrate the raw data using noise diode on-off
        #######  fit the noise diode on - off, and normalise the trend in time  #########
        nd_excess_recv_fit = np.ones(visibility_recv.shape)
        if initial_flag.all():
            pass
        else:
            for i_timestamp in np.arange(nd_excess_recv.shape[0]):
                if nd_excess_recv.mask[i_timestamp,:].all():
                    pass
                else:
                    nd_excess_time = nd_excess_recv[i_timestamp,:].copy()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ###  calculate the moving_median of the noise diode on - off signal
                        nd_excess_time_mf = moving_median_masked(nd_excess_time, window_size=self.nd_window_movingmedian)
                        ###  flagging outliers
                        nd_excess_time_residual = nd_excess_time_mf - nd_excess_time
                        nd_excess_time.mask = remove_outliers_zscore_mad(nd_excess_time_residual.data, nd_excess_time_residual.mask, self.nd_zscoreflag_threshold)
                        ###  interpolating masked regions
                        nd_excess_time = interpolate_1d_masked_array(nd_excess_time, kind='linear')
                        ###  gaussian smoothing
                        nd_excess_recv[i_timestamp] = scipy.ndimage.gaussian_filter(nd_excess_time, sigma=self.nd_gausm_sigma, mode='nearest')

            for i_freq in np.arange(nd_excess_recv.shape[1]):
                if nd_excess_recv.mask[:,i_freq].all():
                    pass
                else:
                    nd_excess_freq = nd_excess_recv[:,i_freq].copy()
                    nd_excess_freq.mask = remove_outliers_zscore_mad(nd_excess_freq.data, nd_excess_freq.mask, self.nd_zscoreflag_threshold)
                    nd_excess_freq.mask, p_fit = polynomial_flag_outlier(nd_on_index, nd_excess_freq.data, nd_excess_freq.mask, self.nd_polyflag_deg, self.nd_polyflag_threshold)
                    p_poly = np.polyfit(nd_on_index[~nd_excess_freq.mask], nd_excess_freq.data[~nd_excess_freq.mask], deg=self.nd_polyfit_deg)
                    nd_excess_recv_fit[:,i_freq] = np.polyval(p_poly, np.arange(visibility_recv.shape[0]))

        visibility_recv = visibility_recv / (nd_excess_recv_fit / np.median(nd_excess_recv_fit))
        del nd_excess_recv_fit
        gc.collect()


        #############   self calibration   ##############
        temperature_noaver_recv = np.ma.masked_array(np.zeros(visibility_recv.shape), mask=initial_flag)
        gain_selfcali_params = {}
        baseline_selfcali_params = {}
        time_scaled = self.scaling_time_Legendre(timestamps)

        #####  update the mask to avoid the point source ######
        mask_update = initial_flag.copy() + point_source_flag
        visibility_recv = np.ma.masked_array(visibility_recv, mask=mask_update)

        #####  self calibration   #####
        sky_sc=ac.SkyCoord(ra=right_ascension.flatten() * u.deg, dec=declination.flatten() * u.deg) # pointings in observation
        pix_coords=skycoord_to_pixel(sky_sc, wcs_map)

        model_rebuild = np.ma.ones_like(visibility_recv)
        for i_freq in np.arange(visibility_recv.shape[1]):
            for i_time in np.arange(visibility_recv.shape[0]):
                pix_coords_x = pix_coords[0][i_time]
                pix_coords_y = pix_coords[1][i_time]
                x_index = np.argmin(np.abs(pix_coords_x - np.arange(map_shape[0])))
                y_index = np.argmin(np.abs(pix_coords_y - np.arange(map_shape[1])))
                model_rebuild.data[i_time,i_freq] = map_model.data[x_index,y_index,i_freq].copy()
                model_rebuild.mask[i_time,i_freq] = map_model.mask[x_index,y_index,i_freq].copy()

        #####  combine the mask of data and model ######
        mask_combine = visibility_recv.mask + model_rebuild.mask + (~np.isfinite(visibility_recv.data)) + (~np.isfinite(model_rebuild.data))
        visibility_recv.mask = mask_combine
        model_rebuild.mask = mask_combine

        # ================================================================
        # Estimate multiple candidate initial gains g0
        # ================================================================
        # The constant gain term g0 dominates the solution.
        # Bad g0 choices can trap the optimizer in poor local minima.
        #
        # We compute several robust, complementary estimators and
        # let the model-selection loop choose the best one.
        # ================================================================
        # --- (a) Standard deviation ratio -------------------------------
        # Assumes roughly linear scaling between d and m.
        g0_std = np.ma.std(visibility_recv, axis=0) / np.ma.std(model_rebuild, axis=0)

        # --- (b) Covariance-based slope (demeaned) -----------------------
        # Least-squares slope through the origin after mean removal.
        d0 = visibility_recv - np.ma.mean(visibility_recv, axis=0)
        m0 = model_rebuild - np.ma.mean(model_rebuild, axis=0)
        g0_cov = np.ma.sum(d0 * m0, axis=0) / np.ma.sum(m0 * m0, axis=0)

        # --- (c) robust linear regression slope using Huber loss -----
        # Robust non-parametric linear slope estimator.
        g0_Huber = []
        for i_freq in np.arange(visibility_recv.shape[1]):
            mask_freq = mask_combine[:, i_freq]
            if np.sum(~mask_freq) < 2:  # Require at least 2 valid points for fit
                g0_Huber.append(np.nan)
            else:
                y_freq = visibility_recv.data[~mask_freq, i_freq]
                x_freq = model_rebuild.data[~mask_freq, i_freq]
                try:
                    g0_Huber.append(self.robust_slope_intercept(y_freq, x_freq)[0])
                except Exception as e:
                    print(f"Warning: Fit failed for freq {i_freq}: {e}. Using NaN.")
                    g0_Huber.append(np.nan)

        ###########    fit the gain and baseline using fit_gain_noconstant_base_Legendre_auto()   ###########
        for i_freq in np.arange(visibility_recv.shape[1]):
            if visibility_recv[:,i_freq].mask.all():
                gain_selfcali_params['index_freq_'+str(i_freq)+' '+receiver_name] = np.nan
                baseline_selfcali_params['index_freq_'+str(i_freq)+' '+receiver_name] = np.nan
                pass
            else:
                gain, baseline, c_gain, c_base = self.fit_gain_noconstant_base_Legendre_auto(
                                                                     timestamps,
                                                                     visibility_recv.data[:,i_freq],
                                                                     model_rebuild.data[:,i_freq],
                                                                     visibility_recv.mask[:,i_freq],
                                                                     model_rebuild.mask[:,i_freq],
                                                                     [g0_std[i_freq], g0_cov[i_freq], g0_Huber[i_freq]],
                                                                     default_gain=self.gain_polyfit_deg,
                                                                     default_base=self.baseline_polyfit_deg,
                                                                     deg_range = self.deg_range,
                                                                     deg_step = self.deg_step,
                                                                     )

        
                gain_selfcali_params['index_freq_'+str(i_freq)+' '+receiver_name] = c_gain
                baseline_selfcali_params['index_freq_'+str(i_freq)+' '+receiver_name] = c_base
                temperature_noaver_recv.data[:,i_freq] = (visibility_recv.data[:,i_freq] - baseline) / gain


        return temperature_noaver_recv, gain_selfcali_params, baseline_selfcali_params, time_scaled, map_model, wcs_map

    def gather_and_set_result(self,
                              result_list: list[np.ndarray],
                              scan_data: TimeOrderedData,
                              point_source_flag: np.ndarray,
                              nd_excess: np.ndarray,
                              nd_on_index: np.ndarray,
                              flag_report_writer: ReportWriter,
                              output_path: str,
                              block_name: str):

        """
        Combine the `np.ma.MaskedArray`s in `result_list` into a new data set
        :param result_list: `list` of `np.ndarray`s created from the self-calibration
        :param scan_data: time ordered data containing the scanning part of the observation
        :param point_source_flag: mask for point sources
        :param flag_report_writer: report of the flag
        :param output_path: path to store results
        """

        result_list = np.array(result_list, dtype='object')
        temperature_noaver = np.ma.masked_array([result_list[i][0] for i in range(np.shape(result_list)[0])]).transpose(1, 2, 0)
        gain_selfcali_params_list = [result_list[i][1] for i in range(np.shape(result_list)[0])]
        gain_selfcali_params = {}
        for d in gain_selfcali_params_list:
            gain_selfcali_params.update(d)
        baseline_selfcali_params_list = [result_list[i][2] for i in range(np.shape(result_list)[0])]
        baseline_selfcali_params = {}
        for d in baseline_selfcali_params_list:
            baseline_selfcali_params.update(d)
        time_scaled = result_list[0][3]
        map_model = result_list[0][4]
        wcs_map = result_list[0][5]

        ################  combine HH and VV, combine the mask of HH and VV firstly ###################
        temperature_noaver_antennas = []
        mask_antennas = []
        antenna_list = scan_data._antenna_name_list
        receivers_list = [str(receiver) for i_receiver, receiver in enumerate(scan_data.receivers)]

        for antenna in antenna_list:
            indices = [index for index, receiver in enumerate(receivers_list) if antenna in receiver]
            selected_mask = [temperature_noaver.mask[:,:,i] for i in indices]
            mask_antennas.append(np.sum(selected_mask, axis=0))
            selected_vis = [temperature_noaver.data[:,:,i] for i in indices]
            temperature_noaver_antennas.append(np.mean(selected_vis, axis=0))

        temperature_noaver_antennas = np.ma.masked_array(temperature_noaver_antennas, mask=mask_antennas)
        temperature_noaver_antennas = temperature_noaver_antennas.transpose(1, 2, 0)


        self.set_result(result=Result(location=ResultEnum.GAIN_SELFCALI_PARAMS, result=gain_selfcali_params, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.BASELINE_SELFCALI_PARAMS, result=baseline_selfcali_params, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.MAP_SELFCALI, result=temperature_noaver_antennas, allow_overwrite=True)) 
        self.set_result(result=Result(location=ResultEnum.MAP_MODEL, result=map_model, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.TIME_SCALED_SELFCALI_POLYFIT, result=time_scaled, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.WCS_MODEL, result=wcs_map, allow_overwrite=True))
        if self.do_delete_auto_data:
            scan_data.delete_visibility_flags_weights(polars='auto')
            self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data, allow_overwrite=True))

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = ['...........................', 'Running GainSelfCalibrationPlugin with '+f"MuSEEK version: {branch} ({commit})", 'Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S")]
        flag_report_writer.write_to_report(lines)

        if self.do_store_context:
            context_file_name = 'gain_selfcalibration_plugin.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=output_path)


    def robust_slope_intercept(self, y, x):
        """
        Compute robust linear regression slope and intercept using Huber loss.
    
        This function estimates the slope and intercept of a linear model y = slope * x + intercept
        using robust optimization with Huber loss, which is less sensitive to outliers than ordinary least squares.
        It uses scipy.optimize.least_squares for the optimization.
    
        Parameters:
        -----------
        y : array-like
            Dependent variable (response values).
        x : array-like
            Independent variable (predictor values). Must be the same length as y.
    
        Returns:
        --------
        slope : float
            Estimated slope of the linear regression.
        intercept : float
            Estimated intercept of the linear regression.
    
        Notes:
        ------
        - Initial guess: Slope starts at 1.0 (neutral scaling), intercept at mean(y) for data centering.
        - Huber loss (f_scale=1.0): Balances squared loss for small residuals and linear loss for large ones, providing robustness.
        - If needed, adjust f_scale for more/less sensitivity to outliers (lower value = more robust but potentially slower convergence).
        - Assumes no missing values; preprocess data if necessary.
        - For large datasets, this is efficient (iterative, O(n) per iteration) compared to median-based methods like Theil-Sen.
    
        Example:
        --------
        >>> import numpy as np
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 4, 6, 8, 10]) + np.random.normal(0, 1, 5)  # Noisy linear data
        >>> slope, intercept = robust_slope_intercept(y, x)
        >>> print(slope, intercept)
        """
        def residuals(param):
            """
            Inner function to compute residuals for the linear model.
            param[0]: slope
            param[1]: intercept
            """
            return y - param[0] * x - param[1]  # Residuals with slope and intercept
    
        res = least_squares(residuals, [1.0, np.mean(y)], loss='huber', f_scale=1.0)
        return res.x[0], res.x[1]  # Slope, intercept

    def scaling_time_Legendre(self, t):
        """
        Scales time to [-1, 1] for orthogonal polynomial stability.
        """
        t_min = np.min(t)
        t_max = np.max(t)
        if t_max == t_min:
            return np.zeros_like(t)
        # Scale to range [-1, 1]
        return 2 * (t - t_min) / (t_max - t_min) - 1


    def fit_gain_noconstant_base_Legendre_auto(self,
        t, d, m, mask_d, mask_m, g0_candidates,
        default_gain=3, default_base=3,
        deg_range=1, deg_step=1,
    ):
        """
        Automatic gain + baseline calibration using a multiplicative model:
    
            d(t) ≈ g(t) * m(t) + b(t)
    
        where:
            g(t) : time-dependent gain (Legendre series, no constant term)
            b(t) : additive baseline (Legendre series, includes constant)
    
        This routine performs:
            • automatic initial gain (g0) selection
            • automatic gain/baseline polynomial degree selection
            • constrained fitting enforcing g(t) > 0
            • model-space residual minimization:
                  || m(t) - (d(t) - b(t)) / g(t) ||²
    
        --------------------
        PERFORMANCE STRATEGY
        --------------------
        This is a *cached* implementation:
            ✔ cache time scaling
            ✔ cache Legendre basis evaluations
    
        This removes the dominant repeated-cost operations while
        keeping the code readable and debuggable.
    
        --------------------
        PARAMETERS
        --------------------
        t : array
            Time samples
    
        d : array
            Raw data (to be calibrated)
    
        m : array
            Model data (reference)
    
        mask_d : boolean array
            External mask for d (True = exclude)

        mask_m : boolean array
            External mask for m (True = exclude)
    
        default_gain, default_base : int
            Preferred Legendre polynomial orders
    
        deg_range : int
            Search ± range around default degrees
    
        deg_step : int
            Step size in degree search

        g0_candidates : list
            candidate constant-gain estimators
    
        --------------------
        RETURNS
        --------------------
        gain : array
            Best-fit gain evaluated on full time grid
    
        baseline : array
            Best-fit baseline evaluated on full time grid
    
        c_gain : array
            Gain Legendre coefficients (c_gain[0] = g0)
    
        c_base : array
            Baseline Legendre coefficients
        """
    
        # ================================================================
        # 1. Construct a single, consistent fitting mask
        # ================================================================
        # We exclude:
        #   • NaNs in data
        #   • NaNs in model
        #   • user-provided masked samples
        #
        # This mask is reused everywhere to guarantee consistent indexing.
        # ================================================================
        mask = np.isfinite(d) & np.isfinite(m) & (~mask_d) & (~mask_m)
        if not np.any(mask):
            raise RuntimeError("No valid data after masking.")
    
        d_fit = d[mask]   # raw data used in fitting
        m_fit = m[mask]   # model data used in fitting
    
        # ================================================================
        # 2. Scale time for Legendre stability (ONCE)
        # ================================================================
        # Legendre polynomials are numerically stable on [-1, 1].
        # scaling_time_Legendre() is assumed to perform this mapping.
        #
        # We evaluate it once on the full time grid and reuse it.
        # ================================================================
        tt_full = self.scaling_time_Legendre(t)
        tt = tt_full[mask]

        # ================================================================
        # 3. Select g0 candidates
        # ================================================================
        # Enforce:
        #   • positivity (physical gain)
        #   • uniqueness
        #   • finiteness
        g0_candidates = [
            g for g in dict.fromkeys(g0_candidates)
            if np.isfinite(g) and g > 0
        ]

        if not g0_candidates:
            raise RuntimeError("No valid g0 candidates.")
    
        # ================================================================
        # 4. Generate polynomial degree candidates
        # ================================================================
        # We search in a local neighborhood around the default degrees.
        # Ordering is by Manhattan distance from the default to
        # encourage early convergence.
        # ================================================================
        candidates = sorted(
            {
                (max(0, default_gain + dg),
                 max(0, default_base + db))
                for dg in range(-deg_range, deg_range + 1, deg_step)
                for db in range(-deg_range, deg_range + 1, deg_step)
            },
            key=lambda x: abs(x[0] - default_gain)
                        + abs(x[1] - default_base)
        )
    
        # ================================================================
        # 5. Cache Legendre basis evaluations (CORE SPEEDUP)
        # ================================================================
        # Polynomial evaluation dominates runtime if done repeatedly.
        #
        # We precompute:
        #   • P_k(tt) for gain terms (k >= 1)
        #   • P_j(tt) for baseline terms (j >= 0)
        #
        # This cache is reused for all g0 and degree combinations.
        # ================================================================
        max_gain = max(g for g, _ in candidates)
        max_base = max(b for _, b in candidates)
    
        P_gain = {
            k: legval(tt, [0]*k + [1])
            for k in range(1, max_gain + 1)
        }
    
        P_base = {
            j: legval(tt, [0]*j + [1])
            for j in range(0, max_base + 1)
        }
    
        # ================================================================
        # 5. Exhaustive but ordered model selection
        # ================================================================
        best_score = np.inf
        best_result = None
    
        for gain_deg, base_deg in candidates:
            for g0 in g0_candidates:
    
                # --------------------------------------------------------
                # Construct the linear least-squares system
                #
                # Unknowns:
                #   δg_k  (k = 1..gain_deg)
                #   b_j   (j = 0..base_deg)
                #
                # Model:
                #   d - g0*m ≈ Σ δg_k * (m * P_k) + Σ b_j * P_j
                # --------------------------------------------------------
                cols = []
    
                for k in range(1, gain_deg + 1):
                    cols.append(m_fit * P_gain[k])
    
                for j in range(0, base_deg + 1):
                    cols.append(P_base[j])
    
                if not cols:
                    continue
    
                X = np.vstack(cols).T
                y = d_fit - g0 * m_fit
    
                # --------------------------------------------------------
                # Solve linear least squares
                # --------------------------------------------------------
                coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    
                # --------------------------------------------------------
                # Reassemble full gain and baseline coefficients
                # --------------------------------------------------------
                c_gain = np.zeros(gain_deg + 1)
                c_gain[0] = g0
                c_gain[1:] = coeff[:gain_deg]
    
                c_base = coeff[gain_deg:]
    
                # --------------------------------------------------------
                # Evaluate on full time grid
                # --------------------------------------------------------
                gain = legval(tt_full, c_gain)
                baseline = legval(tt_full, c_base)
    
                # --------------------------------------------------------
                # Enforce physical gain positivity
                # --------------------------------------------------------
                if np.any(gain <= 0):
                    continue
    
                # --------------------------------------------------------
                # Model-space residual (what actually matters)
                # --------------------------------------------------------
                model = (d - baseline) / gain
                resid = m_fit - model[mask]
                score = np.mean(resid**2)
    
                # --------------------------------------------------------
                # Track best solution
                # --------------------------------------------------------
                if score < best_score:
                    best_score = score
                    best_result = (gain, baseline, c_gain, c_base)
    
        if best_result is None:
            raise RuntimeError("All candidate fits failed.")
    
        return best_result




