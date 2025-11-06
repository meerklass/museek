import os

from museek.definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.noise_diode_flagger_plugin',
        'museek.plugin.known_rfi_plugin',
        'museek.plugin.rawdata_flagger_plugin',
        'museek.plugin.scan_track_split_plugin',
        'museek.plugin.point_source_flagger_plugin',
        'museek.plugin.aoflagger_plugin',
        'museek.plugin.aoflagger_secondrun_plugin',
        'museek.plugin.antenna_flagger_plugin',
        'museek.plugin.noise_diode_plugin',
        'museek.plugin.gain_calibration_plugin',
        'museek.plugin.aoflagger_postcalibration_plugin',
        #'museek.plugin.aoflagger_cross_plugin',
        #'museek.plugin.single_dish_calibrator_plugin',
        #'museek.plugin.zebra_remover_plugin',
        #'museek.plugin.apply_external_gain_solution_plugin',
    ],
    #context=os.path.join('/idia/users/wkhu/calibration_results/noise_diode_2d/', '1675632179/gain_calibration_plugin.pickle')

)

InPlugin = ConfigSection(
    block_name='1675632179',  # observation time stamp
    #receiver_list=['m000h','m000v','m012h','m012v','m024h','m024v','m036h','m036v'],
    receiver_list=None,      # receivers to be processed, `None` means all available receivers is used
    token=None,  # archive token
    data_folder='/idia/projects/meerklass/MEERKLASS-1/SCI-20220822-MS-01/',  # only relevant if `token` is `None`
    #data_folder='/idia/projects/hi_im/SCI-20230907-MS-01/',  # only relevant if `token` is `None`
    force_load_auto_from_correlator_data=True,  # if `True`, the local `cache` folder is ignored
    force_load_cross_from_correlator_data=True,  # if `True`, the local `cache` folder is ignored
    do_save_visibility_to_disc=True, # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_store_context=True,
    context_folder=None,  # directory to store results, if `None`, 'results/' is chosen
)


AntennaFlaggerPlugin = ConfigSection(
    elevation_std_threshold=1e-2,  # standard deviation threshold of individual dishes elevation in degrees
    elevation_threshold=0.1,  # time points with elevation reading deviations exceeding this threshold are flagged
    outlier_threshold=0.1,  # antenna outlier threshold [degrees]
    elevation_flag_threshold=0.5, # if the fraction of flagged elevation exceeds this, all time dumps are flagged
    outlier_flag_threshold=0.5, # if the flag fraction of outlier flagging exceeds this, all time dumps are flagged
)


PointSourceFlaggerPlugin = ConfigSection(
    n_jobs=26,
    verbose=0,
    point_source_file_path='/idia/projects/meerklass/MEERKLASS-1/uhf_data/OT2023/radio_source_catalog/',
    beam_threshold=1., # times of the beam size around the point source to be masked 
    point_sources_match_flux=5.,  # flux threshold above which the point sources are selected, [Jy]
    point_sources_match_raregion=30., # the ra distance to the median of observed ra to select the point sources, [deg]
    point_sources_match_decregion=10., # the dec region to the median of observed dec to select the point sources [deg]
    beamsize=57.5,  # the beam fwhm used to smooth the Synch model [arcmin]
    beam_frequency=1500., # reference frequency at which the beam fwhm are defined [MHz]
)

ApplyExternalGainSolutionPlugin = ConfigSection(
    gain_file_path='/home/amadeus/Documents/fix/postdoc_UWC/work/MeerKLASS/calibration/download/level2/'
)

ZebraRemoverPlugin = ConfigSection(
    reference_channel=3000,
    zebra_channels=range(350, 498),
)

AoflaggerPlugin = ConfigSection(
    n_jobs=26,
    verbose=0,
    mask_type='vis',  # the data to which the flagger will be applied, ['vis', 'flag_fraction', 'rms', 'inverse', 'inverse_timemedian']
    first_threshold=0.1,  # First threshold value
    threshold_scales=[0.5, 0.55, 0.62, 0.75, 1],
    smoothing_kernel=(20, 40),  # Smoothing, kernel window size in time and frequency axis
    smoothing_sigma=(7.5, 15),  # Smoothing, kernel sigma in time and frequency axis
    struct_size=(3, 3),  # size of struct for dilation in time and frequency direction [pixels]
    channel_flag_threshold=0.6,
    time_dump_flag_threshold=0.6,
    flag_combination_threshold=1,
    do_store_context=True
)

AoflaggerCrossPlugin = ConfigSection(
    n_jobs=26,
    verbose=0,
    mask_type='vis',  # the data to which the flagger will be applied, ['vis', 'flag_fraction', 'rms', 'inverse', 'inverse_timemedian']
    first_threshold=3.5,  # First threshold value
    threshold_scales=[0.5, 0.55, 0.62, 0.75, 1],
    smoothing_kernel=(20, 40),  # Smoothing, kernel window size in time and frequency axis
    smoothing_sigma=(7.5, 15),  # Smoothing, kernel sigma in time and frequency axis
    struct_size=(3, 3),  # size of struct for dilation in time and frequency direction [pixels]
    channel_flag_threshold=0.6,
    time_dump_flag_threshold=0.6,
    flag_combination_threshold=1,
    do_store_context=True
)

AoflaggerSecondRunPlugin = ConfigSection(
    n_jobs=13,
    verbose=0,
    mask_type='vis',  # the data to which the flagger will be applied, ['vis', 'inverse'] for 1d aoflagger
    first_threshold=0.25,  # First threshold value
    threshold_scales=[0.5, 0.55, 0.62, 0.75, 1],
    smoothing_kernel=80,  # Smoothing, kernel window size in frequency axis
    smoothing_sigma=30,  # Smoothing, kernel sigma in frequency axis
    struct_size=(3, 3),  # size of struct for dilation in time and frequency direction [pixels]
    channel_flag_threshold=0.6,
    time_dump_flag_threshold=0.6,
    flag_combination_threshold=1,
    do_store_context=True
)

NoiseDiodePlugin = ConfigSection(
    n_jobs=13,
    verbose=0,
    flag_combination_threshold=1,
    zscoreflag_threshold = 5., # threshold (times of MAD) for flagging noise diode excess using modified zscore method
    polyflag_deg = 5, # degree of the polynomials used for fitting and flagging noise diode excess
    polyflag_threshold = 3., # threshold (times of MAD) for flagging noise diode excess using polynomials fit
    polyfit_deg = 5, # degree of the polynomials used for fitting flagged noise diode excess
    zscore_antenaflag_threshold = 10, # threshold (times of MAD) for flagging the rms of noise diode excess of receivers using modified zscore method
    noise_diode_excess_lowlim = 5., # threshold for flagging the mean value of noise diode excess of receivers
)


AoflaggerPostCalibrationPlugin = ConfigSection(
    n_jobs=13,
    verbose=0,
    first_threshold_rms=0.1,  # First threshold value
    first_threshold_flag_fraction=0.2,  # First threshold value
    threshold_scales=[0.5, 0.55, 0.62, 0.75, 1],
    smoothing_kernel_rms=80,  # Smoothing, kernel window size in frequency axis
    smoothing_sigma_rms=30,  # Smoothing, kernel sigma in frequency axis
    smoothing_kernel_flag_fraction=80,  # Smoothing, kernel window size in frequency axis
    smoothing_sigma_flag_fraction=30,  # Smoothing, kernel sigma in frequency axis
    struct_size=(3, 3),  # size of struct for dilation in time and frequency direction [pixels]
    channel_flag_threshold=0.4,
    time_dump_flag_threshold=0.4,
    flag_combination_threshold=1,
    poly_fit_degree=5, # degree of polynomials used to fit the data with the time median removed
    poly_fit_threshold=5., # threshold (times of MAD) of polynomials fitting flagging
    correlation_threshold_ant=0.5, # correlation coefficient threshold between calibrated data and synch model for excluding bad antennas.
    synch_model=['s1'], # list of str, the synch model used, see https://pysm3.readthedocs.io/en/latest/models.html#synchrotron
    nside=128,  #resolution parameter at which the synchrotron model is to be calculated
    beamsize=57.5,  # the beam fwhm used to smooth the Synch model [arcmin]
    beam_frequency=1500., # reference frequency at which the beam fwhm are defined [MHz]
    zscore_antenatempflag_threshold=5., # threshold for flagging the antennas based on their average temperature using modified zscore method
    do_store_context=True
)


KnownRfiPlugin = ConfigSection(
    gsm_900_uplink=None,
    gsm_900_downlink=(925, 960),
    gsm_1800_uplink=None,
    #gps=(1170, 1390),
    #extra_rfi=[(1524, 1630)],
    gps=None,
    extra_rfi=[
    (765, 775),  # Vodacom
    (801, 811),  # MTN
    (811, 821)   # Telkom
    ]
)

RawdataFlaggerPlugin = ConfigSection(
        flag_lower_threshold=5.0,
        do_store_context=False
)

GainCalibrationPlugin = ConfigSection(
        cali_method='corr', # method to do the calibration 'corr' or 'rms'
        synch_model=['s1'], # list of str, the synch model used, see https://pysm3.readthedocs.io/en/latest/models.html#synchrotron
        nside=128,  #resolution parameter at which the synchrotron model is to be calculated
        beamsize=57.5,  # the beam fwhm used to smooth the Synch model [arcmin]
        beam_frequency=1500., # reference frequency at which the beam fwhm are defined [MHz]
        frequency_high=1015., # high frequency cut for the scan data [MHz]
        frequency_low=580., # low frequency cut for the scan data [MHz]
        flag_combination_threshold=1,
        do_store_context=True,
        zscoreflag_threshold = 5., # threshold for flagging noise diode excess using modified zscore method
        polyflag_deg = 5, # degree of the polynomials used for fitting and flagging noise diode excess
        polyflag_threshold = 3., # threshold for flagging noise diode excess using polynomials fit
        polyfit_deg = 5, # degree of the polynomials used for fitting flagged noise diode excess
        window_movingmedian = 20, # The size of the window for the moving median calculation for frequency spectrum of noise diode signal
        nd_gausm_sigma = 20, # The size of the window for the Gaussian Smooth of Noise Diode Excess frequency spectrum
)

ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=True,
    do_store_context=True
)
