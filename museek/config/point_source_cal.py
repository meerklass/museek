import os

from museek.definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
#        'museek.plugin.in_plugin',
#        'museek.plugin.noise_diode_flagger_plugin',
#        'museek.plugin.known_rfi_plugin',
#        'museek.plugin.rawdata_flagger_plugin',
#        'museek.plugin.scan_track_split_plugin',
#        'museek.plugin.extract_calibrators_plugin',
#        'museek.plugin.antenna_flagger_plugin',
#        'museek.plugin.aoflagger_tracking_plugin',
        'museek.plugin.point_source_calibration_plugin',
    ],
    context=os.path.join('/home/mgrsantos/projects/data/context/', '1675021905/aoflagger_tracking_plugin.pickle')

)


InPlugin = ConfigSection(
    block_name="1675021905",  # observation time stamp
#    receiver_list=['m000h','m000v'],
    receiver_list=['m000h','m000v','m005h','m005v','m023h','m023v'],
    # receiver_list=None,  # receivers to be processed, `None` means all available receivers is used
    token=None,  # archive token
    data_folder="/home/mgrsantos/projects/data/blocks",  # only relevant if `token` is `None`
    # data_folder='/idia/projects/hi_im/SCI-20230907-MS-01/',  # only relevant if `token` is `None`
    force_load_auto_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    force_load_cross_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    do_save_visibility_to_disc=True,  # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_store_context=False,
    context_folder="/home/mgrsantos/projects/data/context/",  # directory to store results, if `None`, 'results/' is chosen
    load_visibilities_auto=True,  # if `True`, auto-correlation visibilities are loaded in InPlugin
    load_visibilities_cross=False,  # if `True`, cross-correlation visibilities are loaded in InPlugin
    cache_folder="/home/mgrsantos/projects/data/cache",  # directory for cache files; defaults to ROOT_DIR/cache
    suppress_katpoint_warnings=True,  # if `True`, suppress noisy katpoint catalogue warnings
)

NoiseDiodeFlaggerPlugin = ConfigSection(
    verbose=0,
)


KnownRfiPlugin = ConfigSection(
    gsm_900_uplink=None,
    gsm_900_downlink=(925, 960),
    gsm_1800_uplink=None,
    #gps=(1170, 1390),
    #extra_rfi=[(1524, 1630)],
    gps=None,
    extra_rfi=[
    (544, 580),   # band edges
    (1015, 1088), # band edges
    (765, 778),   # Vodacom
    (801, 811),   # MTN
    (811, 821)    # Telkom
    ],
    verbose=0,
)

RawdataFlaggerPlugin = ConfigSection(
        flag_lower_threshold=5.0,
        do_store_context=False,
)


ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=True,  # erase from memory to free it up for next plugins
    do_store_context=True,
    keep_scan=False,
    keep_track=True,
)

ExtractCalibratorsPlugin = ConfigSection(
    # Essential parameters for target validation testing
    n_calibrator_observations=2,  # number of single dish calibrators present in the data. Only one before and one after is allowed.
    calibrator_names=['HydraA','PictorA'],  # Corresponding calibrator names. Same order as data: [before,after]
    n_pointings=7,  # Valid consecutive tracks required for each calibrator
    max_gap_seconds=40.0,  # Maximum allowed time gap between calibrator track scans in seconds
    min_duration_seconds=20.0,  # Minimum scan duration in seconds to be considered valid
    verbose=0,
)


AntennaFlaggerPlugin = ConfigSection(
    elevation_std_threshold=1e-2,  # standard deviation threshold of individual dishes elevation in degrees
    elevation_threshold=0.1,  # time points with elevation reading deviations exceeding this threshold are flagged
    outlier_threshold=0.1,  # antenna outlier threshold [degrees]
    elevation_flag_threshold=0.5, # if the fraction of flagged elevation exceeds this, all time dumps are flagged
    outlier_flag_threshold=0.5, # if the flag fraction of outlier flagging exceeds this, all time dumps are flagged
)

AoflaggerTrackingPlugin = ConfigSection(
    n_jobs=6,
    verbose=0,
    mask_type='vis',  # the data to which the flagger will be applied, ['vis', 'flag_fraction', 'rms', 'inverse', 'inverse_timemedian']
    first_threshold=0.1,  # First threshold value
    first_threshold_flag_fraction=0.25,  # First threshold value for aoflagger on flagged fracion
    threshold_scales=[0.5, 0.55, 0.62, 0.75, 1],
    smoothing_kernel=(2, 40),  # Smoothing, kernel window size in time and frequency axis
    smoothing_sigma=(1, 15),  # Smoothing, kernel sigma in time and frequency axis
    smoothing_kernel_flag_fraction=80,  # Smoothing, kernel window size in frequency axis
    smoothing_sigma_flag_fraction=30,  # Smoothing, kernel sigma in frequency axis
    struct_size=(1, 3),  # size of struct for dilation in time and frequency direction [pixels]
    channel_flag_threshold=0.6,
    time_dump_flag_threshold=0.6,
    flag_combination_threshold=1,
    do_store_context=True
)

PointSourceCalibrationPlugin = ConfigSection(
    n_jobs=6,
    verbose=0,
    flag_combination_threshold=1,
    on_source_separation_threshold_deg=0.1,
    beam_file_path='/home/mgrsantos/projects/data/MeerKAT_U_band_primary_beam_aa_highres.npz',
    receiver_models_dir=os.path.join(ROOT_DIR, 'model/receiver_models'),
    spillover_model_file=os.path.join(ROOT_DIR, 'model/MK_U_Tspill_AsBuilt_atm_mask.dat'),
    synch_model='s1',
    synch_nside=128,
    synch_fwhm_ref_deg=1.68,
    synch_fwhm_ref_freq_MHz=850.0,
    synch_freq_step_MHz=20.0,
    do_store_context=True
)
