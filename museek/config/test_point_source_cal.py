import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.noise_diode_flagger_plugin',
        'museek.plugin.known_rfi_plugin',
        'museek.plugin.rawdata_flagger_plugin',
        'museek.plugin.scan_track_split_plugin',
        'museek.plugin.extract_calibrators_plugin',
        'museek.plugin.antenna_flagger_plugin',
        'museek.plugin.aoflagger_tracking_plugin',
        'museek.plugin.point_source_calibration_plugin',
    ],
    #context=os.path.join('/idia/users/wkhu/calibration_results/noise_diode_2d/', '1675632179/gain_calibration_plugin.pickle')

)

InPlugin = ConfigSection(
#    block_name='1675632179',  # observation time stamp
    block_name='1675021905',  # observation time stamp
    receiver_list=['m000h','m000v', 'm001h', 'm001v'],
    #receiver_list=None,      # receivers to be processed, `None` means all available receivers is used
    token=None,  # archive token
    data_folder='/idia/projects/meerklass/MEERKLASS-1/SCI-20220822-MS-01/',  # only relevant if `token` is `None`
    #data_folder='/idia/projects/hi_im/SCI-20230907-MS-01/',  # only relevant if `token` is `None`
    force_load_auto_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    force_load_cross_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    do_save_visibility_to_disc=True, # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access (stored in museek local cache folder)
    do_store_context=True,
    context_folder='/idia/users/msantos/museek',  # directory to store results, if `None`, 'results/' is chosen
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


ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=True,  # erase from memory to free it up for next plugins
    do_store_context=True
)

ExtractCalibratorsPlugin = ConfigSection(
    # Essential parameters for target validation testing
    n_calibrator_observations=2,  # number of single dish calibrators present in the data. Only one before and one after is allowed.
    calibrator_names=['HydraA','PictorA'],  # Corresponding calibrator names. Same order as data: [before,after]
    n_pointings=7,  # Valid consecutive tracks required for each calibrator
    max_gap_seconds=40.0,  # Maximum allowed time gap between calibrator track scans in seconds
    min_duration_seconds=20.0,  # Minimum scan duration in seconds to be considered valid
)


AntennaFlaggerPlugin = ConfigSection(
    elevation_std_threshold=1e-2,  # standard deviation threshold of individual dishes elevation in degrees
    elevation_threshold=0.1,  # time points with elevation reading deviations exceeding this threshold are flagged
    outlier_threshold=0.1,  # antenna outlier threshold [degrees]
    elevation_flag_threshold=0.5, # if the fraction of flagged elevation exceeds this, all time dumps are flagged
    outlier_flag_threshold=0.5, # if the flag fraction of outlier flagging exceeds this, all time dumps are flagged
)

AoflaggerTrackingPlugin = ConfigSection(
    n_jobs=8,
    verbose=0,
    mask_type='vis',  # the data to which the flagger will be applied, ['vis', 'flag_fraction', 'rms', 'inverse', 'inverse_timemedian']
    first_threshold=0.1,  # First threshold value
    first_threshold_flag_fraction=0.25,  # First threshold value for aoflagger on flagged fracion
    threshold_scales=[0.5, 0.55, 0.62, 0.75, 1],
    smoothing_kernel=(2, 40),  # Smoothing, kernel window size in time and frequency axis
    smoothing_sigma=(1, 15),  # Smoothing, kernel sigma in time and frequency axis
    smoothing_kernel_flag_fraction=80,  # Smoothing, kernel window size in frequency axis
    smoothing_sigma_flag_fraction=30,  # Smoothing, kernel sigma infrequency axis
    struct_size=(1, 3),  # size of struct for dilation in time and frequency direction [pixels]
    channel_flag_threshold=0.6,
    time_dump_flag_threshold=0.6,
    flag_combination_threshold=1,
    do_store_context=True
)

PointSourceCalibrationPlugin = ConfigSection(
    n_jobs=8,
    verbose=0,
    flag_combination_threshold=1,
    beam_file_path='/idia/projects/meerklass/beams/uhf/MeerKAT_U_band_primary_beam_aa_highres.npz',
    receiver_models_dir=os.path.join(ROOT_DIR, 'data/receiver_models'),
    spillover_model_file=os.path.join(ROOT_DIR, 'data/MK_U_Tspill_AsBuilt_atm_mask.dat'),
    do_store_context=True
)
