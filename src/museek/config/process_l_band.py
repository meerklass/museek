import os

from ivory.utils.config_section import ConfigSection

from museek.definitions import PACKAGE_DIR

Pipeline = ConfigSection(
    plugins=[
        "museek.plugin.in_plugin",
        "museek.plugin.noise_diode_flagger_plugin",
        "museek.plugin.known_rfi_plugin",
        "museek.plugin.scan_track_split_plugin",
        "museek.plugin.antenna_flagger_plugin",
        "museek.plugin.aoflagger_plugin",
        # 'museek.plugin.point_source_flagger_plugin',
    ]
)

InPlugin = ConfigSection(
    block_name="1631379874",  # observation time stamp
    receiver_list=[
        "m000h",
        "m000v",
        "m008h",
        "m008v",
        "m013h",
        "m013v",
        "m028h",
        "m028v",
        "m037h",
        "m037v",
        "m063h",
        "m063v",
    ],
    token=None,  # archive token
    data_folder="/idia/projects/hi_im/SCI-20210212-MS-01/",  # only relevant if `token` is `None`
    force_load_auto_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    force_load_cross_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    load_visibilities_auto=True,  # auto-correlation visibilities are needed by the flaggers below
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_store_context=True,
    context_folder=None,  # directory to store results, if `None`, 'results/' is chosen
)

AntennaFlaggerPlugin = ConfigSection(
    elevation_std_threshold=1e-2,  # standard deviation threshold of individual dishes elevation in degrees
    elevation_threshold=0.1,  # time points with elevation reading deviations exceeding this threshold are flagged
    outlier_threshold=0.1,  # antenna outlier threshold [degrees]
    elevation_flag_threshold=0.5,  # if the fraction of flagged elevation exceeds this, all time dumps are flagged
    outlier_flag_threshold=0.5,  # if the flag fraction of outlier flagging exceeds this, all time dumps are flagged
)

PointSourceFlaggerPlugin = ConfigSection(
    point_source_file_path=os.path.join(PACKAGE_DIR, "model/radio_point_sources.txt"),
    beam_threshold=1.0,  # times of the beam size around the point source to be masked
    point_sources_match_flux=5.0,  # flux threshold above which the point sources are selected, [Jy]
    point_sources_match_raregion=30.0,  # the ra distance to the median of observed ra to select the point sources, [deg]
    point_sources_match_decregion=10.0,  # the dec region to the median of observed dec to select the point sources [deg]
    beamsize=57.5,  # the beam fwhm used to smooth the Synch model [arcmin]
    beam_frequency=1500.0,  # reference frequency at which the beam fwhm are defined [MHz]
)

AoflaggerPlugin = ConfigSection(
    n_jobs=13,
    verbose=0,
    mask_type="vis",  # the data to which the flagger will be applied
    first_threshold=0.05,  # First threshold value
    threshold_scales=[0.5, 0.55, 0.62, 0.75, 1],
    smoothing_kernel=(
        20,
        40,
    ),  # Smoothing, kernel window size in time and frequency axis
    smoothing_sigma=(7.5, 15),  # Smoothing, kernel sigma in time and frequency axis
    struct_size=(
        6,
        6,
    ),  # size of struct for dilation in time and frequency direction [pixels]
    channel_flag_threshold=0.6,
    time_dump_flag_threshold=0.6,
    flag_combination_threshold=1,
    do_store_context=True,
)

KnownRfiPlugin = ConfigSection(
    gsm_900_uplink=(890, 915),
    gsm_900_downlink=(935, 960),
    gsm_1800_uplink=(1710, 1785),
    gps=(1170, 1390),
    extra_rfi=[(1524, 1630)],
)

ScanTrackSplitPlugin = ConfigSection(do_delete_unsplit_data=True, do_store_context=True)
