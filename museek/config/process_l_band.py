import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.out_plugin',
        'museek.plugin.noise_diode_flagger_plugin',
        'museek.plugin.known_rfi_plugin',
        'museek.plugin.scan_track_split_plugin',
        'museek.plugin.standing_wave_fit_plugin',
        'museek.plugin.standing_wave_fit_scan_plugin',
    ]
)

InPlugin = ConfigSection(
    block_name='1634252028',  # observation time stamp
    receiver_list=None,
    token=None,  # archive token
    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_store_context=True,
    context_folder=None,  # directory to store results, if `None`, 'results/' is chosen
)

OutPlugin = ConfigSection(
    output_folder=None  # folder to store results, `None` means default location is chosen
)

AntennaFlaggerPlugin = ConfigSection(
    elevation_threshold=1e-2,  # standard deviation threshold of individual dishes elevation in degrees
    outlier_threshold=0.1,  # degrees
)

PointSourceFlaggerPlugin = ConfigSection(
    point_source_file_path=os.path.join(ROOT_DIR, 'data/radio_point_sources.txt'),
    angle_threshold=0.5
)

ApplyExternalGainSolutionPlugin = ConfigSection(
    gain_file_path='/home/amadeus/Documents/fix/postdoc_UWC/work/MeerKLASS/calibration/download/level2/'
)

ZebraRemoverPlugin = ConfigSection(
    reference_channel=3000,
    zebra_channels=range(350, 498),
)

AoflaggerPlugin = ConfigSection(
    n_jobs=13,
    verbose=0,
    first_threshold=0.05,  # First threshold value
    threshold_scales=[0.5, 0.55, 0.62, 0.75, 1],
    smoothing_kernel=(20, 40),  # Smoothing, kernel window size in time and frequency axis
    smoothing_sigma=(7.5, 15),  # Smoothing, kernel sigma in time and frequency axis
    struct_size=(6, 6),  # size of struct for dilation in time and frequency direction [pixels]
    channel_flag_threshold=0.6,
    time_dump_flag_threshold=0.6,
    flag_combination_threshold=1,
    do_store_context=True
)

KnownRfiPlugin = ConfigSection(
    gsm_900_uplink=(890, 915),
    gsm_900_downlink=(935, 960),
    gsm_1800_uplink=(1710, 1785),
    gps=(1170, 1390),
    extra_rfi=[(1524, 1630)]
)

ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=True,
    do_store_context=True
)


StandingWaveFitPlugin = ConfigSection(
    target_channels=range(570, 765),  # 975 to 1015 MHz (yes HI & no RFI)
    pointing_labels=['on centre 1',
                     'off centre top',
                     'on centre 2',
                     'off centre right',
                     'on centre 3',
                     'off centre down',
                     'on centre 4',
                     'off centre left',
                     'on centre 5'],
    do_store_parameters=True
)

StandingWaveFitScanPlugin = ConfigSection(
    target_channels=range(570, 765),  # 975 to 1015 MHz (yes HI & no RFI)
    footprint_ra_dec=None,
    do_store_parameters=True
    # footprint_ra_dec=((332.41, 357.85), (-35.35, -25.96))  # roughly a 2 % degree margin around the footprint
    # target_channels=range(570, 1410),  # 975 to 1151 MHz (yes HI & little RFI)
    # target_channels=range(2723, 2918),  # 1425 to 1465 MHz (no HI & no RFI)
)