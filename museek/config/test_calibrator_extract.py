import os

from museek.definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        "museek.plugin.in_plugin",
        "museek.plugin.noise_diode_flagger_plugin",
        "museek.plugin.known_rfi_plugin",
        "museek.plugin.rawdata_flagger_plugin",
        "museek.plugin.scan_track_split_plugin",
        "museek.plugin.extract_calibrators_plugin",
    ],
    # context=os.path.join('/idia/users/wkhu/calibration_results/noise_diode_2d/', '1675632179/gain_calibration_plugin.pickle')
)

InPlugin = ConfigSection(
    #    block_name='1675632179',  # observation time stamp
    block_name="1675021905",  # observation time stamp
    receiver_list=["m000h", "m000v"],
    # receiver_list=None,      # receivers to be processed, `None` means all available receivers is used
    token=None,  # archive token
    data_folder="/idia/projects/meerklass/MEERKLASS-1/SCI-20220822-MS-01/",  # only relevant if `token` is `None`
    # data_folder='/idia/projects/hi_im/SCI-20230907-MS-01/',  # only relevant if `token` is `None`
    force_load_auto_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    force_load_cross_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    do_save_visibility_to_disc=True,  # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_store_context=True,
    context_folder="/idia/users/msantos/museek",  # directory to store results, if `None`, 'results/' is chosen
)


RawdataFlaggerPlugin = ConfigSection(flag_lower_threshold=5.0, do_store_context=False)


KnownRfiPlugin = ConfigSection(
    gsm_900_uplink=None,
    gsm_900_downlink=(925, 960),
    gsm_1800_uplink=None,
    # gps=(1170, 1390),
    # extra_rfi=[(1524, 1630)],
    gps=None,
    extra_rfi=[(765, 775), (801, 811), (811, 821)],  # Vodacom  # MTN  # Telkom
)

ScanTrackSplitPlugin = ConfigSection(do_delete_unsplit_data=True, do_store_context=True)

ExtractCalibratorsPlugin = ConfigSection(
    # Essential parameters for target validation testing
    n_calibrator_observations=2,  # number of single dish calibrators present in the data. Only one before and one after is allowed.
    calibrator_names=[
        "HydraA",
        "PictorA",
    ],  # Corresponding calibrator names. Same order as data: [before,after]
    n_pointings=7,  # Valid consecutive tracks required for each calibrator
    max_gap_seconds=40.0,  # Maximum allowed time gap between calibrator track scans in seconds
    min_duration_seconds=20.0,  # Minimum scan duration in seconds to be considered valid
)
