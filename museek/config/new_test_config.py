import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.out_plugin',
        'museek.plugin.noise_diode_flagger_plugin',
        # 'museek.plugin.known_rfi_plugin',
        # 'museek.plugin.scan_track_split_plugin',
        # 'museek.plugin.meerkat_enthusiast_plugin'
        'museek.plugin.known_rfi_plugin',
        'museek.plugin.aoflagger_plugin',
        'museek.plugin.scan_track_split_plugin'

    ]
)

InPlugin = ConfigSection(
    block_name='1631379874',  # observation time stamp
    receiver_list=['m000h',
                   'm000v',
                   'm008h',
                   'm008v',
                   'm013h',
                   'm013v'

                   ],
    token=None,  # archive token
    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_store_context=True,
    context_folder=None,  # directory to store results, if `None`, 'results/' is chosen
)

OutPlugin = ConfigSection(
    output_folder= None # folder to store results, `None` means default location is chosen
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


