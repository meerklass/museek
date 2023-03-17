import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.out_plugin',
        'museek.plugin.point_source_flagger_plugin',
        'museek.plugin.zebra_remover_plugin',
        # 'museek.plugin.bandpass_plugin'
    ]
)

InPlugin = ConfigSection(
    block_name='1638898468',
    # block_name='1652898082',
    # block_name='1677195529',
    # block_name='1631379874',  # observation time stamp
    # block_name='1632184922',  # observation time stamp
    # block_name='1632760885',
    receiver_list=['m000h',
                   'm000v',
                   'm008h',
                   'm008v',
                   'm012h',
                   'm012v',
                   'm014h',
                   'm014v',
                   'm019h',
                   'm019v',
                   'm028h',
                   'm028v'],
    token=None,  # archive token
    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_use_noise_diode=True,
    do_store_context=False,
    context_folder=None,  # directory to store results, if `None`, a 'results/' is chosen
)

OutPlugin = ConfigSection(
    output_folder=None  # this means default location is chosen
)

PointSourceFlaggerPlugin = ConfigSection(
    point_source_file_path=os.path.join(ROOT_DIR, 'data/radio_point_sources.txt'),
    angle_threshold=0.5
)

ApplyGainSolutionPlugin = ConfigSection(
    gain_file_path='/home/amadeus/Documents/fix/postdoc_UWC/work/MeerKLASS/calibration/download/level2/'
)

ZebraRemoverPlugin = ConfigSection(
    reference_channel=3000,
    zebra_channels=range(350, 498),
)

BandpassPlugin = ConfigSection(
    target_channels=range(570, 765),
    centre_coord=(79.95, -45.78),
    pointing_tolerance = .1
)
