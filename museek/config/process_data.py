import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.out_plugin',
        'museek.plugin.antenna_flagger_plugin',
        'museek.plugin.aoflagger_plugin',
        'museek.plugin.point_source_flagger_plugin',
        'museek.plugin.zebra_remover_plugin',
        'museek.plugin.apply_external_gain_solution_plugin',
        'museek.plugin.bandpass_plugin'
    ]
)

InPlugin = ConfigSection(
    block_name='1631379874',  # observation time stamp
    receiver_list=['m000h',
                   'm000v',
                   'm001h',
                   'm001v',
                   'm002h',
                   'm002v',
                   'm008h',
                   'm008v',
                   'm012h',
                   'm012v',
                   'm013h',
                   'm013v',
                   'm014h',
                   'm014v',
                   'm031h',
                   'm031v',
                   'm032h',
                   'm032v',
                   'm037h',
                   'm037v',
                   'm022h',
                   'm022v',
                   'm026h',
                   'm026v',
                   'm052h',
                   'm052v',
                   'm063h',
                   'm063v'],
    token=None,  # archive token
    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_use_noise_diode=True,
    do_store_context=True,
    context_folder=None,  # directory to store results, if `None`, 'results/' is chosen
)

OutPlugin = ConfigSection(
    output_folder=None  # this means default location is chosen
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
    first_threshold=0.1,  # First threshold value
    threshold_scales=[0.5, 0.55, 0.62, 0.75, 1],
    smoothing_kernel=(20, 40),  # Smoothing, kernel window size in time and frequency axis
    smoothing_sigma=(7.5, 15),  # Smoothing, kernel sigma in time and frequency axis
    struct_size=(6, 1),  # size of struct for dilation in time and frequency direction [pixels]
    channel_flag_threshold=0.6,
    flag_combination_threshold=1
)

BandpassPlugin = ConfigSection(
    target_channels=range(570, 765),
    pointing_threshold=5.,
    n_pointings=5,
    n_centre_observations=3
)
