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
        # 'museek.plugin.meerkat_enthusiast_plugin'

    ]
)

InPlugin = ConfigSection(
    block_name='1631732038',  # observation time stamp
    receiver_list=[
        # 'm000h',
    #                'm000v',
    #                'm001h',
    #                'm001v',
    #                'm002h',
    #                'm002v',
    #                'm003h',
    #                'm003v',
    #                'm004h',
    #                'm004v',
    #                'm005h',
    #                'm005v',
    #                'm006h',
    #                'm006v',
    #                'm007h',
    #                'm007v',
    #                'm008h',
    #                'm008v',
    #                'm009h',
    #                'm009v',
                   'm010h',
                   'm010v',
    #                'm011h',
    #                'm011v',
    #                'm012h',
    #                'm012v',
    #                'm013h',
    #                'm013v',
    #                'm014h',
    #                'm014v',
    #                'm015h',
    #                'm015v',
    #                'm016h',
    #                'm016v',
    #                'm017h',
    #                'm017v',
    #                'm018h',
    #                'm018v',
    #                'm019h',
    #                'm019v',
    #                'm020h',
    #                'm020v',
    #                'm021h',
    #                'm021v',
    #                'm022h',
    #                'm022v',
    #                'm023h',
    #                'm023v',
    #                'm024h',
    #                'm024v',
    #                'm025h',
    #                'm025v',
    #                'm026h',
    #                'm026v',
    #                'm027h',
    #                'm027v',
    #                'm028h',
    #                'm028v',
    #                'm029h',
    #                'm029v',
    #                'm030h',
    #                'm030v',
    #                'm031h',
    #                'm031v',
    #                'm032h',
    #                'm032v',
    #                'm033h',
    #                'm033v',
    #                'm034h',
    #                'm034v',
    #                'm035h',
    #                'm036v',
    #                'm037h',
    #                'm037v',
    #                'm038h',
    #                'm038v',
    #                'm039h',
    #                'm039v',
                   'm040h',
                   'm040v',
    #                'm041h',
    #                'm041v',
    #                'm042h',
    #                'm042v',
    #                'm043h',
    #                'm043v',
    #                'm044h',
    #                'm044v',
    #                'm045h',
    #                'm045v',
                   'm046h',
                   'm046v',
    #                'm047h',
    #                'm047v',
    #                'm048h',
    #                'm048v',
    #                'm049h',
    #                'm049v',
    #                'm050h',
    #                'm050v',
    #                'm051h',
    #                'm051v',
    #                'm052h',
    #                'm052v',
    #                'm053h',
    #                'm053v',
    #                'm054h',
    #                'm054v',
    #                'm055h',
    #                'm055v',
    #                'm056h',
    #                'm056v',
    #                'm057h',
    #                'm057v',
    #                'm058h',
    #                'm058v',
                   'm059h',
                   'm059v',
    #                'm060h',
    #                'm060v',
    #                'm061h',
    #                'm061v',
    #                'm062h',
    #                'm062v',
                   'm063h',
                   'm063v'],
    #                'm064h',
    #                'm064v'],
                   
    # receiver_list=None,
    token=None,  # archive token
    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_store_context=False,
    context_folder=None,  # directory to store results, if `None`, 'results/' is chosen
)

OutPlugin = ConfigSection(
    output_folder= None # folder to store results, `None` means default location is chosen
)

AoflaggerPlugin = ConfigSection(
    n_jobs=64,
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


