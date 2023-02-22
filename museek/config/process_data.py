from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.out_plugin',
        # 'museek.plugin.zebra_remover_plugin',
        'museek.plugin.bandpass_plugin'
    ]
)

InPlugin = ConfigSection(
    block_name='1631379874',  # observation time stamp
    receiver_list=['m000h'],
    token=None,  # archive token
    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_use_noise_diode=True,
    do_store_context=True,
    context_folder=None,  # directory to store results, if `None`, a 'results/' is chosen
)

OutPlugin = ConfigSection(
    output_folder = None  # this means default location is chosen
)

ZebraRemoverPlugin = ConfigSection(
    reference_channel=3000,
    zebra_channels=range(350, 498),
)

BandpassPlugin = ConfigSection(
    target_channels=range(570, 765),
    zebra_channels=range(379, 498),
)
