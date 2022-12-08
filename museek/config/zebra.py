from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_out_plugin',
        'museek.plugin.zebra_plugin',
    ]
)
InOutPlugin = ConfigSection(
    # block_name='1632760885',  # observation time stamp
    block_name='1631379874',  # observation time stamp
    # receiver_list=['m000h', 'm049h', 'm050h', 'm061h', 'm057h', 'm058h', 'm059h'],
    # receiver_list=None,
    receiver_list=['m000h'],
    do_use_noise_diode=True,
    token=None,  # archive token
    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    output_folder=None,  # directory to store results, if `None`, a 'results/' is chosen
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True
)
ZebraPlugin = ConfigSection(
    zebra_channel=488  # corresponds to 958 MHz
    # zebra_channel=3000  # corresponds to 958 MHz
)
