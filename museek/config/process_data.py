from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_out_plugin',
        'museek.plugin.zebra_remover_plugin',
    ]
)

InOutPlugin = ConfigSection(
    # block_name='1631379874',  # observation time stamp
    # block_name='1632184922',  # observation time stamp
    block_name='1631552188',  # observation time stamp   DATA NOT DOWNLOADED PROPERLY
    # block_name='1632760885',  # observation time stamp  DATA NOT DOWNLOADED PROPERLY
    receiver_list=['m000h', 'm000v', 'm007v','m007h'],
    token=None,  # archive token
    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    output_folder=None,  # directory to store results, if `None`, a 'results/' is chosen
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_use_noise_diode=True
)

ZebraRemoverPlugin = ConfigSection(
    reference_channel=3000,
    zebra_channels=range(379, 498),
    do_create_maps_of_frequency=False,
    satellite_free_dump_dict = {
        '1631379874': (1500, 'end'),
        '1632184922': (0, 500),
        '1632760885': (1500, 'end'),
        '1631552188': (0, 'end'),
        '1638898468': (0, 1500),
        '1637354605': (0, 1500)
    }
)
