from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.out_plugin',
        'museek.plugin.sanity_check_observation_plugin',
    ]
)

InPlugin = ConfigSection(
    block_name='1677195529',  # observation time stamp
    receiver_list=None,
    token='eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LT'
          'Eua2F0LmFjLnphIiwiaWF0IjoxNjc3NTgyNTY4LCJwcmVmaXgiOlsiMTY3NzE5NTUyOSJdLCJleHAiOjE2NzgxODczNjgsInN1YiI6Im'
          'FtYWRldXMud2lsZEBvdXRsb29rLmNvbSIsInNjb3BlcyI6WyJyZWFkIl19.91-EMklJ0ZSwiXUpquG3kiIL_Dn_D6a7uK9B26OTx9tCch'
          'LjF2kcWSx_YgdFR6NeXQ6ybVdavB0f6CGtm0cjeg',  # archive token

    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_use_noise_diode=False,
    do_store_context=True,
    context_folder=None,  # directory to store results, if `None`, a 'results/' is chosen
)
OutPlugin = ConfigSection(
    output_folder = None  # this means default location is chosen
)

SanityCheckObservationPlugin = ConfigSection(
    # the receiver index to use primarily for plots, relative to the `receiver_list` of `InOutPlugin`.
    reference_receiver_index=0,
    elevation_sum_square_difference_threshold=1e-2,  # degrees^2
    elevation_square_difference_threshold=1e-3,  # degrees^2
    elevation_antenna_standard_deviation_threshold=1e-2,  # standard deviation threshold of individual dishes
)
