from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_out_plugin',
        'museek.plugin.sanity_check_observation_plugin',
    ]
)

InOutPlugin = ConfigSection(
    block_name='1631379874',  # observation time stamp
    receiver_list=None,
    token=None,  # archive token
    data_folder='/idia/projects/hi_im/SCI-20210212-MS-01/',  # only relevant if `token` is `None`
    output_folder=None,  # directory to store results, if `None`, a 'results/' is chosen
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True
)

SanityCheckObservationPlugin = ConfigSection(
    # the receiver index to use primarily for plots, relative to the `receiver_list` of `InOutPlugin`.
    reference_receiver_index=0,
    elevation_sum_square_difference_threshold=1e-2,  # degrees^2
    elevation_square_difference_threshold=1e-3,  # degrees^2
    elevation_antenna_standard_deviation_threshold=1e-2,  # standard deviation threshold of individual dishes
)
