from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.out_plugin',
        'museek.plugin.scan_track_split_plugin',
        'museek.plugin.sanity_check_observation_plugin',
    ]
)

InPlugin = ConfigSection(
    block_name='1675021905',  # observation time stamp
    receiver_list=None,
    token=None,  # archive token
    data_folder='/idia/raw/hi_im/SCI-20220822-MS-01/',  # only relevant if `token` is `None`
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=True,
    do_store_context=False,
    context_folder=None,  # directory to store results, if `None`, 'results/' is chosen
)
OutPlugin = ConfigSection(
    output_folder=None  # folder to store results, `None` means default location is chosen
)
ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=False,
    do_store_context=True
)
SanityCheckObservationPlugin = ConfigSection(
    # the receiver index to use primarily for plots, relative to the `receiver_list` of `InOutPlugin`.
    reference_receiver_index=0,
    elevation_sum_square_difference_threshold=1e-2,  # degrees^2
    elevation_square_difference_threshold=1e-3,  # degrees^2
    elevation_antenna_standard_deviation_threshold=1e-2,  # standard deviation threshold of individual dishes
    closeness_to_sunset_sunrise_threshold=30., # minute, threshold of the time difference 
    # between sunset/sunrise and start/end time
)
