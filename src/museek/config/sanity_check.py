from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        "museek.plugin.in_plugin",
        "museek.plugin.out_plugin",
        "museek.plugin.scan_track_split_plugin",
        "museek.plugin.sanity_check_observation_plugin",
    ]
)

InPlugin = ConfigSection(
    block_name="1721666164",  # "Capture Block ID" of the observation
    receiver_list=None,
    token="eyJ0e...",
    data_folder=None,  # only relevant if `token` is `None`
    force_load_auto_from_correlator_data=False,
    force_load_cross_from_correlator_data=False,
    do_save_visibility_to_disc=False,  # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_store_context=False,
    context_folder=None,  # base directory to store results, if `None`, './results/' is chosen
)

OutPlugin = ConfigSection()

ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=False, do_store_context=False
)

SanityCheckObservationPlugin = ConfigSection(
    # the receiver index to use primarily for plots, relative to the `receiver_list` of `InOutPlugin`.
    reference_receiver_index=0,
    elevation_sum_square_difference_threshold=1e-2,  # degrees^2
    elevation_square_difference_threshold=1e-3,  # degrees^2
    elevation_antenna_standard_deviation_threshold=1e-2,  # standard deviation threshold of individual dishes
    closeness_to_sunset_sunrise_threshold=30.0,  # minute, threshold of the time difference
    # between sunset/sunrise and start/end time
)
