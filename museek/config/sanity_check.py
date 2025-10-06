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
    block_name="1721666164",  # observation time stamp
    receiver_list=None,
    token="eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzIxNzQxNDgzLCJwcmVmaXgiOlsiMTcyMTY2NjE2NCJdLCJleHAiOjE3MjIzNDYyODMsInN1YiI6Im1ncnNhbnRvc0B1d2MuYWMuemEiLCJzY29wZXMiOlsicmVhZCJdfQ.usaFy6ytIwa0WIRGtGEhDa7ebXPw3SITvj16oAOUquAYlC1B-6KSA1oRgEuWPocsS0wpNgMrHtt2mRclviFHFw",
    data_folder=None,  # only relevant if `token` is `None`
    force_load_from_correlator_data=False,  # if `True`, the local `cache` folder is ignored
    # if `True`, the extracted visibilities, flags and weights are stored to disc for quicker access
    do_save_visibility_to_disc=False,
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
