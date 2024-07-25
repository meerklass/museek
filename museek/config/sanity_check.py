"""Sanity check museek configuration template."""

from ivory.utils.config_section import ConfigSection

# Order of the plugins to run
Pipeline = ConfigSection(
    plugins=[
        "museek.plugin.in_plugin",
        "museek.plugin.out_plugin",
        "museek.plugin.scan_track_split_plugin",
        "museek.plugin.sanity_check_observation_plugin",
    ]
)

# Plugin parameters -- these are meant to serve as templates.
# For sanity check the `block_name` and `token` parameters should be overridden
# by passing the desire values to the `mussek` command,
# e.g. `museek ----InPlugin-block-name=<value> --InPlugin-toke=<value> museek.config.sanity_check`.

InPlugin = ConfigSection(
    # -- Parameters to change --
    # 10-digit Block number (capture block ID) of the observation (required)
    block_name="1234567890",
    # RDB token from the archive (required)
    token=None,
    # Directory to store the context output.
    # If `None`, './results/' will be used.
    context_folder=None,
    # -- Unused or fixed parameters --
    # Directory containing the full correlator data to read from. Not relevent for
    # sanity check as we only need to read the RDB metadata providing the token.
    # But this can serve as a backup in the case that network connectivity to the
    # archive is not possible, in which case, the correlator data should be downloaded
    # to the specified `data_folder` and `token` set to `None`.
    data_folder=None,
    # Provide a list of recivers to read the data from,
    # or read from all receivers if set to None (default)
    receiver_list=None,
    # If `True, force re-reading from the correlator data in the `data_folde`,
    # ignoring the local `cache` made from previous InPlugin runs with
    # `do_save_visibility_to_disc=True`
    force_load_from_correlator_data=False,
    # If `True`, extracted the visibilities, flags and weights and store them in the
    # cache direcroy (museek/cache).
    do_save_visibility_to_disc=False,
    # Store context output of the InPlugin (the .pickle files)
    do_store_context=False,
)

OutPlugin = ConfigSection()

ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=False, do_store_context=True
)

SanityCheckObservationPlugin = ConfigSection(
    # The receiver index to use primarily for plots, relative to the `receiver_list`
    # of `In/OutPlugin`.
    reference_receiver_index=0,
    elevation_sum_square_difference_threshold=1e-2,  # in degrees^2
    elevation_square_difference_threshold=1e-3,  # in degrees^2
    # standard deviation threshold of individual dishes
    elevation_antenna_standard_deviation_threshold=1e-2,
    # minute, threshold of the time difference between sunset/sunrise and start/end time
    closeness_to_sunset_sunrise_threshold=30.0,
)
