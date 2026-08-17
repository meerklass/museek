"""Pipeline config for testing museek I/O."""

from ivory.utils.config_section import ConfigSection

CONTEXT_FOLDER = "./context"
DATA_FOLDER = "/idia/raw/meerklass/SCI-20230907-MS-01"

Pipeline = ConfigSection(
    plugins=[
        "museek.plugin.in_plugin",
    ],
)

InPlugin = ConfigSection(
    block_name="1716154890",
    token=None,  # Not loading from token
    receiver_list=None,  # Use all available receivers
    data_folder=DATA_FOLDER,
    force_load_auto_from_correlator_data=True,  # Don't use local "cache"
    force_load_cross_from_correlator_data=True,  # Don't use local "cache"
    do_save_visibility_to_disc=True,  # Extract and save visibilities, flags and weights
    do_store_context=True,
    context_folder=CONTEXT_FOLDER,  # Directory to store results
)
