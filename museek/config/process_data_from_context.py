import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.scan_track_split_plugin',
        'museek.plugin.bandpass_plugin',
    ],
    context=os.path.join(ROOT_DIR, 'results/1631379874/aoflagger_plugin.pickle')
)

BandpassPlugin = ConfigSection(
    # target_channels=range(570, 765),
    target_channels=None,
    pointing_threshold=5.,
    n_pointings=5,
    n_centre_observations=3
)

ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=True,
    do_store_context=True
)
