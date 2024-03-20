import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.out_plugin',
        'museek.plugin.scan_track_split_plugin',
        'museek.plugin.point_source_calibrator_plugin'
    ],
    context=os.path.join(ROOT_DIR, 'results/1675623808/in_plugin.pickle')
)

OutPlugin = ConfigSection(
    output_folder=None  # folder to store results, `None` means default location is chosen
)

ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=True,
    do_store_context=False
)