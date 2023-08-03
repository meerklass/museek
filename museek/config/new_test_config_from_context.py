import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.meerkat_enthusiast_plugin'

    ],
    context=os.path.join(ROOT_DIR, 'results/1631379874/scan_track_split_plugin.pickle')
)