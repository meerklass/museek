import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        "museek.plugin.scan_track_split_plugin",
        # 'museek.plugin.standing_wave_fit_plugin',
        "museek.plugin.standing_wave_fit_scan_plugin",
        "museek.plugin.standing_wave_correction_plugin",
    ],
    context=os.path.join(ROOT_DIR, "results/1631379874/aoflagger_plugin.pickle"),
)

# StandingWaveFitPlugin = ConfigSection(
#     target_channels=range(570, 765),  # 975 to 1015 MHz (yes HI & no RFI)
#     # target_channels=range(570, 1410),  # 975 to 1151 MHz (yes HI & little RFI)
#     # target_channels=range(2723, 2918),  # 1425 to 1465 MHz (no HI & no RFI)
#     pointing_labels=['on centre 1',
#                      'off centre top',
#                      'on centre 2',
#                      'off centre right',
#                      'on centre 3',
#                      'off centre down',
#                      'on centre 4',
#                      'off centre left',
#                      'on centre 5']
# )

StandingWaveFitScanPlugin = ConfigSection(
    target_channels=range(570, 765),  # 975 to 1015 MHz (yes HI & no RFI)
    footprint_ra_dec=(
        (332.41, 357.85),
        (-35.35, -25.96),
    ),  # roughly a 2 % degree margin around the footprint
    # target_channels=range(570, 1410),  # 975 to 1151 MHz (yes HI & little RFI)
    # target_channels=range(2723, 2918),  # 1425 to 1465 MHz (no HI & no RFI)
)

ScanTrackSplitPlugin = ConfigSection(do_delete_unsplit_data=True, do_store_context=True)
