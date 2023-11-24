import os

from definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.out_plugin',
        'museek.plugin.noise_diode_flagger_plugin',
        'museek.plugin.known_rfi_plugin',
        'museek.plugin.scan_track_split_plugin',
        'museek.plugin.standing_wave_fit_scan_plugin',
        # 'museek.plugin.standing_wave_correction_plugin'
    ],
    context=os.path.join(ROOT_DIR, 'results/1631379874/in_plugin.pickle')  # done
    # context=os.path.join(ROOT_DIR, 'results/1633970780/in_plugin.pickle')  # done
    # context=os.path.join(ROOT_DIR, 'results/1638898468/in_plugin.pickle')  # done
    # context=os.path.join(ROOT_DIR, 'results/1632760885/in_plugin.pickle')  # missing 008
    # context=os.path.join(ROOT_DIR, 'results/1634252028/in_plugin.pickle')  # done
    # context=os.path.join(ROOT_DIR, 'results/1632184922/in_plugin.pickle')  # missing 008
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
    footprint_ra_dec=None,
    do_store_parameters=True
    # footprint_ra_dec=((332.41, 357.85), (-35.35, -25.96))  # roughly a 2 % degree margin around the footprint
    # target_channels=range(570, 1410),  # 975 to 1151 MHz (yes HI & little RFI)
    # target_channels=range(2723, 2918),  # 1425 to 1465 MHz (no HI & no RFI)
)

ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=True,
    do_store_context=True
)


OutPlugin = ConfigSection(
    output_folder=None  # folder to store results, `None` means default location is chosen
)


KnownRfiPlugin = ConfigSection(
    gsm_900_uplink=(890, 915),
    gsm_900_downlink=(935, 960),
    gsm_1800_uplink=(1710, 1785),
    gps=(1170, 1390),
    extra_rfi=[(1524, 1630)]
)
