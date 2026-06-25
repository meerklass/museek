import os

from museek.definitions import ROOT_DIR
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        'museek.plugin.in_plugin',
        'museek.plugin.noise_diode_flagger_plugin',
        'museek.plugin.known_rfi_plugin',
        'museek.plugin.rawdata_flagger_plugin',
        'museek.plugin.scan_track_split_plugin',
        'museek.plugin.antenna_flagger_plugin',
        'museek.plugin.simulate_scan_plugin',
        # downstream science/calibration plugins run on the simulated scan_data, e.g.:
        # 'museek.plugin.point_source_flagger_plugin',
        # 'museek.plugin.aoflagger_plugin',
        # 'museek.plugin.gain_calibration_plugin',
    ],
)

InPlugin = ConfigSection(
    block_name="1675021905",
    receiver_list=['m000h', 'm000v', 'm005h', 'm005v', 'm023h', 'm023v'],  # small set for quick sim tests
    token=None,
    data_folder="/home/mgrsantos/projects/data/blocks",
    force_load_auto_from_correlator_data=False,
    force_load_cross_from_correlator_data=False,
    do_save_visibility_to_disc=True,
    do_store_context=False,
    context_folder="/home/mgrsantos/projects/data/context/",
    load_visibilities_auto=True,
    load_visibilities_cross=False,
    cache_folder="/home/mgrsantos/projects/data/cache",
    suppress_katpoint_warnings=True,
)

NoiseDiodeFlaggerPlugin = ConfigSection(
    verbose=0,
)

KnownRfiPlugin = ConfigSection(
    gsm_900_uplink=None,
    gsm_900_downlink=(925, 960),
    gsm_1800_uplink=None,
    gps=None,
    extra_rfi=[
        (544, 580),   # band edges
        (1015, 1088), # band edges
        (765, 778),   # Vodacom
        (801, 811),   # MTN
        (811, 821),   # Telkom
    ],
    verbose=0,
)

RawdataFlaggerPlugin = ConfigSection(
    flag_lower_threshold=5.0,
    do_store_context=False,
)

ScanTrackSplitPlugin = ConfigSection(
    do_delete_unsplit_data=True,
    do_store_context=False,
    keep_scan=True,    # we simulate the scan data
    keep_track=False,  # not needed: scan_track_split still sets TRACK_DATA=None and
                       # antenna_flagger guards `if track_data is not None`, so it just skips it
)

AntennaFlaggerPlugin = ConfigSection(
    elevation_std_threshold=1e-2,
    elevation_threshold=0.1,
    outlier_threshold=0.1,
    elevation_flag_threshold=0.5,
    outlier_flag_threshold=0.5,
)

SimulateScanPlugin = ConfigSection(
    beam_file_path='/home/mgrsantos/projects/data/MeerKAT_U_band_primary_beam_aa_highres.npz',
    receiver_models_dir=os.path.join(ROOT_DIR, 'museek/model/receiver_models'),
    noise_diodes_dir=os.path.join(ROOT_DIR, 'museek/model/noise_diodes'),
    spillover_model_file=os.path.join(ROOT_DIR, 'museek/model/MK_U_Tspill_AsBuilt_atm_mask.dat'),
    synch_model='s1',
    synch_nside=128,
    n_sim_freq=30,           # beam channels (within band) where the foreground is beam-convolved
    disc_radius_deg=8.0,
    include_noise_diode=True,
    n_jobs=6,                # joblib workers for Simeer's integrate_tod
    do_store_context=True,
    verbose=1,
)
