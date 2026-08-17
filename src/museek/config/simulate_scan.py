import os

from ivory.utils.config_section import ConfigSection

from museek.definitions import PACKAGE_DIR

Pipeline = ConfigSection(
    plugins=[
        "museek.plugin.in_plugin",
        "museek.plugin.noise_diode_flagger_plugin",
        "museek.plugin.known_rfi_plugin",
        "museek.plugin.rawdata_flagger_plugin",
        "museek.plugin.scan_track_split_plugin",
        "museek.plugin.antenna_flagger_plugin",
        "museek.plugin.read_calibrator_gains_plugin",  # optional: load a saved gain onto scan_data
        "museek.plugin.simulate_scan_plugin",
        # downstream science/calibration plugins run on the simulated scan_data, e.g.:
        # 'museek.plugin.point_source_flagger_plugin',
        # 'museek.plugin.aoflagger_plugin',
        # 'museek.plugin.gain_calibration_plugin',
    ],
)

InPlugin = ConfigSection(
    block_name="1675021905",
    receiver_list=[
        "m001h",
        "m001v",
        "m007h",
        "m007v",
        "m014h",
        "m014v",
        "m021h",
        "m021v",
        "m028h",
        "m028v",
    ],
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
        (544, 580),  # band edges
        (1015, 1088),  # band edges
        (765, 778),  # Vodacom
        (801, 811),  # MTN
        (811, 821),  # Telkom
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
    keep_scan=True,  # we simulate the scan data
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

# Loads a saved calibrator gain onto scan_data; SimulateScanPlugin then multiplies T by it.
# Remove this plugin from the Pipeline to simulate without a read gain.
ReadCalibratorGainsPlugin = ConfigSection(
    model_components_files=[
        "/home/mgrsantos/projects/data/context/1675021905/model_components.pkl"
    ],
    periods=None,
    verbose=0,
)

SimulateScanPlugin = ConfigSection(
    beam_file_path="/home/mgrsantos/projects/data/MeerKAT_U_band_primary_beam_aa_highres.npz",
    receiver_models_dir=os.path.join(PACKAGE_DIR, "model/receiver_models"),
    noise_diodes_dir=os.path.join(PACKAGE_DIR, "model/noise_diodes"),
    spillover_model_file=os.path.join(
        PACKAGE_DIR, "model/MK_U_Tspill_AsBuilt_atm_mask.dat"
    ),
    synch_model="s1",
    synch_nside=128,
    n_sim_freq=30,  # beam channels (within band) where the foreground is beam-convolved
    disc_radius_deg=8.0,
    include_noise_diode=True,
    include_cmb=True,  # CMB monopole (~2.7 K, Rayleigh-Jeans) added per channel
    # --- point sources (Phase 2). method: None | 'healpix' (cube+Simeer) | 'primary_beam' (analytic) ---
    point_source_method="primary_beam",
    point_source_catalog=os.path.join(PACKAGE_DIR, "model/1Jy_cat.txt"),
    primary_beam_file="/home/mgrsantos/projects/data/MeerKAT_U_band_primary_beam_aa_highres.npz",
    point_source_min_flux_Jy=1.0,  # cut evaluated at point_source_flux_cut_freq_MHz
    point_source_flux_cut_freq_MHz=800.0,  # band-middle frequency for the flux cut (consistent across sources)
    point_source_radius_deg=6.0,
    # --- gain (Phase 2): vis = gain * T_total, gain = read_calibrator_gain (CALIBRATOR_GAIN result)
    #     x smooth_poly(f) x (1 + standing_waves). Leave both synth terms None to use the read gain
    #     alone; with no read gain and both None the gain is unity (vis stays in Kelvin). ---
    use_read_gain=True,  # require & apply CALIBRATOR_GAIN; set False (and drop
    # ReadCalibratorGainsPlugin from the Pipeline) to skip the read gain
    gain_smooth_poly=[1.0],  # unity constant: read_gain x 1 x (1 + standing_waves)
    gain_standing_waves=[
        (0.5, 0.01, 0.0)
    ],  # 1% standing wave: 0.5 m path displacement, ~1.8 cycles across UHF band
    # --- HI signal (Phase 2). method: None | 'limtod' (Gaussian-field mock) | 'file' (user .npz cube) ---
    hi_method="limtod",  # Gaussian-field HI mock (Gaussian-beam smoothed)
    hi_n_freq=256,  # channels for the limtod field (interpolated onto the scan grid)
    hi_target_rms_mK=0.3,  # rescale the HI map to this RMS in mK; None -> use hi_gaussian_params['amp']
    hi_gaussian_params={
        "alpha": -1.0,
        "beta": 1.0,
        "xi": 0.01,
        "seed": 0,
    },  # "vaguely cosmological"
    hi_file=None,  # path to a user HI cube .npz (keys: maps (n_freq,n_pix), freq_MHz) for 'file'
    # --- noise (Phase 2): vis = gain * (1 + 1/f gain noise) * T_total * (1 + white radiometer noise) ---
    oneoverf_params=dict(f0=1.335e-5, fc=1.099e-3, alpha=2, white_n_variance=5e-6),
    white_noise_scale=1.0,  # scale on radiometer white noise N(0,1)/sqrt(dnu*tau); 0=off, 1.0=nominal, scales linearly
    noise_seed=0,  # base seed for reproducible 1/f + white noise
    n_jobs=6,  # joblib workers for Simeer's integrate_tod
    do_store_context=True,
    verbose=1,
)
