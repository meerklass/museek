"""Helper functions for MuSEEK notebooks."""

import gc
import logging
import os
import pickle
from collections.abc import Callable, Sequence
from functools import reduce
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from numpy.typing import NDArray

from museek.enums.result_enum import ResultEnum

logger = logging.getLogger(__name__)

# Telecom downlink frequency ranges in MHz. These are used to mask out data.
DOWNLINK_RANGES = {
    "vodacom": (765, 775),
    "mtn": (801, 811),
    "telkom": (811, 821),
}

# Maps each raw flag name to the pipeline plugin that creates it.
FLAG_PLUGIN_MAP: dict[str, str] = {
    "SARAO": "in_plugin",
    "noise_diode_on": "noise_diode_flagger_plugin",
    "known_rfi": "known_rfi_plugin",
    "rawdata_low_value": "rawdata_flagger_plugin",
    "point_source": "point_source_flagger_plugin",
    "aoflagger": "aoflagger_plugin",
    "aoflagger_secondrun": "aoflagger_secondrun_plugin",
    "elevation_flag": "antenna_flagger_plugin",
    "outlier_antenna_flag": "antenna_flagger_plugin",
    "noise_diode_bad_behavior": "noise_diode_plugin",
}


def ant_names_to_ant_nums(
    ant_name: list[str] | npt.NDArray[np.str_],
) -> npt.NDArray[np.int_]:
    """Convert antenna names (e.g. 'm001') to integer numbers (e.g. 1)."""
    return np.char.lstrip(ant_name, "m").astype(int)


def read_pickle(pickle_file: str | Path):
    """Load the structure data from a MuSEEK's pickle file.

    ``pickle.load`` can execute arbitrary code, so ``pickle_file`` must come from a
    trusted source (i.e. MuSEEK's own pipeline output).
    """
    with open(pickle_file, "rb") as stream:
        struct = pickle.load(stream)
    return struct


def trim_frequencies(
    freq_array: NDArray, frequency_range: tuple[float, float], return_slice=False
) -> NDArray | tuple[NDArray, slice]:
    """Trim frequencies to the requested range."""

    if frequency_range[0] < freq_array.min() or frequency_range[1] > freq_array.max():
        raise ValueError(
            f"Requested frequency range {frequency_range} is outside the available range "
            f"{(freq_array.min(), freq_array.max())}"
        )
    start = int(np.searchsorted(freq_array, frequency_range[0], side="right"))
    stop = int(np.searchsorted(freq_array, frequency_range[1], side="left"))
    if start >= stop:
        raise ValueError(f"No frequencies remain inside range {frequency_range}")
    freq_slice = slice(start, stop)
    if return_slice:
        return freq_array[freq_slice], freq_slice
    else:
        return freq_array[freq_slice]


def circular_mean_deg(angles: NDArray) -> float:
    """Circular mean of angles in degrees (handles wraparound at 360/0)."""
    radians = np.deg2rad(angles)
    return float(
        np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
        % 360.0
    )


def wrap_to_nearest(values: NDArray, reference: float) -> NDArray:
    """Shift each value by a multiple of 360 deg to the branch nearest `reference`.

    Unlike ``np.unwrap``, this doesn't rely on ``values`` being in any
    particular order, so it's suitable for unordered/scattered points (e.g. a
    catalog subset) rather than only a continuous, time-ordered sequence.
    """
    return reference + (values - reference + 180.0) % 360.0 - 180.0


def unwrap_ra_deg(ra: NDArray) -> NDArray:
    """Unwrap a time-ordered RA track (deg) across the 360/0 branch cut.

    ``np.unwrap`` alone picks whichever branch minimises the jump from the
    first sample, which can leave the track sitting on the negative side
    (e.g. ``(353, 355, 359, 1, 3)`` -> ``(-7, -5, -1, 1, 3)``). Shift the
    whole (already internally-consistent) track up by a multiple of 360 so
    it instead continues past 360 (e.g. ``(353, 355, 359, 361, 363)``),
    which reads more naturally on a plot.
    """
    unwrapped = np.unwrap(ra, period=360.0)
    min_val = unwrapped.min()
    if min_val < 0:
        unwrapped = unwrapped + 360.0 * np.ceil(-min_val / 360.0)
    return unwrapped


def determine_scan_direction(ra: NDArray, dec: NDArray) -> Literal["rising", "setting"]:
    """Classify a scan block as rising or setting from a single antenna's RA/Dec track.

    Rising: RA decreases as Dec increases (RA and Dec anti-correlated).
    Setting: RA increases as Dec increases (RA and Dec positively correlated).
    ``ra``/``dec`` must be 1D, equal length, and non-empty (one antenna's track).
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if ra.shape != dec.shape:
        raise ValueError(f"ra and dec shapes must match, got {ra.shape} vs {dec.shape}")
    if ra.size == 0:
        raise ValueError("ra and dec must not be empty")
    ra_unwrapped = unwrap_ra_deg(ra)
    slope = np.polyfit(dec, ra_unwrapped, 1)[0]
    return "rising" if slope < 0 else "setting"


def load_point_sources(
    ra_center: float,
    dec_center: float,
    ra_radius: float = 30.0,
    dec_radius: float = 10.0,
    catalog_dir: str | Path | None = None,
) -> dict[str, NDArray]:
    """Return RA/Dec of 1 Jy and 5 Jy point sources within a sky region.

    Parameters
    ----------
    ra_center, dec_center : float
        Centre of the sky region in degrees.
    ra_radius, dec_radius : float
        Half-widths of the selection box in RA and Dec (degrees).
    catalog_dir : str or Path or None
        Directory containing the point source catalog
        If None, default to "/idia/projects/meerklass/MEERKLASS-1/museek/radio_source_catalog/".

    Returns
    -------
    dict with keys ``ra_1jy``, ``dec_1jy``, ``ra_5jy``, ``dec_5jy`` (NDArray).
    """
    if catalog_dir is None:
        catalog_dir = Path(
            "/idia/projects/meerklass/MEERKLASS-1/museek/radio_source_catalog/"
        )
    else:
        catalog_dir = Path(catalog_dir)

    # --- 1 Jy catalogue (pipe-delimited, has header) ---
    file_1jy = catalog_dir / "1jy_scat-V8_new.txt"
    with open(file_1jy) as fh:
        header_1jy = fh.readline().strip().lstrip("#").split("|")
    cat_1jy = np.genfromtxt(
        file_1jy,
        delimiter="|",
        names=header_1jy,
        skip_header=1,
        dtype=None,
        encoding=None,
        autostrip=True,
    )
    ra_1jy = np.asarray(cat_1jy["RA"], dtype=float)
    dec_1jy = np.asarray(cat_1jy["DEC"], dtype=float)

    # --- 5 Jy catalogue (pipe-delimited, no header; columns RA, DEC) ---
    file_5jy = catalog_dir / "5jy_scat-V0_new.txt"
    cat_5jy = np.atleast_2d(
        np.genfromtxt(file_5jy, delimiter="|", usecols=(1, 2), dtype=float)
    )
    ra_5jy = cat_5jy[:, 0]
    dec_5jy = cat_5jy[:, 1]

    def _select(ra_cat: NDArray, dec_cat: NDArray) -> tuple[NDArray, NDArray]:
        delta_ra = (ra_cat - ra_center + 180.0) % 360.0 - 180.0
        delta_dec = dec_cat - dec_center
        mask = (
            (np.abs(delta_ra) <= ra_radius)
            & (np.abs(delta_dec) <= dec_radius)
            & np.isfinite(ra_cat)
            & np.isfinite(dec_cat)
        )
        return ra_cat[mask], dec_cat[mask]

    ra_1jy_sel, dec_1jy_sel = _select(ra_1jy, dec_1jy)
    ra_5jy_sel, dec_5jy_sel = _select(ra_5jy, dec_5jy)

    logger.info(
        "Point sources loaded: %d × 1 Jy, %d × 5 Jy",
        len(ra_1jy_sel),
        len(ra_5jy_sel),
    )
    return {
        "ra_1jy": ra_1jy_sel,
        "dec_1jy": dec_1jy_sel,
        "ra_5jy": ra_5jy_sel,
        "dec_5jy": dec_5jy_sel,
    }


def _validate_receiver_antenna_ordering(receivers: NDArray, antennas: NDArray) -> None:
    """Raise ValueError if receivers are not ordered as [ant+'h', ant+'v', ...] for each antenna.

    This ordering is relied on when reshaping the raw `receivers` axis into
    `(antennas, feeds)` dimensions during dataset construction; the `receivers`
    axis itself is not persisted as a dataset coordinate.
    """
    if len(receivers) != 2 * len(antennas):
        raise ValueError(
            f"Expected 2 receivers per antenna, got {len(receivers)} receivers "
            f"and {len(antennas)} antennas."
        )
    expected = np.array(
        [suffix for ant in antennas for suffix in (ant + "h", ant + "v")]
    )
    if not np.array_equal(receivers, expected):
        mismatch = receivers[receivers != expected]
        raise ValueError(
            "Receivers are not ordered as [ant+'h', ant+'v', ...] for each antenna. "
            f"First mismatched receiver(s): {mismatch[:5]}"
        )


_VALID_FLAG_DIMS = frozenset(
    {"timestamps", "frequencies", "antennas", "feeds", "stokes"}
)

_FLAG_UFUNCS: dict[str, np.ufunc] = {
    "or": np.logical_or,
    "and": np.logical_and,
    "xor": np.logical_xor,
}

_XR_FLAG_UFUNCS: dict[str, Callable[[xr.DataArray, xr.DataArray], xr.DataArray]] = {
    "or": xr.ufuncs.logical_or,
    "and": xr.ufuncs.logical_and,
    "xor": xr.ufuncs.logical_xor,
}


def reduce_flags(
    flag_da: xr.DataArray,
    output_dims: tuple[str, ...],
    operator: Literal["or", "and", "xor"],
) -> xr.DataArray:
    """Reduce a boolean flag DataArray by applying a logical operator.

    Collapses every dimension in the input that is absent from ``output_dims``,
    applying ``operator`` uniformly along each such axis in turn (e.g. reducing
    ``"feeds"`` collapses H/V, reducing ``"stokes"`` collapses Stokes
    parameters, exactly like reducing ``"timestamps"`` or ``"frequencies"``).

    This operates *within* a single flag array, folding away its own dims — the
    counterpart to :func:`combine_flags`, which joins *separate* flag variables
    elementwise without touching dims. :func:`select_and_flag` chains both: first
    :func:`combine_flags` to merge named flags into one, then this function (if
    needed) to collapse whatever dims that merged flag has beyond the target
    variable's own dims.

    Parameters
    ----------
    flag_da : xr.DataArray
        Input boolean flag. Dims must be a subset of
        ``{"timestamps", "frequencies", "antennas", "feeds", "stokes"}``.
    output_dims : tuple[str, ...]
        Dimensions to retain in the output; must be a subset of the input's
        dims. The output DataArray will have its dims ordered to match this
        tuple.
    operator : {"or", "and", "xor"}
        Logical operator applied for every reduction step.

    Returns
    -------
    xr.DataArray
        Boolean DataArray with ``dims == output_dims``.

    Raises
    ------
    ValueError
        If ``output_dims`` contains an unknown dim, or a dim not present in
        ``flag_da``.
    """
    output_set = set(output_dims)
    input_dims = set(flag_da.dims)

    # --- Validate output_dims ---
    invalid = output_set - _VALID_FLAG_DIMS
    if invalid:
        raise ValueError(
            f"Unknown output_dims: {sorted(invalid)}. "
            f"Allowed values: {sorted(_VALID_FLAG_DIMS)}"
        )
    unreachable = output_set - input_dims
    if unreachable:
        raise ValueError(
            f"Output dims {sorted(unreachable)} are not present in "
            f"input dims {sorted(input_dims)}."
        )

    ufunc = _FLAG_UFUNCS[operator]
    result = flag_da

    # --- Reduce every dim not in output_dims ---
    for dim in [d for d in result.dims if d not in output_set]:
        axis = result.dims.index(dim)
        reduced = ufunc.reduce(result.values, axis=axis)
        new_dims = [d for d in result.dims if d != dim]
        new_coords = {k: v for k, v in result.coords.items() if k in new_dims}
        result = xr.DataArray(reduced, dims=new_dims, coords=new_coords)

    # --- Transpose to match requested output_dims order ---
    if output_dims and list(result.dims) != list(output_dims):
        result = result.transpose(*output_dims)

    return result


def combine_flags(
    ds: xr.Dataset,
    flags: str | Sequence[str] | xr.DataArray | Sequence[xr.DataArray],
    combine: Literal["or", "and", "xor"] = "or",
) -> xr.DataArray:
    """Resolve flag name(s) and/or DataArray(s) and combine them into one boolean flag.

    Joins *separate* flag variables (e.g. ``SARAO``, ``known_rfi``,
    ``rawdata_low_value``) elementwise via xarray's own broadcasting/alignment —
    it never touches or reduces any dimension. This is the counterpart to
    :func:`reduce_flags`, which instead collapses dims *within* a single
    already-resolved flag array. :func:`select_and_flag` calls this first (to
    merge named flags into one), then :func:`reduce_flags` if that merged flag
    still has dims the target variable doesn't.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to resolve string flag names against.
    flags : str, DataArray, or a sequence of either
        Flags to combine. Strings are looked up as ``ds[name]``.
    combine : {"or", "and", "xor"}
        Logical operator applied elementwise across the resolved flags.

    Returns
    -------
    xr.DataArray
        A single boolean flag. If only one flag is given, it is returned unchanged.
    """
    if isinstance(flags, (str, xr.DataArray)):
        flags = [flags]
    resolved = [ds[f] if isinstance(f, str) else f for f in flags]
    if len(resolved) == 1:
        return resolved[0]
    ufunc = _XR_FLAG_UFUNCS[combine]
    return reduce(ufunc, resolved)


def select_and_flag(
    ds: xr.Dataset,
    var: str,
    flags: str | Sequence[str] | xr.DataArray | Sequence[xr.DataArray] | None = None,
    combine: Literal["or", "and", "xor"] = "or",
    sel: dict | None = None,
    isel: dict | None = None,
    nearest: dict | None = None,
    dropna_dim: str | None = None,
) -> xr.DataArray:
    """Select, flag-mask, and optionally dropna a dataset variable.

    This handles the common notebook pattern of slicing a variable, combining one or
    more boolean flags, masking with those flags, and dropping flagged points — but
    intentionally does not perform any reduction (mean, median, etc.); callers do that
    on the returned DataArray.

    Internally chains the module's two flag helpers: :func:`combine_flags` first
    (joins the named/given flags elementwise into one), then :func:`reduce_flags`
    only if that combined flag still carries dims absent from the (post-selection)
    variable — e.g. masking ``ra``/``dec`` (dims ``timestamps, antennas``) with a
    flag that still has ``frequencies``/``feeds`` collapses those away via OR
    before the final ``.where()``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to select from.
    var : str
        Name of the data variable in ``ds`` to select and mask.
    flags : str, DataArray, or a sequence of either, optional
        Flag(s) to combine (via :func:`combine_flags`) and apply as a mask. If the
        combined flag has dims not present in the (post-``sel``/``isel``/``nearest``)
        variable, those extra dims are collapsed via :func:`reduce_flags` using
        ``combine`` as the reduction operator.
    combine : {"or", "and", "xor"}
        Logical operator used both to combine multiple ``flags`` and to collapse any
        flag dims not present in ``var``.
    sel, isel : dict, optional
        Passed to ``.sel()``/``.isel()`` on both the variable and (for the keys
        present in its dims) the resolved flags, so both stay aligned.
    nearest : dict, optional
        Like ``sel``, but applied in a separate ``.sel(..., method="nearest")`` call.
        Kept separate from ``sel`` because xarray cannot mix an exact-match indexer
        (e.g. a list of antenna names) with ``method="nearest"`` in a single call.
    dropna_dim : str, optional
        If given, drop NaNs along this dimension after masking.

    Returns
    -------
    xr.DataArray
    """

    def _apply_selectors(da: xr.DataArray) -> xr.DataArray:
        if isel:
            keys = {k: v for k, v in isel.items() if k in da.dims}
            if keys:
                da = da.isel(**keys)
        if sel:
            keys = {k: v for k, v in sel.items() if k in da.dims}
            if keys:
                da = da.sel(**keys, drop=True)
        if nearest:
            keys = {k: v for k, v in nearest.items() if k in da.dims}
            if keys:
                da = da.sel(**keys, method="nearest", drop=True)
        return da

    da = _apply_selectors(ds[var])

    if flags is not None:
        flag_da = _apply_selectors(combine_flags(ds, flags, combine=combine))
        extra_dims = [d for d in flag_da.dims if d not in da.dims]
        if extra_dims:
            output_dims = tuple(d for d in flag_da.dims if d not in extra_dims)
            flag_da = reduce_flags(flag_da, output_dims=output_dims, operator=combine)
        da = da.where(~flag_da)

    if dropna_dim is not None:
        da = da.dropna(dim=dropna_dim)

    return da


def load_context_to_ds(
    pickle_file: str | Path,
    cache_file: str | Path | None = None,
    skip_cache: bool = False,
    frequency_range: Literal["auto"] | tuple[float, float] = "auto",
    extra_attrs: dict | None = None,
) -> xr.Dataset:
    """Load MuSEEK pickle data as a single dataset with scan and calibrated visibility.

    ``extra_attrs``, if given, is added to ``ds.attrs`` last, after all attributes
    computed by this function, so it can be used to override them if needed.

    The returned dataset is trimmed to the requested frequency range for both
    scan and calibrated visibility data and stores the original untrimmed
    frequency axis in ``ds.attrs["original_frequencies"]``.

    Flags
    -----
    Raw flags (``raw_flags_<name>``, dims ``timestamps, frequencies, antennas, feeds``
    unless noted), in pipeline execution order. ``raw_flags_combined`` is the OR of all
    of these except ``point_source``, which is a calibration-time prior rather than a
    persistent data-quality flag:

    - ``SARAO``: RFI/data-quality flags already embedded in the SARAO archive data
      itself (loaded as-is, not computed by MuSEEK).
    - ``noise_diode_on``: timestamps where the calibration noise diode is firing (or
      in its on/off transition).
    - ``known_rfi``: frequency channels within pre-defined, catalogued terrestrial RFI
      bands (e.g. GSM/GPS), flagged for the whole observation regardless of the data.
    - ``rawdata_low_value``: raw visibility samples below a configured minimum
      threshold, indicating receiver dropouts or invalid correlator readings.
    - ``point_source`` (dims ``timestamps, frequencies, antennas``, no ``feeds``):
      samples where the beam points at or near a known bright point source; used as a
      prior mask by ``aoflagger_plugin`` and not included in ``raw_flags_combined``.
    - ``aoflagger``: RFI detected by AOFlagger's SumThreshold algorithm run on the raw
      visibility, per receiver.
    - ``aoflagger_secondrun``: a second AOFlagger pass run on the time-averaged
      flagged-fraction spectrum, catching channels already heavily flagged.
    - ``elevation_flag``: per-antenna timestamps where elevation deviates too far from
      that antenna's median (mispointing/slew); antennas with unstable elevation or too
      high a flagged fraction are flagged entirely.
    - ``outlier_antenna_flag``: at each timestamp, antennas whose pointing is a
      statistical outlier relative to the rest of the array.
    - ``noise_diode_bad_behavior``: antennas/receivers whose noise-diode excess signal
      is poorly fit over time, an outlier in fit residuals, or too low to trust for
      diode-based calibration.

    Calibrated flags (``cal_flags_<name>``, applied to ``cal_vis``), built by the
    post-calibration AOFlagger stage. ``cal_flags_combined`` is the final mask actually
    applied to the calibrated visibility (the OR of all of the below):

    - ``HH_VV_combined``: cellphone-band RFI (GSM/GPS) masking applied at the
      calibrated-data stage, combined with the mask carried forward from raw flagging.
    - ``temp_outlier_flag``: antennas whose median calibrated temperature is a
      statistical outlier relative to the other antennas.
    - ``temp_fluctuation_flag``: antennas whose calibrated temperature standard
      deviation is a statistical outlier (unusually noisy/unstable gain).
    - ``polynomial_fit_flag``: per-timestamp frequency-domain outliers relative to a
      low-order polynomial fit of the time-median-subtracted spectrum.
    - ``synch_correlation_flag``: antennas whose calibrated sky map correlates poorly
      with a synchrotron sky model, indicating a bad or miscalibrated antenna.
    - ``freq_flaggedfraction_flag``: frequency channels flagged in full because their
      already-flagged fraction (across good antennas/times) exceeds a threshold.
    """

    # Check first if cache file exists. If it does, open and return it skipping the rest
    cache_file_path = Path(cache_file).resolve() if cache_file is not None else None
    if cache_file_path is not None:
        cache_file_path = cache_file_path.parent / f"{cache_file_path.stem}.nc"

        # Warn and check cache availability
        if skip_cache:
            logger.warning("skip_cache=True: skipping cache load")
        elif cache_file_path.exists():
            logger.info(f"Loading data from cache: {cache_file_path}")
            return xr.open_dataset(cache_file_path)
        else:
            logger.warning(f"Cache file not found: {cache_file_path}")

        # Ensure cache directory exists and is writable for later writing
        if not cache_file_path.exists():
            try:
                cache_file_path.parent.mkdir(parents=True, exist_ok=True)
                if not os.access(cache_file_path.parent, os.W_OK):
                    logger.warning(
                        f"Cache directory is not writable: {cache_file_path.parent}"
                    )
            except OSError as e:
                logger.warning(
                    f"Cannot access cache directory {cache_file_path.parent}: {e}"
                )

    logger.info(f"Loading data from pickle file: {pickle_file}")
    data = read_pickle(pickle_file)

    # TODO: Need to check that the required Enum exists in the data structure,
    # otherwise raise an error

    # Block level metadata (small, keep in memory)
    logger.info("Extracting metadata")
    block_name = data.get(ResultEnum.BLOCK_NAME).result
    obs_date = data.get(ResultEnum.OBSERVATION_DATE).result.isoformat()
    scan_start = data.get(ResultEnum.SCAN_OBSERVATION_START).result
    scan_end = data.get(ResultEnum.SCAN_OBSERVATION_END).result

    # Need to extract scan data to get frequency, timestamps, and receivers.
    # This is very large, so we avoid making copies and will delete it after extracting
    # what we need.
    scan_data = data.get(ResultEnum.SCAN_DATA).result

    # Extract frequencies and trim to requested range
    # freq_select is technically part of calibrated data, but we need it to trim the
    # scan data to the same frequency range as the calibrated data."
    original_frequencies = scan_data.frequencies.array.squeeze() / 1e6  # MHz
    freq_select = data.get(ResultEnum.FREQ_SELECT).result.squeeze() / 1e6  # MHz
    if frequency_range == "auto":
        min_freq = max(original_frequencies.min(), freq_select.min())
        max_freq = min(original_frequencies.max(), freq_select.max())
    else:
        min_freq, max_freq = frequency_range
    logger.info(f"Frequencies will be trimmed to range: {min_freq} - {max_freq} MHz")
    frequencies, frequency_slice = trim_frequencies(
        original_frequencies, (min_freq, max_freq), return_slice=True
    )
    freq_select, freq_select_slice = trim_frequencies(
        freq_select, (min_freq, max_freq), return_slice=True
    )
    # Validate that the trimmed scan and calibrated frequency axes match.
    if not np.array_equal(frequencies, freq_select):
        raise ValueError(
            "Scan and calibrated frequency axes do not match after trimming\n"
            f"Scan frequencies shape: {frequencies.shape}\n"
            f"Calibrated frequencies shape: {freq_select.shape}\n"
            f"Scan frequencies: {frequencies}\n"
            f"Calibrated frequencies: {freq_select}"
        )

    # Extract timestamps. This is in UNIX epoch.
    timestamps = scan_data.timestamps.array.squeeze()

    # Receivers are antennas with feed name, e.g. m001h, m001v, and are stored as
    # a list of Receiver objects, so we extract the names
    receivers = np.asarray([rec.name for rec in scan_data.receivers], dtype=str)

    # Antennas are the unique antenna names, e.g. m001, m002, and are stored as
    # a list of strings in the _antenna_name_list attribute,
    antennas = np.asarray(scan_data._antenna_name_list, dtype=str)

    # Validate that receivers are ordered as [ant+'h', ant+'v', ...] for each antenna.
    # This ordering is relied on below to reshape the trailing receivers axis of raw
    # arrays into (antennas, feeds) via a plain numpy reshape (a view, not a copy).
    # The `receivers` array itself is not persisted as a dataset coordinate.
    _validate_receiver_antenna_ordering(receivers, antennas)
    feeds = np.asarray(["h", "v"], dtype=str)

    # Extract raw scan vis and trim frequencies
    logger.info("Extracting raw visibility and flags")
    raw_vis = scan_data.visibility.array  # shape (timestamps, frequencies, receivers)
    raw_vis = raw_vis[:, frequency_slice, :]

    # Extract raw flags, avoiding the expensive FlagList.array class property, which
    # constructs 4D array, by instead accessing individual FlagElement objects.
    flag_elements = scan_data.flags._flags
    # The flag names are stored as a list of strings in FLAG_NAME_LIST enum with order
    # matching the order of the FlagElement objects in flag_elements.
    # TODO: FLAG_NAME_LIST will be deprecated once point source calibration is merged.
    flag_names = [f"raw_flags_{k}" for k in data.get(ResultEnum.FLAG_NAME_LIST).result]

    # Extract RA and Dec
    ra = scan_data.right_ascension.array.squeeze()
    dec = scan_data.declination.array.squeeze()

    # Determine scan direction (rising/setting) from the first antenna whose ra/dec
    # track is valid, in case an antenna's track is degenerate (e.g. empty/mismatched).
    scan_direction = None
    for ant_idx in range(ra.shape[1]):
        try:
            scan_direction = determine_scan_direction(ra[:, ant_idx], dec[:, ant_idx])
            break
        except ValueError:
            continue
    if scan_direction is None:
        raise ValueError(
            "Could not determine scan direction: no antenna had a valid ra/dec track"
        )

    # Delete scan_data and garbage collect to free memory.
    del scan_data
    gc.collect()

    # === PROCESS CALIBRATED DATA ===
    logger.info("Extracting calibrated visibility and flags")
    # CALIBRATED_VIS is a masked array, keeping it will make the data format
    # inconsistent since scan data store data and flags as separate Numpy arrays.
    # Thus, we only extract the data.
    calibrated_vis_data = (
        data.get(ResultEnum.CALIBRATED_VIS).result.data[:, freq_select_slice, :] / 1e6
    )  # Kelvin

    cal_flag_names = data.get(ResultEnum.CALIBRATED_VIS_FLAG_NAME_LIST).result
    r_vis_synch_ant = data.get(ResultEnum.CORRELATION_COEFFICIENT_VIS_SYNCH_ANT).result

    # Extract calibrated flags data before deleting to preserve reference
    # Note: cal_flags is a dictionary with keys corresponding to cal_flag_names and
    # values as numpy arrays of different shapes:
    # HH_VV_combined <class 'numpy.ndarray'> (2800, 3275, 61) bool
    # temp_outlier_flag <class 'numpy.ndarray'> (61,) bool
    # temp_fluctuation_flag <class 'numpy.ndarray'> (61,) bool
    # polynomial_fit_flag <class 'numpy.ndarray'> (2800, 3275, 61) bool
    # synch_correlation_flag <class 'numpy.ndarray'> (61,) bool
    # freq_flaggedfraction_flag <class 'numpy.ndarray'> (3275,) bool
    cal_flags = data.get(ResultEnum.CALIBRATED_VIS_FLAG).result
    # The final calibration flags, which include full-band flags when the flag fraction
    # is higher than a certain percentage, is only stored in the mask of CALIBRATED_VIS
    # and is not in CALIBRATED_VIS_FLAG
    cal_flags_combined = data.get(ResultEnum.CALIBRATED_VIS).result.mask[
        :, freq_select_slice, :
    ]

    # Point-source flags are stored separately (not in FLAG_NAME_LIST) because they are
    # used as a prior mask inside aoflagger_plugin, not as a persistent flag layer.
    point_source_flag = data.get(ResultEnum.POINT_SOURCE_FLAG).result

    # Build raw_flag_name_list: FLAG_NAME_LIST order + "point_source" inserted after
    # "rawdata_low_value" to reflect pipeline execution order.
    raw_flag_name_list = list(data.get(ResultEnum.FLAG_NAME_LIST).result)
    try:
        insert_idx = raw_flag_name_list.index("rawdata_low_value") + 1
    except ValueError:
        insert_idx = len(raw_flag_name_list)
    raw_flag_name_list.insert(insert_idx, "point_source")

    del data
    gc.collect()

    # Build dataset incrementally
    logger.info("Building XArray dataset")

    n_ant = len(antennas)

    def _split_feeds(arr: NDArray) -> NDArray:
        """Reshape a (..., 2*n_ant) receivers-last array into (..., n_ant, 2) feeds.

        This is a view, not a copy: receivers is the trailing/fastest-varying axis,
        validated as exactly 2*n_ant long with [h, v] interleaved per antenna, so
        splitting it into (antennas, feeds) never requires touching the data.
        """
        return arr.reshape(arr.shape[:-1] + (n_ant, 2))

    # Initialize with scan data, keeping the receivers axis until every raw array
    # (including the OR-combined raw_flags_combined below) has been built, then
    # reshape all of them from (..., receivers) to (..., antennas, feeds) in one pass.
    scan_dims = ("timestamps", "frequencies", "receivers")
    data_vars = {"raw_vis": (scan_dims, raw_vis)}

    # Add scan flags incrementally, trimming frequencies to match the requested range,
    # noting that the flag arrays are all shape (timestamps, frequencies, receivers),
    # so we slice the frequency axis.
    for flag_elem, name in zip(flag_elements, flag_names):
        data_vars[name] = (scan_dims, flag_elem.array[:, frequency_slice, :])
    # Build the combined raw flags. This yields the same results as calling
    # scan_data.flags.combine(), but the operation does not make copies of flag arrays
    # in the data_vars dictionary that we just build
    raw_flags_combined = np.logical_or.reduce(
        [v[1] for k, v in data_vars.items() if k.startswith("raw_flags_")]
    )
    data_vars["raw_flags_combined"] = (scan_dims, raw_flags_combined)

    # Reshape every scan-dims variable from (..., receivers) to (..., antennas, feeds).
    scan_dims = ("timestamps", "frequencies", "antennas", "feeds")
    for name, (_, arr) in list(data_vars.items()):
        data_vars[name] = (scan_dims, _split_feeds(arr))

    # Add synchrotron correlation and calibrated data
    # Point-source flag has antennas dim (not receivers) — shape (timestamps, freq, antennas)
    cal_dims = ("timestamps", "frequencies", "antennas")
    data_vars["raw_flags_point_source"] = (
        cal_dims,
        point_source_flag[:, freq_select_slice, :],
    )
    data_vars["r_vis_synch_ant"] = (("antennas",), r_vis_synch_ant)
    # cal_vis and every cal_flags_* variable below get a trailing size-1 "stokes"
    # dim (currently only "I") via np.expand_dims, a view not a copy, in preparation
    # for future Stokes Q/U/V support.
    cal_dims = ("timestamps", "frequencies", "antennas", "stokes")
    data_vars["cal_vis"] = (cal_dims, np.expand_dims(calibrated_vis_data, axis=-1))

    # Add calibrated flags incrementally
    # Since cal_flags is a dictionary with keys corresponding to cal_flag_names and
    # values as numpy arrays of different shapes, we need to handle each flag array
    # according to its shape. For example, some flags may have shape
    # (timestamps, frequencies, antennas, stokes), while others may have shape
    # (antennas, stokes) or (frequencies, stokes). We will add them to the
    # dataset accordingly.
    for i, name in enumerate(cal_flag_names):
        if name == "HH_VV_combined" or name == "polynomial_fit_flag":
            cal_dims = ("timestamps", "frequencies", "antennas", "stokes")
            cf = cal_flags[name][:, freq_select_slice, :]
        elif (
            name == "temp_outlier_flag"
            or name == "temp_fluctuation_flag"
            or name == "synch_correlation_flag"
        ):
            cal_dims = ("antennas", "stokes")
            cf = cal_flags[name]
        elif name == "freq_flaggedfraction_flag":
            cal_dims = ("frequencies", "stokes")
            cf = cal_flags[name][freq_select_slice]
        else:
            raise ValueError(f"Unknown calibrated flag name: {name}")
        data_vars[f"cal_flags_{name}"] = (cal_dims, np.expand_dims(cf, axis=-1))
    # Add the final calibration flags
    data_vars["cal_flags_combined"] = (
        ("timestamps", "frequencies", "antennas", "stokes"),
        np.expand_dims(cal_flags_combined, axis=-1),
    )

    # Add RA/Dec as variables
    data_vars["ra"] = (("timestamps", "antennas"), ra)
    data_vars["dec"] = (("timestamps", "antennas"), dec)

    # Create dataset with all coordinates
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "timestamps": ("timestamps", timestamps),
            "frequencies": ("frequencies", frequencies),
            "antennas": ("antennas", antennas),
            "feeds": ("feeds", feeds),
            "stokes": ("stokes", np.array(["I"], dtype=str)),
        },
    )

    # Delete all intermediate arrays immediately
    logger.info("Cleaning up intermediate files")
    del (
        raw_vis,
        data_vars,
        receivers,
        feeds,
        frequencies,
        timestamps,
        flag_names,
        flag_elements,
    )
    del (
        calibrated_vis_data,
        cal_flags,
        ra,
        dec,
        antennas,
        r_vis_synch_ant,
        cal_flag_names,
    )
    gc.collect()

    # Add attributes...
    ds.attrs.update(
        {
            "block_name": block_name,
            "observation_date": obs_date,
            "scan_start": scan_start,
            "scan_end": scan_end,
            "scan_direction": scan_direction,
            "source_file": str(pickle_file),
            "raw_flag_name_list": raw_flag_name_list,
        }
    )

    # Add units as variable attributes
    ds["raw_vis"].attrs["units"] = "uncalibrated"
    ds["cal_vis"].attrs["units"] = "K_RJ"
    ds["ra"].attrs["units"] = "deg"
    ds["dec"].attrs["units"] = "deg"
    ds["frequencies"].attrs["units"] = "MHz"
    ds["timestamps"].attrs["units"] = "UNIX epoch seconds"
    ds.attrs["original_frequencies"] = original_frequencies
    ds.attrs["all_meerkat_antennas"] = [f"m{ant_num:03d}" for ant_num in range(64)]

    # Add any user-supplied extra attributes, applied last so they can override
    # attributes computed above if needed.
    if extra_attrs is not None:
        ds.attrs.update(extra_attrs)

    # Write out dataset to cache
    if cache_file_path is not None:
        logger.info(f"Writing dataset to cache: {cache_file_path}")
        ds.to_netcdf(cache_file_path)

    return ds
