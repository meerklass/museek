"""Memory-frugal readers for the MeerKAT primary-beam ``.npz``.

The beam file (e.g. ``MeerKAT_U_band_primary_beam_aa_highres.npz``) stores its ``beam`` array
uncompressed and ~8.6 GB in size. numpy's ``NpzFile`` ignores ``mmap_mode``, so a plain
``np.load(...)["beam"]`` pulls the whole thing into RAM even when only a couple of
polarisation/antenna slices are needed. These helpers memory-map the single ``beam`` member and
read only the slices required, keeping the peak footprint to a few slices rather than the full cube.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as _npf


def npz_member_memmap(npz_path: str | Path, member: str) -> np.memmap:
    """Memory-map one array from an *uncompressed* ``.npz`` without loading the whole thing.

    :param npz_path: path to the ``.npz`` archive
    :param member: array name (with or without the ``.npy`` suffix)
    :return: a read-only ``np.memmap`` of the member; slicing it reads only the touched pages
    :raise ValueError: if the member is compressed (``mmap`` requires ``ZIP_STORED``)
    """
    name = member if member.endswith(".npy") else member + ".npy"
    info = zipfile.ZipFile(npz_path).getinfo(name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{name} is compressed in {npz_path}; cannot memory-map it.")
    with open(npz_path, "rb") as fh:
        fh.seek(info.header_offset + 26)
        n_name, n_extra = struct.unpack("<HH", fh.read(4))  # local-header name/extra lengths
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        major, _minor = _npf.read_magic(fh)
        shape, fortran, dtype = (_npf.read_array_header_2_0(fh) if major >= 2
                                 else _npf.read_array_header_1_0(fh))
        data_offset = fh.tell()
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=data_offset,
                     shape=shape, order="F" if fortran else "C")


def _decode(names: np.ndarray) -> list[str]:
    """Decode a numpy array of (possibly bytes) labels to a list of ``str``."""
    return [n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n) for n in names]


def load_beam_power_cubes(
    beam_file: str | Path,
    polarizations: tuple[str, ...] = ("HH", "VV"),
    antenna: str = "array_average",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read only the requested polarisation power cubes from the beam ``.npz``, via memmap.

    The ``beam`` member has shape ``(n_pol, n_ant, n_freq, n_m, n_l)`` (complex Jones). For each
    requested polarisation this reads the single ``[pol, ant]`` slice, forms ``|.|²`` as float32,
    and frees the complex slice before the next one — so the peak is a couple of ``(n_freq, n_m,
    n_l)`` slices rather than the full ~8.6 GB cube.

    :param beam_file: path to the beam ``.npz``
    :param polarizations: polarisations to materialise (upper-cased; must exist in the file's ``pols``)
    :param antenna: antenna label to select (falls back to index 0 if ``antnames`` is absent/missing it)
    :return: ``(freq_MHz, margin_deg, {pol: power_cube_float32})`` where each cube is ``(n_freq, n_m, n_l)``
    """
    beam_file = Path(beam_file)
    with np.load(beam_file) as data:  # NpzFile reads members lazily; we never touch ``beam`` here
        freq_MHz = np.asarray(data["freq_MHz"], dtype=np.float64)
        margin_deg = np.asarray(data["margin_deg"], dtype=np.float64)
        pols = _decode(np.asarray(data["pols"])) if "pols" in data.files else ["HH", "HV", "VH", "VV"]
        antnames = _decode(np.asarray(data["antnames"])) if "antnames" in data.files else []

    i_ant = antnames.index(antenna) if antenna in antnames else 0
    beam_mm = npz_member_memmap(beam_file, "beam")  # (n_pol, n_ant, n_freq, n_m, n_l), memmapped

    cubes: dict[str, np.ndarray] = {}
    for pol in polarizations:
        pol = pol.upper()
        if pol not in pols:
            raise ValueError(f"Polarization '{pol}' not in beam file (has {pols}).")
        # `beam_mm[pol, ant]` is a memmap view; computing the power reads pages on demand, so we
        # never materialise the full complex slice. Formula matches simeer's MeerKLASSBeam
        # (real**2 + imag**2), so the foreground path is bit-for-bit unchanged.
        jones = beam_mm[pols.index(pol), i_ant]  # (n_freq, n_m, n_l) complex memmap view
        cubes[pol] = (jones.real.astype(np.float32) ** 2 + jones.imag.astype(np.float32) ** 2)
    return freq_MHz, margin_deg, cubes
