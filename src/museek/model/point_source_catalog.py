"""
Point-source catalog loader and flux/temperature helpers for the scan-data simulation.

Parses the pipe-delimited `1Jy_cat.txt` catalog and provides per-source power-law flux
extrapolation and the standard single-dish flux-density -> antenna-temperature conversion.
"""

import numpy as np

JY = 1e-26  # W m^-2 Hz^-1 per Jansky
K_B = 1.380649e-23  # Boltzmann constant, J/K
C = 299792458.0  # m/s

REF_NVSS_HZ = 1400e6
REF_PKS1410_HZ = 1410e6
REF_PKS635_HZ = 635e6
REF_PKS408_HZ = 408e6


def load_catalog(
    path: str, min_flux_Jy: float = 0.0, flux_cut_freq_Hz: float = 800e6
) -> dict:
    """
    Parse the `1Jy_cat.txt` point-source catalog.

    Columns: SOURCE_ID|RA|DEC|NVSS_FLUX|INT_SPEC_INDX|3C_NAME|PKS_JNAME|
             PKS_1410_FLUX|PKS_635_FLUX|PKS_408_FLUX|PKS_SPEC_INDX

    The flux is a single power law `S(nu) = S_ref (nu/nu_ref)^alpha`. The PKS (Parkes) values are
    preferred because they are single-dish total fluxes (like MeerKAT single-dish), whereas NVSS is
    interferometric and resolves out extended emission. Selection per source:

      - if a PKS spectral index is available (PKS_SPEC_INDX != 0 and some PKS flux > 0):
        alpha = PKS_SPEC_INDX, reference flux from PKS_1410 > PKS_635 > PKS_408 (first available);
      - else if NVSS_FLUX > 0: alpha = INT_SPEC_INDX at 1.4 GHz;
      - else the source is dropped.

    Note: the hand-coded calibrator functions in `point_sources.py` use the same PKS_1410 reference
    flux but a *two-point* index from PKS_1410 & PKS_408 rather than the tabulated PKS_SPEC_INDX, so
    they differ from this loader by ~1-4% at U-band.

    :param path: path to the catalog file
    :param min_flux_Jy: drop sources fainter than this; the cut is applied to the flux extrapolated
        to `flux_cut_freq_Hz` (NOT the per-source reference flux, whose reference frequency varies)
    :param flux_cut_freq_Hz: frequency at which the `min_flux_Jy` cut is evaluated (default 800 MHz,
        the middle of the MeerKAT U band)
    :return: dict of equal-length arrays: source_id, ra_deg, dec_deg, s_ref_Jy, nu_ref_Hz, alpha
    """
    sid, ra, dec, s_ref, nu_ref, alpha = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("|")
            try:
                r, d = float(parts[1]), float(parts[2])
                nvss, nvss_idx = float(parts[3]), float(parts[4])
                pks1410, pks635, pks408 = (
                    float(parts[7]),
                    float(parts[8]),
                    float(parts[9]),
                )
                pks_idx = float(parts[10])
            except (IndexError, ValueError):
                continue
            if pks_idx != 0 and (pks1410 > 0 or pks635 > 0 or pks408 > 0):
                a = pks_idx
                if pks1410 > 0:
                    s, nu = pks1410, REF_PKS1410_HZ
                elif pks635 > 0:
                    s, nu = pks635, REF_PKS635_HZ
                else:
                    s, nu = pks408, REF_PKS408_HZ
            elif nvss > 0:
                s, nu, a = nvss, REF_NVSS_HZ, nvss_idx
            else:
                continue
            if min_flux_Jy > 0 and s * (flux_cut_freq_Hz / nu) ** a < min_flux_Jy:
                continue  # cut on the flux extrapolated to a fixed frequency, consistent across sources
            sid.append(parts[0])
            ra.append(r)
            dec.append(d)
            s_ref.append(s)
            nu_ref.append(nu)
            alpha.append(a)

    return dict(
        source_id=np.array(sid),
        ra_deg=np.array(ra),
        dec_deg=np.array(dec),
        s_ref_Jy=np.array(s_ref),
        nu_ref_Hz=np.array(nu_ref),
        alpha=np.array(alpha),
    )


def flux_Jy(catalog: dict, freq_Hz: np.ndarray) -> np.ndarray:
    """
    Power-law flux per source at the requested frequencies: S(nu) = S_ref (nu/nu_ref)^alpha.

    :param catalog: output of `load_catalog` (possibly index-sliced to a source subset)
    :param freq_Hz: scalar or (n_freq,) array
    :return: (n_src, n_freq) if `freq_Hz` is an array, else (n_src,)
    """
    f = np.atleast_1d(np.asarray(freq_Hz, dtype=float))
    s = (
        catalog["s_ref_Jy"][:, None]
        * (f[None, :] / catalog["nu_ref_Hz"][:, None]) ** catalog["alpha"][:, None]
    )
    return s if np.ndim(freq_Hz) else s[:, 0]


def jy_to_kelvin(
    flux_Jy_value: np.ndarray, freq_Hz: np.ndarray, omega_sr: np.ndarray
) -> np.ndarray:
    """
    Convert flux density to peak antenna temperature for a beam (or pixel) of solid angle `omega_sr`:

        T = S [W m^-2 Hz^-1] * lambda^2 / (2 k_B omega)

    Multiply by the normalised beam power (peak 1) at the source offset to get the off-axis response.

    :param flux_Jy_value: flux density in Jy, any shape broadcastable with `freq_Hz`/`omega_sr`
    :param freq_Hz: frequency in Hz
    :param omega_sr: solid angle in steradians (beam solid angle for method B, pixel area for method A)
    :return: temperature in Kelvin
    """
    lam = C / np.asarray(freq_Hz, dtype=float)
    return flux_Jy_value * JY * lam**2 / (2.0 * K_B * omega_sr)


def select_near_track(
    catalog: dict,
    track_ra_deg: np.ndarray,
    track_dec_deg: np.ndarray,
    radius_deg: float,
) -> np.ndarray:
    """
    Indices of catalog sources whose great-circle distance to any pointing on the track is < radius.

    :param catalog: output of `load_catalog`
    :param track_ra_deg, track_dec_deg: pointing track (RA, Dec) in degrees, shape (n_pointing,)
    :param radius_deg: selection radius in degrees
    :return: integer index array of selected sources
    """
    src_ra = np.radians(catalog["ra_deg"])[:, None]
    src_dec = np.radians(catalog["dec_deg"])[:, None]
    t_ra = np.radians(track_ra_deg)[None, :]
    t_dec = np.radians(track_dec_deg)[None, :]
    # great-circle (haversine) separation, (n_src, n_pointing)
    cos_sep = np.sin(src_dec) * np.sin(t_dec) + np.cos(src_dec) * np.cos(
        t_dec
    ) * np.cos(src_ra - t_ra)
    min_sep = np.arccos(np.clip(cos_sep, -1.0, 1.0)).min(axis=1)
    return np.where(min_sep < np.radians(radius_deg))[0]
