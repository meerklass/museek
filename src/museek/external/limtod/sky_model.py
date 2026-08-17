import healpy as hp
import numpy as np


# Example script to generate Gaussian random fields with a given covariance
# Credits: Katrine Alice Glasscock, Philip Bull
def generate_gaussian_field(
    freqs,
    nside,
    amp,
    alpha=1.0,
    beta=1.0,
    xi=1.0,
    f_ell=None,
    nu_ref=300.0,
    ell_ref=100.0,
    fwhm=0.0,
    seed=None,
    min_eigval=1e-10,
):
    """
    Generate a random realisation of a Gaussian field as a series of correlated
    Healpix maps. The field is drawn from a covariance matrix model of the form:

    C_ell(nu1, nu2) = A f_ell ((nu1*nu2)/(nu_ref^2))^beta
                    * exp((-ln(nu1/nu2)^2)/(2*xi**2))

    where we usually take f_ell = (ell / ell_ref)^alpha. Note that the field
    has zero mean (on average) by default. See Alonso et al. (2014) [1405.1751],
    Sect. 4.1, for the algorithm used.

    Parameters:
        freqs (array_like):
            Frequencies at which to generate maps, in MHz.
        nside (int):
            Healpix map nside.
        amp (float):
            Amplitude of the field's covariance, in the map units (e.g. Kelvin).
        alpha (float):
            Power law index of the ell dependence.
        beta (float):
            Power law index of the frequency dependence.
        xi (float):
            Frequency correlation coefficient.
        f_ell (func):
            User-defined function of ell. If this is specified, `alpha` will be
            ignored.
        nu_ref (float):
            Reference frequency, used as the pivot of the frequency power law.
        ell_ref (float):
            Reference ell value, used as the pivot of the ell power law.
        fwhm (float):
            FWHM of the Gaussian smoothing beam, in degrees.
        seed (int):
            Base random seed used for generating the maps.
        min_eigval (float):
            The  minimum eigenvalue to tolerate from the freq.-freq. part of the
            covariance matrix. Modes with eigenvalues lower than this are ignored.

    Returns:
        maps (array_like):
            Array of Healpix maps, of shape `(Nfreqs, Npix)`.
    """
    # Set random seed
    np.random.seed(seed)

    # Set f_ell function
    if f_ell is None:
        f_ell = lambda ell: (ell / ell_ref) ** alpha
    assert callable(f_ell), "f_ell must be a callable function of ell"

    # Set of ell values and ell-dependent covariance factor
    ell_max = 3 * nside - 1  # This is the correct value for a band-limited field
    ells = np.arange(ell_max + 1)
    C_ell = f_ell(ells)
    C_ell[0] = 0.0  # Remove monopole

    # Convert units of fwhm
    fwhm = np.deg2rad(fwhm)

    # Construct frequency-frequency part of covariance matrix and get eigenvecs
    Cnunu = (
        (freqs[:, np.newaxis] * freqs[np.newaxis, :]) / (nu_ref**2.0)
    ) ** beta * np.exp(
        (-np.log(freqs[:, np.newaxis] / freqs[np.newaxis, :]) ** 2.0) / (2.0 * xi**2.0)
    )
    all_eigvals, all_eigvecs = np.linalg.eigh(Cnunu)

    # Cut eigenvectors that have eigenvalues below the threshold
    idxs = np.where(np.abs(all_eigvals) >= min_eigval)
    eigvals = all_eigvals[idxs]
    eigvecs = all_eigvecs[idxs]

    # For each mode, generate a set of alms with the correct ell-dep. covariance
    # and amplitude
    # Nalms = (ell_max + 1) * (ell_max + 2) // 2
    # alms = np.zeros((len(eigvals), Nalms), dtype=np.complex128)
    mode_map = np.zeros((len(eigvals), hp.nside2npix(nside)))
    for mode in range(len(eigvals)):

        # Use synalm to make random map; this is the amplitude map for this eigenmode
        mode_map[mode, :] = hp.synfast(
            cls=amp * eigvals[mode] * C_ell, nside=nside, fwhm=fwhm
        )

    # Multiply mode maps by frequency eigenvectors and sum to get freq. maps
    m = np.einsum("mp,mf->fp", mode_map, eigvecs)
    return m
