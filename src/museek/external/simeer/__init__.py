"""
Simeer: optimal MeerKLASS TOD simulator with native (l, m) beam interpolation.

Author:  Zheng Zhang <zheng.zhang@manchester.ac.uk> (University of Manchester)
License: MIT (see LICENSE)
Version: 0.1.0

Sky TOD vs Full TOD
-------------------

Simeer computes the **Sky TOD**: the noiseless beam-weighted sky signal,
``T_sky(nu, t) = (1/Omega_b(nu)) * integral B(l, m, nu) T(l, m, nu) dOmega``.
It does NOT add gain, 1/f noise, or white noise; the **Full TOD**
assembly (multiplying by gain and injecting noise) is delegated to
:class:`limTOD.TODSim.generate_TOD`, which :class:`SimeerTODSim`
inherits unchanged.

Public API
----------

Sky-TOD generation (no limTOD dependency):

*   :func:`simeer.sky_integrator.integrate_tod` -- Sky TOD for a list of
    pointings; returns ``(n_freq, n_time)`` ndarray.
*   :func:`simeer.sky_integrator.integrate_sample` -- Sky TOD for one
    pointing; returns ``(n_freq,)`` ndarray.

Building blocks (useful for custom pipelines / tests):

*   :class:`MeerKLASSBeam` -- load and query the holographic beam.
*   :func:`simeer.projection.direction_cosines`
*   :func:`simeer.interpolation.precompute_bilinear_weights`
*   :func:`simeer.disc.select_disc`
"""

from .beam import MeerKLASSBeam, synthetic_gaussian_beam
from .sky_integrator import (
    integrate_sample,
    integrate_tod,
    materialise_sky_cube,
    materialize_sky_cube,
)

__all__ = [
    "MeerKLASSBeam",
    "synthetic_gaussian_beam",
    "integrate_sample",
    "integrate_tod",
    "materialize_sky_cube",
    # ``materialise_sky_cube`` is the deprecated British spelling kept for
    # back-compat; will be removed in v0.2.
    "materialise_sky_cube",
]

__version__ = "0.1.0"
__author__ = "Zheng Zhang"
__email__ = "zheng.zhang@manchester.ac.uk"
__license__ = "MIT"

# NOTE (museek vendoring): the upstream ``SimeerTODSim`` lazy-loader (which imports
# ``.simulator`` and depends on limTOD + pygdsm) has been removed -- museek only uses the
# Sky-TOD path (``MeerKLASSBeam`` / ``integrate_tod``) and ``simulator.py`` is not vendored.
