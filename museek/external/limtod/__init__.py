"""Vendored subset of limTOD (MIT, (c) Zheng Zhang; see LICENSE).

museek only uses the 1/f-noise generator and the Gaussian-field sky mock, so only those
are vendored here -- ``flicker_model.py`` is copied verbatim and ``sky_model.py`` keeps
only ``generate_gaussian_field`` (the upstream ``GDSM_sky_model`` and its ``pygdsm``
dependency are omitted). See ../README.md for provenance.
"""

from .flicker_model import sim_noise
from .sky_model import generate_gaussian_field

__all__ = ["sim_noise", "generate_gaussian_field"]

__author__ = "Zheng Zhang"
__license__ = "MIT"
