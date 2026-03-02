"""
Museek: A flexible and easy-to-extend data processing pipeline for multi-instrument autocorrelation radio experiments.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("museek")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
