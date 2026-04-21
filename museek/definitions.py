import os
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)  # This is your Project Root
KILO = 1e3
MEGA = 1e6
GIGA = 1e9

SPEED_OF_LIGHT = 3e8  # m/s
SECONDS_IN_ONE_DAY = 86400  # s


def get_cache_dir() -> str:
    """Determine the cache directory with fallback logic.

    Priority:
    1. If ROOT_DIR is writable by user, use ROOT_DIR/cache
    2. If MUSEEK_CACHE_DIR env var exists, use it
    3. Otherwise, use XDG cache directory: $XDG_CACHE_HOME/museek or ~/.cache/museek

    Returns:
        str: Path to the cache directory

    """
    root_path = Path(ROOT_DIR)

    # Check if ROOT_DIR is writable
    if os.access(root_path, os.W_OK):
        cache_dir = root_path / "cache"
    # Check if MUSEEK_CACHE_DIR environment variable is set
    elif "MUSEEK_CACHE_DIR" in os.environ:
        cache_dir = Path(os.environ["MUSEEK_CACHE_DIR"])
    # Fall back to XDG cache directory
    else:
        xdg_cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
        cache_dir = xdg_cache_home / "museek"

    return str(cache_dir)
