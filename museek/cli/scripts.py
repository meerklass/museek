"""Console entry point wrappers for the shell scripts in the repository.

These wrappers look for an installed script on PATH first and fall back to the
`scripts/` directory in the project source tree (useful for editable installs).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_NAMES = {
    "process_uhf_band": "museek_process_uhf_band.sh",
    "run_notebook": "museek_run_notebook.sh",
}


def _find_script(script_name: str) -> Path | None:
    """Return a path to the script.

    1. If a program with `script_name` is on PATH (shutil.which), return the
       resolved path as a Path object.
    2. Otherwise, assume we're in a development checkout and look for
       `<project_root>/scripts/<script_name>` and return that path if it exists.
    3. If nothing found, return None.
    """
    # 1) check PATH
    which_path = shutil.which(script_name)
    if which_path:
        return Path(which_path)

    # 2) look relative to this file: .../museek/cli/scripts.py -> project root
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    candidate = project_root / "scripts" / script_name
    if candidate.exists():
        return candidate

    return None


def _run_script(script_name: str) -> int:
    """Run the given script, forwarding command-line arguments and return exit code."""
    script = _find_script(script_name)
    if script is None:
        sys.stderr.write(f"Error: could not find script '{script_name}' on PATH or in project 'scripts/' folder.\n")
        return 2

    cmd = [str(script)] + sys.argv[1:]
    try:
        completed = subprocess.run(cmd)
        return completed.returncode
    except FileNotFoundError:
        sys.stderr.write(f"Error: failed to execute script: {script}\n")
        return 3


def process_uhf_band() -> None:
    """Console entry point for `museek_process_uhf_band`.

    This function is designed to be used as a `console_scripts` / PEP 621
    entry point. It forwards any CLI arguments to the shell script.
    """
    sys.exit(_run_script(SCRIPT_NAMES["process_uhf_band"]))


def run_notebook() -> None:
    """Console entry point for `museek_run_notebook`.

    This function is designed to be used as a `console_scripts` / PEP 621
    entry point. It forwards any CLI arguments to the shell script.
    """
    sys.exit(_run_script(SCRIPT_NAMES["run_notebook"]))
