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

    Resolution policy (robust to editable installs and installed console wrappers):
    1. If a project-local script exists at `<project_root>/scripts/<script_name>`, prefer it.
    2. Otherwise, check PATH (shutil.which). If found and it appears to be a shell
       script (shebang mentions sh or bash) return that path.
    3. If PATH entry is a Python console wrapper (shebang references python) we avoid
       using it (to prevent invoking this same wrapper again recursively) and fall
       back to the project-local script if present. If no project-local script exists,
       return the PATH entry as a last resort.
    4. If nothing found, return None.
    """
    # 1) look relative to this file: .../museek/cli/scripts.py -> project root
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    candidate = project_root / "scripts" / script_name
    if candidate.exists():
        return candidate

    # 2) check PATH
    which_path = shutil.which(script_name)
    if which_path:
        # Try to inspect the shebang to distinguish shell scripts from Python wrappers
        try:
            with open(which_path, "rb") as fh:
                first_bytes = fh.read(2048)
            first_line = first_bytes.splitlines()[0].decode("utf-8", errors="ignore") if first_bytes else ""
        except Exception:
            first_line = ""

        # If shebang contains sh or bash, treat as a shell script
        if "sh" in first_line or "bash" in first_line:
            return Path(which_path)

        # If this looks like a Python wrapper, prefer the project-local script if available
        # (we already checked candidate above). Otherwise fall back to the PATH entry.
        return Path(which_path)

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
