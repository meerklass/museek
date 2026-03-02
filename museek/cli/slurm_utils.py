"""Shared utilities for SLURM job submission."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import click


def merge_slurm_options(
    default_options: list[str], user_options: list[str]
) -> list[str]:
    """Merge SLURM options, allowing user options to override defaults.

    Parses both default and user-provided SLURM options to detect duplicates.
    When a user option has the same key as a default option (e.g., both specify
    --output), the user option replaces the default. New user options that don't
    match any default are appended.

    Args:
        default_options: List of default SLURM option lines (without "#SBATCH " prefix)
        user_options: List of user-provided SLURM options (without "#SBATCH " prefix)

    Returns:
        Merged list of SLURM options with user options overriding defaults

    Example:
        >>> defaults = ["--ntasks=1", "--mem=248GB", "--output=output.log"]
        >>> user = ["--output=custom.log", "--mail-user=user@example.com"]
        >>> merge_slurm_options(defaults, user)
        ["--ntasks=1", "--mem=248GB", "--output=custom.log", "--mail-user=user@example.com"]
    """

    def extract_option_key(option: str) -> str:
        """Extract the key from a SLURM option.

        Examples:
            '--output=file.log' -> '--output'
            '--job-name value' -> '--job-name'
            '-o file.log' -> '-o'
        """
        option = option.strip()
        # Handle long options like '--job-name=value' or '--job-name value'
        if option.startswith("--"):
            if "=" in option:
                return option.split("=", 1)[0]
            else:
                # Take first token (handles '--option value' format)
                return option.split()[0] if " " in option else option
        # Handle short options like '-o value' or '-o=value'
        elif option.startswith("-"):
            if "=" in option:
                return option.split("=", 1)[0]
            else:
                return option.split()[0] if " " in option else option
        return option

    # Build dictionary mapping option keys to their values
    options_dict = {}
    default_keys_order = []

    # Add default options (preserve order)
    for opt in default_options:
        key = extract_option_key(opt)
        options_dict[key] = opt
        default_keys_order.append(key)

    # Track new user options that aren't overrides
    new_user_options = []

    # Override with user options
    for opt in user_options:
        key = extract_option_key(opt)
        if key in options_dict:
            # Override existing default
            options_dict[key] = opt
        else:
            # New option not in defaults
            new_user_options.append(opt)

    # Build result: defaults (with overrides) followed by new user options
    result = [options_dict[key] for key in default_keys_order]
    result.extend(new_user_options)

    return result


def build_sbatch_script(
    default_slurm_options: list[str],
    user_slurm_options: list[str],
    venv_path: Path,
    body_lines: list[str],
    success_message: str = "Job completed successfully!",
    error_message: str = "Job failed!",
) -> str:
    """Build a complete sbatch script with common structure.

    This function generates the standard structure for MuSEEK sbatch scripts:
    1. Shebang and SLURM directives (merged from defaults and user options)
    2. Virtual environment activation
    3. Custom body (job-specific commands, logging, execution)
    4. Exit status checking with custom messages

    Args:
        default_slurm_options: Default SLURM options (without "#SBATCH " prefix)
        user_slurm_options: User-provided SLURM options (without "#SBATCH " prefix)
        venv_path: Path to the virtual environment to activate
        body_lines: List of script lines for the job-specific logic
        success_message: Message to display on successful completion
        error_message: Message to display on failure

    Returns:
        Complete sbatch script content as a string

    Example:
        >>> default_opts = ["--job-name=test", "--ntasks=1"]
        >>> user_opts = ["--mem=512GB"]
        >>> body = ['echo "Running test"', "python script.py"]
        >>> script = build_sbatch_script(default_opts, user_opts, Path("/venv"), body)
    """
    # Merge user options with defaults (user options override defaults)
    merged_options = merge_slurm_options(default_slurm_options, user_slurm_options)

    # Build script header with merged SLURM options
    script_lines = ["#!/bin/bash", ""]
    for option in merged_options:
        script_lines.append(f"#SBATCH {option}")

    # Add virtual environment activation
    script_lines.extend(
        [
            "",
            "# Activate Python virtual environment",
            f"source {venv_path}/bin/activate",
            'echo "Python executable is: $(which python)"',
            "echo \"MuSEEK version: $(python -c 'import museek; print(museek.__version__)')\"",
            "",
        ]
    )

    # Add custom body lines
    script_lines.extend(body_lines)

    # Add exit status checking
    script_lines.extend(
        [
            "",
            "# Check exit status",
            "if [ $? -eq 0 ]; then",
            '    echo "=========================================="',
            f'    echo "{success_message}"',
            '    echo "=========================================="',
            "    exit 0",
            "else",
            '    echo "=========================================="',
            f'    echo "Error: {error_message}"',
            '    echo "=========================================="',
            "    exit 1",
            "fi",
        ]
    )

    return "\n".join(script_lines) + "\n"


def submit_sbatch_script(script_content: str, dry_run: bool = False) -> None:
    """Submit a sbatch script to SLURM or display it in dry-run mode.

    Args:
        script_content: The complete sbatch script content to submit
        dry_run: If True, display the script instead of submitting it
    """
    if dry_run:
        # Display the script without submitting
        click.echo("=" * 42)
        click.echo("DRY-RUN: Generated sbatch script")
        click.echo("=" * 42)
        click.echo(script_content)
        click.echo("=" * 42)
        click.echo("To submit this job, run without --dry-run")
        click.echo("=" * 42)
    else:
        # Write to temporary file and submit
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            temp_path = Path(f.name)
            f.write(script_content)

        try:
            click.echo("=" * 42)
            click.echo("Submitting job to Slurm")
            click.echo("=" * 42)
            result = subprocess.run(["sbatch", str(temp_path)], check=False)
            if result.returncode == 0:
                click.echo("Job submitted successfully!")
            else:
                click.echo("Error: Failed to submit job")
                sys.exit(1)
        finally:
            temp_path.unlink(missing_ok=True)
