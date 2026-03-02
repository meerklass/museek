"""Common CLI utilities and decorators for MuSEEK commands."""

from __future__ import annotations

from pathlib import Path

import click


def add_block_name_option():
    """Decorator for adding --block-name / -b option."""
    return click.option(
        "-b",
        "--block-name",
        type=str,
        required=True,
        help="Block name or observation ID (e.g., 1708972386)",
    )


def add_box_option():
    """Decorator for adding --box / -x option."""
    return click.option(
        "-x",
        "--box",
        type=str,
        required=True,
        help="Box number of this block name (e.g., 6)",
    )


def add_slurm_options():
    """Decorator for adding --slurm-options / -s option."""
    return click.option(
        "-s",
        "--slurm-options",
        type=str,
        multiple=True,
        help="Additional SLURM options to pass to sbatch (e.g., --exclusive, --mail-user=user@domain.com). Can be specified multiple times.",
    )


def add_venv_option():
    """Decorator for adding --venv / -v option."""
    return click.option(
        "-v",
        "--venv",
        type=click.Path(
            file_okay=False,
            dir_okay=True,
            writable=False,
            path_type=Path,
            resolve_path=True,
        ),
        default="/idia/projects/meerklass/virtualenv/meerklass",
        help="Path to Python virtual environment. "
        "Use the default shared meerklass environment on ilifu if not specified.",
        show_default=True,
    )


def add_dry_run_option():
    """Decorator for adding --dry-run option."""
    return click.option(
        "--dry-run",
        is_flag=True,
        help="Show the generated sbatch script without submitting",
    )
