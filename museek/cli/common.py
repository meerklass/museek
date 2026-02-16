"""Common CLI utilities and decorators for MuSEEK commands."""

from __future__ import annotations

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
        type=(str, str),
        metavar="<OPTION VALUE>...",
        multiple=True,
        help="Additional SLURM options to pass to sbatch. Can be specified multiple times.",
    )


def add_dry_run_option():
    """Decorator for adding --dry-run option."""
    return click.option(
        "--dry-run",
        is_flag=True,
        help="Show the generated sbatch script without submitting",
    )
