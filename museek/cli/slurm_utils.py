"""Shared utilities for SLURM job submission."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import click


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
