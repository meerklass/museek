"""CLI for processing UHF band data using the MuSEEK pipeline.

Generates and submits a Slurm job to process UHF band data. This script creates
a temporary sbatch script and submits it to Slurm. Designed for processing
MEERKLASS observations.
"""

from __future__ import annotations

from pathlib import Path

import click

from museek.cli.common import (
    add_block_name_option,
    add_box_option,
    add_dry_run_option,
    add_slurm_options,
)
from museek.cli.slurm_utils import submit_sbatch_script


def generate_sbatch_script(
    block_name: str,
    box: str,
    base_context_folder: Path,
    data_folder: Path,
    slurm_options: list[tuple[str, str]],
) -> str:
    """Generate the sbatch script content."""
    context_folder = Path(base_context_folder) / f"BOX{box}" / block_name

    script_lines = [
        "#!/bin/bash",
        "",
        f"#SBATCH --job-name=MuSEEK-Process-UHF-{block_name}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=32",
        "#SBATCH --mem=248GB",
        "#SBATCH --time=48:00:00",
        f"#SBATCH --output=museek-process-uhf-{block_name}.log",
    ]

    # Add additional SLURM options
    for key, value in slurm_options:
        script_lines.append(f"#SBATCH --{key}={value}")

    script_lines.extend(
        [
            "",
            "# Use shared meerklass environment",
            "source /idia/projects/meerklass/virtualenv/meerklass/bin/activate",
            'echo "Python executable is: $(which python)"',
            "",
            "# Log job information",
            'echo "=========================================="',
            'echo "Executing MuSEEK UHF Band Processing"',
            'echo "=========================================="',
            f'echo "Block name:    {block_name}"',
            f'echo "Box:           {box}"',
            f'echo "Data folder:   {data_folder}"',
            f'echo "Context folder: {context_folder}"',
            'echo "=========================================="',
            "",
            "# Execute the pipeline",
            f"museek --InPlugin-block-name={block_name} --InPlugin-context-folder={context_folder} \\",
            f"    --InPlugin-data-folder={data_folder} museek.config.process_uhf_band",
            "",
            "# Check exit status",
            "if [ $? -eq 0 ]; then",
            '    echo "=========================================="',
            '    echo "MuSEEK pipeline completed successfully!"',
            '    echo "=========================================="',
            "    exit 0",
            "else",
            '    echo "=========================================="',
            '    echo "Error: MuSEEK pipeline failed!"',
            '    echo "=========================================="',
            "    exit 1",
            "fi",
        ]
    )

    return "\n".join(script_lines) + "\n"


@click.command(
    context_settings=dict(
        help_option_names=["-h", "--help"],
    )
)
@add_block_name_option()
@add_box_option()
@click.option(
    "-c",
    "--base-context-folder",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path, resolve_path=True
    ),
    default="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline",
    help="Base directory for context/output. Final output: <base-context-folder>/BOX<box>/<block-name>",
    show_default=True,
)
@click.option(
    "-d",
    "--data-folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path, resolve_path=True),
    default="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/raw",
    help="Path to raw data folder",
    show_default=True,
)
@add_slurm_options()
@add_dry_run_option()
def main(
    block_name: str,
    box: str,
    base_context_folder: Path,
    data_folder: Path,
    slurm_options: tuple[tuple[str, str], ...],
    dry_run: bool,
) -> None:
    """Generate and submit a Slurm job to process UHF band data using the MuSEEK pipeline.

    \b
    EXAMPLES:
      museek_run_process_uhf_band --block-name 1675632179 --box 6
      museek_run_process_uhf_band --block-name 1675632179 --box 6 --base-context-folder /custom/pipeline
      museek_run_process_uhf_band --block-name 1675632179 --box 6 --dry-run
      museek_run_process_uhf_band --block-name 1675632179 --box 6 --slurm-options mail-user user@uni.edu --slurm-options mail-type ALL

    \b
    DEFAULT SLURM PARAMETERS:
      Job name:       MuSEEK-<block_name>
      Tasks:          1
      CPUs per task:  32
      Memory:         248GB
      Max time:       48 hours
      Log output:     museek-<block_name>.log

    \b
    REQUIREMENTS:
      - Access to Ilifu
      - meerklass-1 group permission for raw data access
      - sbatch command available (Slurm)
    """
    # Generate the sbatch script
    script_content = generate_sbatch_script(
        block_name=block_name,
        box=box,
        base_context_folder=base_context_folder,
        data_folder=data_folder,
        slurm_options=list(slurm_options),
    )

    # Submit or display the script
    submit_sbatch_script(script_content, dry_run=dry_run)


if __name__ == "__main__":
    main()
