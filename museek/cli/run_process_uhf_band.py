"""CLI for processing UHF band data using the MuSEEK pipeline.

Generates and submits a Slurm job to process UHF band data. This script creates
a temporary sbatch script and submits it to Slurm. Designed for processing
MEERKLASS observations.
"""

from __future__ import annotations

import click

from museek.cli.slurm_utils import submit_sbatch_script


def generate_sbatch_script(
    block_name: str,
    box: str,
    base_context_folder: str,
    data_folder: str,
    slurm_options: list[str],
) -> str:
    """Generate the sbatch script content."""
    context_folder = f"{base_context_folder}/BOX{box}/{block_name}"

    script_lines = [
        "#!/bin/bash",
        "",
        f"#SBATCH --job-name='MuSEEK-{block_name}'",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=32",
        "#SBATCH --mem=248GB",
        "#SBATCH --time=48:00:00",
        f"#SBATCH --output=museek-{block_name}-stdout.log",
        f"#SBATCH --error=museek-{block_name}-stderr.log",
    ]

    # Add additional SLURM options
    for option in slurm_options:
        script_lines.append(f"#SBATCH {option}")

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
            f'echo "Block name: {block_name}"',
            f'echo "Box: {box}"',
            f'echo "Data folder: {data_folder}"',
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
@click.option(
    "--block-name",
    required=True,
    help="Block name or observation ID (e.g., 1675632179)",
)
@click.option(
    "--box",
    required=True,
    help="Box number of this block name (e.g., 6)",
)
@click.option(
    "--base-context-folder",
    default="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline",
    help="Path to the base context/output folder. The final context folder will be <base-context-folder>/BOX<box>/<block-name>",
    show_default=True,
)
@click.option(
    "--data-folder",
    default="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/raw",
    help="Path to raw data folder",
    show_default=True,
)
@click.option(
    "--slurm-options",
    multiple=True,
    help="Additional SLURM options to pass to sbatch. Each --slurm-options takes ONE flag. Can be specified multiple times.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the generated sbatch script without submitting",
)
def main(
    block_name: str,
    box: str,
    base_context_folder: str,
    data_folder: str,
    slurm_options: tuple[str, ...],
    dry_run: bool,
) -> None:
    """Generate and submit a Slurm job to process UHF band data using the MuSEEK pipeline.

    \b
    EXAMPLES:
      museek_run_process_uhf_band --block-name 1675632179 --box 6
      museek_run_process_uhf_band --block-name 1675632179 --box 6 --base-context-folder /custom/pipeline
      museek_run_process_uhf_band --block-name 1675632179 --box 6 --dry-run
      museek_run_process_uhf_band --block-name 1675632179 --box 6 --slurm-options --mail-user=user@uni.edu --slurm-options --mail-type=ALL

    \b
    DEFAULT SLURM PARAMETERS:
      Job name:       MuSEEK-<block_name>
      Tasks:          1
      CPUs per task:  32
      Memory:         248GB
      Max time:       48 hours
      Output:         museek-<block_name>-stdout.log
      Error:          museek-<block_name>-stderr.log

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
