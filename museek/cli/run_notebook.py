"""CLI for running MuSEEK Jupyter notebooks using papermill.

Generates and submits a Slurm job to execute a MuSEEK Jupyter notebook using
papermill. This script creates a temporary sbatch script and submits it to Slurm.
Kernel validation is performed before submission.

Notebook Discovery:
    Notebooks are searched in the following order:
    1. Absolute path to notebook file (if provided)
    2. Installed package location (museek/notebooks/<notebook_name>.ipynb)
    3. Current working directory (./<notebook_name>.ipynb)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from museek.cli.common import (
    add_block_name_option,
    add_box_option,
    add_dry_run_option,
    add_slurm_options,
)
from museek.cli.slurm_utils import submit_sbatch_script


def validate_kernel(kernel: str) -> None:
    """Check if the kernel is available, install meerklass kernel if needed."""
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and kernel in result.stdout:
            click.echo(f"Kernel '{kernel}' is available")
            return
    except FileNotFoundError:
        click.echo("Warning: jupyter command not found")
        return

    click.echo(f"Warning: Kernel '{kernel}' not found in jupyter kernelspec list")

    if kernel == "meerklass":
        click.echo("Installing meerklass kernel...")
        result = subprocess.run(
            ["python", "-m", "ipykernel", "install", "--name", "meerklass", "--user"],
            check=False,
        )
        if result.returncode != 0:
            click.echo("Error: Failed to install meerklass kernel")
            sys.exit(1)
        click.echo("Successfully installed meerklass kernel")
    else:
        click.echo(
            f"Error: Kernel '{kernel}' is not available and will not be auto-installed"
        )
        click.echo(
            "Please install the kernel or specify a different kernel with --kernel option"
        )
        click.echo("Available kernels:")
        subprocess.run(["jupyter", "kernelspec", "list"], check=False)
        sys.exit(1)


def validate_notebook(notebook_name: str) -> Path:
    """Verify that the notebook exists and return its path.

    Search order:
    1. If notebook_name is an absolute path, use it directly (must be a .ipynb file)
    2. Installed package location (museek/notebooks/<notebook_name>.ipynb)
    3. Current working directory (./<notebook_name>.ipynb)

    Returns:
        Path to the notebook file.

    Raises:
        SystemExit: If notebook is not found in any location.
    """
    notebook_path = Path(notebook_name)

    # 1. Check if absolute path was provided
    if notebook_path.is_absolute():
        if not notebook_path.exists():
            click.echo(f"Error: Notebook not found: {notebook_path}")
            sys.exit(1)
        return notebook_path

    # 2. Check installed package location
    try:
        import museek

        pkg_path = Path(museek.__file__).parent
        pkg_notebook = pkg_path / "notebooks" / f"{notebook_name}.ipynb"
        if pkg_notebook.exists():
            return pkg_notebook
    except Exception:
        pass

    # 3. Check current working directory
    cwd_notebook = Path.cwd() / f"{notebook_name}.ipynb"
    if cwd_notebook.exists():
        return cwd_notebook

    # Notebook not found in any location
    click.echo(f"Error: Notebook '{notebook_name}.ipynb' not found")
    click.echo("Searched in:")
    click.echo("  1. Installed package location (museek/notebooks/)")
    click.echo("  2. Current working directory (.)")
    click.echo("")
    click.echo("You can also provide an absolute path to the notebook.")
    sys.exit(1)


def generate_sbatch_script(
    notebook_path: Path,
    block_name: str,
    box: str,
    base_output_dir: Path,
    kernel: str,
    notebook_name: str,
    parameters: list[tuple[str, str]],
    slurm_options: list[tuple[str, str]],
) -> str:
    """Generate the sbatch script content.

    Best-effort uses pathlib to construct output paths and uses the notebook's stem
    for the output filename so absolute notebook paths do not create malformed
    filenames.
    """
    # Output: <base_output_dir>/BOX<box>/<block_name>/<notebook_name>_<block_name>.ipynb
    output_dir = Path(base_output_dir) / f"BOX{box}" / block_name
    # Use the notebook name's stem (base name without suffix) for the output file
    output_notebook = output_dir / f"{Path(notebook_name).stem}_{block_name}.ipynb"

    # Build papermill parameters string
    # Always include mandatory parameters: block_name and data_path
    param_lines = []
    param_lines.append(f"    -p block_name {block_name} \\")
    data_path = output_dir / f"{block_name}"
    param_lines.append(f"    -p data_path {data_path.as_posix()} \\")
    # Add any additional parameters provided by the user
    for param_name, param_value in parameters:
        param_lines.append(f"    -p {param_name} {param_value} \\")

    script_lines = [
        "#!/bin/bash",
        "",
        f"#SBATCH --job-name='MuSEEK-Notebook-{block_name}'",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=32",
        "#SBATCH --mem=248GB",
        "#SBATCH --time=01:00:00",
        f"#SBATCH --output={notebook_name}-{block_name}.log",
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
            'echo "Papermill version: $(papermill --version)"',
            "",
            "# Log job information",
            'echo "=========================================="',
            'echo "Executing MuSEEK Notebook"',
            'echo "=========================================="',
            f'echo "Notebook:      {notebook_path}"',
            f'echo "Block name:    {block_name}"',
            f'echo "Box:           {box}"',
            f'echo "Kernel:        {kernel}"',
            f'echo "Output:        {output_notebook}"',
            'echo "=========================================="',
            "",
            "# Create output directory",
            f"mkdir -p {output_dir}",
            "",
            "# Execute notebook using papermill with collected parameters",
            f"papermill -k {kernel} \\",
        ]
    )

    # Add papermill parameters with proper formatting
    if param_lines:
        # Add all parameter lines except the last (which needs backslash removed)
        script_lines.extend(param_lines[:-1])
        # Add the last parameter line with backslash replaced by the input/output files
        last_param = param_lines[-1].rstrip(" \\")
        script_lines.append(last_param + " \\")

    script_lines.extend(
        [
            f"    {notebook_path} \\",
            f"    {output_notebook}",
            "",
            "# Check if papermill executed successfully",
            "if [ $? -eq 0 ]; then",
            '    echo "=========================================="',
            '    echo "Notebook executed successfully!"',
            f'    echo "Output saved to: {output_notebook}"',
            '    echo "=========================================="',
            "    exit 0",
            "else",
            '    echo "=========================================="',
            '    echo "Error: Notebook execution failed!"',
            '    echo "Check the output files for details"',
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
    "-n",
    "--notebook",
    type=str,
    required=True,
    help="Name of the notebook to run (e.g., calibrated_data_check-postcali) or absolute path to notebook file",
)
@add_block_name_option()
@add_box_option()
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path, resolve_path=True
    ),
    default="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline",
    help="Base directory for notebook output. The final output folder will be <output-path>/BOX<box>/<block-name>/",
    show_default=True,
)
@click.option(
    "-k",
    "--kernel",
    type=str,
    default="meerklass",
    help="Jupyter kernel to use for execution",
    show_default=True,
)
@click.option(
    "-p",
    "--parameters",
    type=(str, str),
    metavar="<NAME VALUE>...",
    multiple=True,
    help="Extra notebook parameters to pass papermill. Can be specified multiple times.",
)
@add_slurm_options()
@add_dry_run_option()
def main(
    notebook: str,
    block_name: str,
    box: str,
    output_dir: Path,
    kernel: str,
    parameters: tuple[tuple[str, str], ...],
    slurm_options: tuple[tuple[str, str], ...],
    dry_run: bool,
) -> None:
    """Generate and submit a Slurm job to execute a MuSEEK Jupyter notebook using papermill.

    The notebook is searched for in multiple locations (in order):
      1. Absolute path to notebook file (if provided)
      2. Installed package (museek/notebooks/<notebook_name>.ipynb)
      3. Current working directory (./<notebook_name>.ipynb)

    \b
    EXAMPLES:
      # Using notebook name (searches in standard locations)
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 --parameters data_path /custom/path/
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 --dry-run
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 --slurm-options mail-user user@uni.edu --slurm-options mail-type ALL

      # Using absolute path to notebook:
      museek_run_notebook --notebook /custom/path/my_notebook.ipynb --block-name 1708972386 --box 6

    \b
    DEFAULT SLURM PARAMETERS:
      Job name:       MuSEEK-Notebook-<block_name>
      Tasks:          1
      CPUs per task:  32
      Memory:         248GB
      Max time:       1 hour
      Log output:         <notebook_name>-<block_name>.log

    """
    # Validate kernel (only when not in dry-run mode to avoid side effects)
    if not dry_run:
        validate_kernel(kernel)

    # Validate notebook exists
    notebook_path = validate_notebook(notebook)

    # Generate the sbatch script
    script_content = generate_sbatch_script(
        notebook_path=notebook_path,
        block_name=block_name,
        box=box,
        base_output_dir=output_dir,
        kernel=kernel,
        notebook_name=notebook,
        parameters=list(parameters),
        slurm_options=list(slurm_options),
    )

    # Submit or display the script
    submit_sbatch_script(script_content, dry_run=dry_run)


if __name__ == "__main__":
    main()
