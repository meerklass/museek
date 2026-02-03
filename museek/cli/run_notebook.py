"""CLI for running MuSEEK Jupyter notebooks using papermill.

Generates and submits a Slurm job to execute a MuSEEK Jupyter notebook using
papermill. This script creates a temporary sbatch script and submits it to Slurm.
Kernel validation is performed before submission.

Notebook Discovery:
    Notebooks are searched in the following order:
    1. Installed package location (museek/notebooks/) - works for all installs
    2. Current working directory + notebooks/
    3. Absolute path (if provided)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from museek.cli.slurm_utils import submit_sbatch_script


def find_notebooks_directory() -> Path | None:
    """Find the notebooks directory by searching multiple locations.

    Search order:
    1. Installed package location (museek/notebooks/) - works for all install types
    2. Current working directory + notebooks/

    Returns:
        Path to notebooks directory if found, None otherwise.
    """
    search_paths = []

    # 1. Check installed package location (works for all install types)
    try:
        import museek

        pkg_path = Path(museek.__file__).parent
        # Notebooks are now inside the package
        search_paths.append(pkg_path / "notebooks")
    except Exception:
        pass

    # 2. Check current working directory
    search_paths.append(Path.cwd() / "notebooks")

    # Return the first valid notebooks directory found
    for path in search_paths:
        if path.exists() and path.is_dir():
            return path

    return None


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

    If notebook_name is an absolute path, use it directly.
    Otherwise, search for notebooks in standard locations (see find_notebooks_directory).
    """
    # Check if absolute path was provided
    notebook_path = Path(notebook_name)
    if notebook_path.is_absolute():
        if not notebook_path.exists():
            click.echo(f"Error: Notebook not found: {notebook_path}")
            sys.exit(1)
        return notebook_path

    # Search for notebook in standard locations
    notebooks_dir = find_notebooks_directory()

    if notebooks_dir is None:
        click.echo("Error: Could not find notebooks directory.")
        click.echo("Tried searching in:")
        click.echo("  1. Installed package location (museek/notebooks/)")
        click.echo("  2. Current working directory + notebooks/")
        click.echo("")
        click.echo("You can also provide an absolute path to the notebook.")
        sys.exit(1)

    notebook_path = notebooks_dir / f"{notebook_name}.ipynb"

    if not notebook_path.exists():
        click.echo(f"Error: Notebook not found: {notebook_path}")
        click.echo(f"Available notebooks in {notebooks_dir}:")
        notebooks = list(notebooks_dir.glob("*.ipynb"))
        if notebooks:
            for nb in notebooks:
                click.echo(f"  {nb.stem}")
        else:
            click.echo("  No notebooks found")
        sys.exit(1)

    return notebook_path


def generate_sbatch_script(
    notebook_path: Path,
    block_name: str,
    box: str,
    output_path: str,
    kernel: str,
    notebook_name: str,
    parameters: list[tuple[str, str]],
    slurm_options: list[str],
) -> str:
    """Generate the sbatch script content.

    Best-effort uses pathlib to construct output paths and uses the notebook's stem
    for the output filename so absolute notebook paths do not create malformed
    filenames.
    """
    output_dir_path = Path(output_path) / f"BOX{box}" / str(block_name)
    output_dir = str(output_dir_path)
    # Use the notebook name's stem (base name without suffix) for the output file
    output_notebook_path = output_dir_path / f"{Path(notebook_name).stem}_output.ipynb"
    output_notebook = str(output_notebook_path)

    # Build papermill parameters string
    papermill_params = ""
    if parameters:
        param_parts = []
        for param_name, param_value in parameters:
            param_parts.extend(["-p", param_name, param_value])
        papermill_params = " ".join(f'"{p}"' for p in param_parts)

    script_lines = [
        "#!/bin/bash",
        "",
        "#SBATCH --job-name='MuSEEK-Notebook'",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=32",
        "#SBATCH --mem=248GB",
        "#SBATCH --time=01:00:00",
        f"#SBATCH --output=notebook-{block_name}-stdout.log",
        f"#SBATCH --error=notebook-{block_name}-stderr.log",
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
            f'mkdir -p "{output_dir}"',
            "",
            "# Execute notebook using papermill with collected parameters",
            f'papermill -k "{kernel}" \\',
        ]
    )

    if papermill_params:
        script_lines.append(f"    {papermill_params} \\")

    script_lines.extend(
        [
            f'    "{notebook_path}" "{output_notebook}"',
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
    "--notebook",
    required=True,
    help="Name of the notebook to run (e.g., calibrated_data_check-postcali) or absolute path to notebook file",
)
@click.option(
    "--block-name",
    required=True,
    help="Block name or observation ID (e.g., 1708972386)",
)
@click.option(
    "--box",
    required=True,
    help="Box number of this block name (e.g., 6)",
)
@click.option(
    "--output-path",
    default="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline",
    help="Base directory for notebook output. The final output folder will be <output-path>/BOX<box>/<block-name>/",
    show_default=True,
)
@click.option(
    "--kernel",
    default="meerklass",
    help="Jupyter kernel to use for execution",
    show_default=True,
)
@click.option(
    "-p",
    "--parameters",
    type=(str, str),
    multiple=True,
    help="Parameters to pass to the notebook via papermill. Can be specified multiple times.",
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
    notebook: str,
    block_name: str,
    box: str,
    output_path: str,
    kernel: str,
    parameters: tuple[tuple[str, str], ...],
    slurm_options: tuple[str, ...],
    dry_run: bool,
) -> None:
    """Generate and submit a Slurm job to execute a MuSEEK Jupyter notebook using papermill.

    The notebook is searched for in multiple locations (in order):
      1. Installed package (works for: pip install, pip install -e, pip install git+...)
      2. ./notebooks/ (current directory)
      3. Absolute path to the notebook (if provided)

    \b
    EXAMPLES:
      # Using notebook name (searches in standard locations)
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 -p data_path /custom/path/
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 --dry-run
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 --slurm-options --mail-user=user@uni.edu --slurm-options --mail-type=ALL

      # Using absolute path to notebook:
      museek_run_notebook --notebook /custom/path/my_notebook.ipynb --block-name 1708972386 --box 6

    \b
    DEFAULT SLURM PARAMETERS:
      Job name:       MuSEEK-Notebook
      Tasks:          1
      CPUs per task:  32
      Memory:         248GB
      Max time:       1 hour
      Output:         notebook-<block_name>-stdout.log
      Error:          notebook-<block_name>-stderr.log

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
        output_path=output_path,
        kernel=kernel,
        notebook_name=notebook,
        parameters=list(parameters),
        slurm_options=list(slurm_options),
    )

    # Submit or display the script
    submit_sbatch_script(script_content, dry_run=dry_run)


if __name__ == "__main__":
    main()
