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

import json
import subprocess
import sys
from pathlib import Path

import click

from museek.cli.common import (
    add_block_name_option,
    add_box_option,
    add_dry_run_option,
    add_slurm_options,
    add_venv_option,
)
from museek.cli.slurm_utils import build_sbatch_script, submit_sbatch_script


def get_kernels_info() -> dict[str, dict]:
    """Get information about all available Jupyter kernels.

    Returns:
        Dictionary mapping kernel names to their spec information,
        including the Python executable path if available.
    """
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            kernels_info = {}

            for kernel_name, kernel_data in data.get("kernelspecs", {}).items():
                spec_file = Path(kernel_data["resource_dir"]) / "kernel.json"
                if spec_file.exists():
                    with open(spec_file) as f:
                        spec = json.load(f)
                        kernels_info[kernel_name] = {
                            "resource_dir": kernel_data["resource_dir"],
                            "spec": spec,
                            "argv": spec.get("argv", []),
                        }
            return kernels_info
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def find_kernel_for_venv(venv_path: Path) -> str | None:
    """Find a Jupyter kernel associated with the given virtual environment.

    Searches in two ways:
    1. Checks kernels visible to current jupyter installation
    2. Checks for kernels installed directly in the venv's share directory

    Args:
        venv_path: Path to the virtual environment

    Returns:
        Name of the kernel if found, None otherwise
    """
    venv_path_resolved = venv_path.resolve()

    # Expected Python executable in the venv
    venv_python = venv_path / "bin" / "python"
    venv_python_resolved = venv_python.resolve() if venv_python.exists() else None

    # Strategy 1: Check kernels visible to current jupyter
    kernels_info = get_kernels_info()

    for kernel_name, info in kernels_info.items():
        # Check if kernel is installed inside this venv
        resource_dir = Path(info.get("resource_dir", ""))
        try:
            resource_dir_resolved = resource_dir.resolve()
            if venv_path_resolved in resource_dir_resolved.parents:
                return kernel_name
        except (OSError, RuntimeError):
            pass

        # Check if kernel's Python executable matches venv's Python (for absolute paths)
        if venv_python_resolved:
            argv = info.get("argv", [])
            if argv:
                kernel_python = Path(argv[0])
                if kernel_python.is_absolute():
                    try:
                        kernel_python_resolved = kernel_python.resolve()
                        if kernel_python_resolved == venv_python_resolved:
                            return kernel_name
                    except (OSError, RuntimeError):
                        pass

    # Strategy 2: Check for kernels installed in venv's share directory
    # (These might not be visible to current jupyter if running from different env)
    venv_kernels_dir = venv_path / "share" / "jupyter" / "kernels"
    if venv_kernels_dir.exists():
        # Look for kernel directories
        for kernel_dir in venv_kernels_dir.iterdir():
            if kernel_dir.is_dir():
                kernel_json = kernel_dir / "kernel.json"
                if kernel_json.exists():
                    # Found a kernel! Return its name
                    return kernel_dir.name

    return None


def list_available_kernels() -> None:
    """Print a list of available Jupyter kernels."""
    click.echo("Available Jupyter kernels:")
    subprocess.run(["jupyter", "kernelspec", "list"], check=False)


def validate_kernel(kernel: str) -> None:
    """Check if the specified kernel is available.

    Args:
        kernel: Name of the kernel to validate

    Raises:
        SystemExit: If the kernel is not available
    """
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
        click.echo("Error: jupyter command not found")
        click.echo("Please install Jupyter: pip install jupyter")
        sys.exit(1)

    click.echo(f"Error: Kernel '{kernel}' not found")
    list_available_kernels()
    sys.exit(1)


def determine_kernel(venv_path: Path, kernel: str | None) -> str:
    """Determine which kernel to use for notebook execution.

    If a kernel is explicitly provided, validate and use it.
    Otherwise, try to find a kernel associated with the venv.

    Args:
        venv_path: Path to the virtual environment
        kernel: Explicitly specified kernel name (or None)

    Returns:
        Name of the kernel to use

    Raises:
        SystemExit: If no suitable kernel can be found
    """
    if kernel:
        # Use explicitly specified kernel
        click.echo(f"Using specified kernel: {kernel}")
        validate_kernel(kernel)
        return kernel

    # Try to find kernel associated with venv
    click.echo(f"Searching for kernel associated with venv: {venv_path}")
    detected_kernel = find_kernel_for_venv(venv_path)

    if detected_kernel:
        click.echo(f"Found kernel '{detected_kernel}' for venv")
        return detected_kernel

    # No kernel found
    click.echo(f"Error: No kernel found for venv: {venv_path}")
    click.echo("")
    click.echo("You can either:")
    click.echo("  1. Specify a kernel explicitly with --kernel option")
    click.echo("  2. Install a kernel for your venv:")
    click.echo(f"     source {venv_path}/bin/activate")
    click.echo("     python -m ipykernel install --user --name <kernel_name>")
    click.echo("")
    list_available_kernels()
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
        if not notebook_path.is_file():
            click.echo(f"Error: Notebook path is not a file: {notebook_path}")
            sys.exit(1)
        if notebook_path.suffix != ".ipynb":
            click.echo(f"Error: Notebook must be a .ipynb file: {notebook_path}")
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
    venv_path: Path,
    parameters: list[tuple[str, str]],
    slurm_options: list[str],
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
    param_lines.append(f'    -p block_name "{block_name}" \\')
    data_path = output_dir / f"{block_name}"
    param_lines.append(f'    -p data_path "{data_path.as_posix()}" \\')
    # Add any additional parameters provided by the user
    for param_name, param_value in parameters:
        param_lines.append(f'    -p {param_name} "{param_value}" \\')

    # Define default SLURM options
    default_slurm_options = [
        f"--job-name='MuSEEK-Notebook-{block_name}'",
        "--ntasks=1",
        "--cpus-per-task=32",
        "--mem=248GB",
        "--time=01:00:00",
        f"--output={Path(notebook_name).stem}-{block_name}.log",
    ]

    # Build job-specific body
    body_lines = [
        'echo "Papermill version: $(papermill --version)"',
        "",
        "# Log job information",
        'echo "=========================================="',
        'echo "Executing MuSEEK Notebook"',
        'echo "=========================================="',
        f'echo "Notebook:      "{notebook_path}""',
        f'echo "Block name:    {block_name}"',
        f'echo "Box:           {box}"',
        f'echo "Kernel:        {kernel}"',
        f'echo "Output:        "{output_notebook}""',
        'echo "=========================================="',
        "",
        "# Create output directory",
        f'mkdir -p "{output_dir}"',
        "",
        "# Execute notebook using papermill with collected parameters",
        f"papermill -k {kernel} \\",
    ]

    # Add papermill parameters
    if param_lines:
        body_lines.extend(param_lines[:-1])
        last_param = param_lines[-1].rstrip(" \\")
        body_lines.append(last_param + " \\")

    body_lines.extend(
        [
            f'    "{notebook_path}" \\',
            f'    "{output_notebook}"',
        ]
    )

    return build_sbatch_script(
        default_slurm_options=default_slurm_options,
        user_slurm_options=slurm_options,
        venv_path=venv_path,
        body_lines=body_lines,
        success_message=f'Notebook executed successfully!\\nOutput saved to: "{output_notebook}"',
        error_message="Notebook execution failed!\\nCheck the output files for details",
    )


@click.command(
    context_settings=dict(help_option_names=["-h", "--help"], max_content_width=100)
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
    help="Base directory for notebook output. The final output folder will be <output-dir>/BOX<box>/<block-name>/",
    show_default=True,
)
@add_venv_option()
@click.option(
    "-k",
    "--kernel",
    type=str,
    default=None,
    help="Jupyter kernel to use for execution. If not specified, auto-detects from venv.",
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
    kernel: str | None,
    venv: Path,
    parameters: tuple[tuple[str, str], ...],
    slurm_options: tuple[str, ...],
    dry_run: bool,
) -> None:
    """Generate and submit a Slurm job to execute a MuSEEK Jupyter notebook using papermill.

    This command is desined specifically for post-calibration and observer notebooks,
    thus requring --block-name and --box parameters, but in principle it can be used
    to run any Jupyter notebook on a SLURM cluster. The two required parameters will
    simply be written at the top of the notebook without being used in that case.

    The notebook is searched for in multiple locations (in order):
      1. Absolute path to notebook file (if provided)
      2. Installed package (museek/notebooks/<notebook_name>.ipynb)
      3. Current working directory (./<notebook_name>.ipynb)

    \b
    EXAMPLES:
      # Using notebook name (searches in standard locations):
      # Standard usage for post-calibration notebook with block name and box:
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6
      # Dry run to show the generated sbatch script without submitting:
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 \\
        --dry-run
      # Passing additional parameters to the notebook (e.g., data_path):
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 \\
        --parameters data_path /custom/path/
      # Passing additional SLURM options (e.g., email notification):
      museek_run_notebook --notebook calibrated_data_check-postcali --block-name 1708972386 --box 6 \\
        --slurm-options --mail-user=user@uni.edu --slurm-options --mail-type=ALL
      # Using absolute path to notebook:
      museek_run_notebook --notebook /custom/path/my_notebook.ipynb --block-name 1708972386 --box 6

    \b
    DEFAULT SLURM PARAMETERS:
      Job name:       MuSEEK-Notebook-<block_name>
      Tasks:          1
      CPUs per task:  32
      Memory:         248GB
      Max time:       1 hour
      Log output:     <notebook_name>-<block_name>.log

    """
    # Validate notebook exists
    notebook_path = validate_notebook(notebook)

    # Determine kernel (always, as detection is read-only with no side effects)
    kernel_to_use = determine_kernel(venv, kernel)

    # Generate the sbatch script
    script_content = generate_sbatch_script(
        notebook_path=notebook_path,
        block_name=block_name,
        box=box,
        base_output_dir=output_dir,
        kernel=kernel_to_use,
        notebook_name=notebook,
        venv_path=venv,
        parameters=list(parameters),
        slurm_options=list(slurm_options),
    )

    # Submit or display the script
    submit_sbatch_script(script_content, dry_run=dry_run)


if __name__ == "__main__":
    main()
