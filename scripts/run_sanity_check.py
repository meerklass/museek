#!/usr/bin/env python3
import subprocess
import click
from pathlib import Path


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}




@click.command(
    context_settings=CONTEXT_SETTINGS,
    epilog="""
    Examples:

    \b
    Providing `archive-url`, run with venv:
    python museek/script/run_sanity_check.py --venv-path ~/venv/meerklass -u https://archive-gw-1.kat.ac.za/1721838156/1721838156_sdp_l0.full.rdb?token=eyJ...1L6w

    \b
    Providing `block-number` and `token`, run with conda env, save output to none default directory:
    python museek/script/run_sanity_check.py --conda-path ~/miniconda3/envs/meerklass -b 1721838156 -t eyJ...1L6w --context-folder /path/to/different/folder
    """,
)
@click.option(
    "-u",
    "--archive-url",
    type=str,
    help="The full archive url of an observation (preferred input). "
    "`--block-number` and `--token` will be automatically extracted from the url.",
)
@click.option(
    "-b",
    "--block-number",
    type=str,
    help="Observation Capture Block ID, usually a 10-digit number. "
    "Not use if `--archive-url` is provided. "
    "`--token` or `--data-folder` must also be given.",
)
@click.option(
    "-t",
    "--token",
    type=str,
    help="RDB token of the observation block. Required metadata will be pull directly "
    "from the RBD file on the SARAO archive over the network.",
)
@click.option(
    "--context-folder",
    required=True,
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    default="/idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/sanity_checks",
    show_default=True,
    help="Context folder - results will be saved to <context folder>/<block number> "
    "directory. Set to ./results if None.",
)
@click.option(
    "--data-folder",
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    help="Directory containing the correlator data of the given block number, "
    "useful for running from the files on disk in the case that network connection to "
    "the SARAO archive is down",
)
@click.option(
    "--venv-path",
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    help="Path to the Python virtual environment to use, e.g. "
    "/idia/projects/meerklass/virtualenv/museek."
    "Take precedence over `--conda-path`.",
)
@click.option(
    "--conda-path",
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    help="Path to the Python `conda` environment to use, "
    "e.g. ~/miniconda3/envs/meerklass. Not use if `--venv-path` is also given",
)
@click.option(
    "--runtime",
    default="00:05:00",
    show_default=True,
    help="Time string for the slurm job",
)
@click.option(
    "--mem",
    default=10,
    show_default=True,
    help="Memory required in GB for the slurm job",
)
@click.option(
    "--cpus-per-task",
    default=1,
    show_default=True,
    help="cpus per task for the slurm job",
)
@click.option(
    "--requeue/--no-requeue",
    default=True,
    show_default=True,
    help="Whether to requeue when the job fails",
)
def run_sanity_check(
    archive_url,
    block_number,
    token,
    data_folder,
    context_folder,
    venv_path,
    conda_path,
    runtime,
    mem,
    cpus_per_task,
    requeue,
):
    """
    Run a sanity check on a given observation, either by making and submitting a slurm
    script or running locally.
    """
    # Check plugin inputs
    if archive_url is not None:
        block_number = archive_url.split("/")[3]
        token = archive_url.split("?token=")[1]
    else:
        if block_number is None:
            raise ValueError("--block-number must be given.")
        if token is None and data_folder is None:
            raise ValueError("either `--token` or `--data-folder` must be given.")

    # Set up museek command line
    museek_cmd = " ".join(
        [
            "museek",
            f"--InPlugin-block-name={block_number}",
            f"--InPlugin-token={token}" if token is not None else "",
            f"--InPlugin-data-folder={data_folder}" if data_folder is not None else "",
            f"--InPlugin-context-folder={context_folder}",
            "museek.config.sanity_check",
        ]
    )

    # Check Python environment inputs and set up source line
    if venv_path is not None:
        python_env = f"source {venv_path}/bin/activate"
    elif conda_path is not None:
        python_env = "\n".join(
            [
                f"source {conda_path.parents[1]}/bin/activate",
                f"conda activate {conda_path.name}",
            ]
        )
    else:
        raise ValueError("either `--venv-path` or `--conda-path` must be given.")

    # Check that the context folder exists, creating the directory if needed
    (context_folder / f"{block_number}").mkdir(parents=True, exist_ok=True)

    # Check that path to slurm log file exists. If not create it.
    Path("./logs").mkdir(parents=True, exist_ok=True)

    slurm_requeu = "#SBATCH --requeue" if requeue else ""

    program = f"""#!/bin/bash

#SBATCH --job-name='sanity-check-{block_number}'
#SBATCH --output=logs/sanity-check-{block_number}-%j.log
#SBATCH --account=b205-meerklass-ag
#SBATCH --partition=Main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}GB
#SBATCH --time={runtime}
{slurm_requeu}

{python_env}
echo "Using Python Environment: $(which python)"

export OMP_NUM_THREADS={cpus_per_task}
export OPENBLAS_NUM_THREADS={cpus_per_task}
export MKL_NUM_THREADS={cpus_per_task}

# Get museek path and dump the template config for the sake of documenting the run
echo "==== museek sanity check pipeline ===="
config_file=$(python -c "import museek; print(museek.__path__[0])")/config/sanity_check.py
echo "museek config file: ${{config_file}}"
echo "---- beginning of config file ----"
cat ${{config_file}}

echo "---- run parameters ----"
echo "Block Number: {block_number}"
echo "Token: {token}"
echo "Context folder: {context_folder}"
echo "Data folder: {data_folder}"

echo "Executing command: {museek_cmd}"

{museek_cmd}
"""
    sbatch_file = Path("./_run_sanity_check.sbatch").resolve()
    with open(sbatch_file, "w") as fl:
        print(f"Generating an sbatch script, saving it to {sbatch_file}:")
        print("-------BEGINING OF SBATCH-------")
        print(program)
        print("-------END OF SBATCH-------")
        fl.write(program)

    subprocess.run(["sbatch", "_run_sanity_check.sbatch"], check=True)


if __name__ == "__main__":
    run_sanity_check()
