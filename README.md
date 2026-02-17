# MuSEEK: Multi-dish Signal Extraction and Emission Kartographer

MuSEEK [muˈziːk] is a flexible and easy-to-extend data processing pipeline for multi-instrument autocorrelation radio telescopes currently under development by the MeerKLASS collaboration. It takes the observed (or simulated) TOD in the time-frequency domain as an input and processes it into HEALPix maps while applying calibration and automatically masking RFI. Several Jupyter notebook templates are also provided for data inspection and verification.

If you want to simply run the UHF pipeline and post-calibration notebooks on Ilifu, you can skip to [Processing UHF data on Ilifu with `museek_run_process_uhf_band` Script](#processing-uhf-data-on-ilifu-with-museek_run_process_uhf_band-script) and [Running the Notebook with `museek_run_notebook` Script](#running-the-notebook-with-museek_run_notebook-script)

The MuSEEK package has been developed at the [Centre for Radio Cosmology](https://sites.google.com/uwc.ac.za/uwcastrophysics/research/crc?authuser=0) at the University of the Western Cape and at the Jodrell Bank Centre for Astrophysics at UoM.
It is inspired by SEEK, developed by the [Software Lab of the Cosmology Research Group](http://www.cosmology.ethz.ch/research/software-lab.html) of the [ETH Institute of Astronomy](http://www.astro.ethz.ch).

The development is coordinated on [GitHub](https://github.com/meerklass/museek) and contributions are welcome.

## Table of Contents

- [Installation](#installation)
    - [General Use](#general-use)
        - [1. Pre-configued Python virtual Environment on Ilifu](#1-pre-configued-python-virtual-environment-on-ilifu)
        - [2. Manual installation directly from GitHub](#2-manual-installation-directly-from-github)
    - [Development Setup](#development-setup)
        - [1. Setup a Python Virtual Environment](#1-setup-a-python-virtual-environment)
        - [2. Manually install MuSEEK in editable mode](#2-manually-install-museek-in-editable-mode)
    - [Using MuSEEK in a Jupyter Notebook](#using-museek-in-a-jupyter-notebook)
- [Available Commands](#available-commands)
- [Running A Pipeline](#running-a-pipeline)
    - [Running Locally](#running-locally)
    - [Changing Pipeline Parameters](#changing-pipeline-parameters)
    - [Running as a SLURM Job](#running-as-a-slurm-job)
    - [Processing UHF data on Ilifu with `museek_run_process_uhf_band` Script](#processing-uhf-data-on-ilifu-with-museek_run_process_uhf_band-script)
    - [Examining Results](#examining-results)
- [Anatomy of Pipeline and Plugins](#anatomy-of-pipeline-and-plugins)
    - [Configuration File](#configuration-file)
    - [Plugins](#plugins)
    - [Available Plugins](#available-plugins)
- [Notebooks](#notebooks)
    - [Running the Notebook with Jupyter](#running-the-notebook-with-jupyter)
    - [Running the Notebook with Papermill](#running-the-notebook-with-papermill)
    - [Running the Notebook with `museek_run_notebook` Script](#running-the-notebook-with-museek_run_notebook-script)
- [Contributing](#contributing)
    - [Code Formatting Guidelines](#code-formatting-guidelines)
    - [Pull Request Guidelines](#pull-request-guidelines)
- [Maintainers](#maintainers)

## Installation

### General Use

If you only need to run pre-existing MuSEEK pipeline or notebooks with no need for additional codes or pipelines development, there are two simple installation options.

#### 1. Pre-configured Python virtual Environment on Ilifu

If you are on Ilifu, you can use the shared `meerklass` Python virtual environment that has been pre-configed with most MeerKLASS-related Python modules, including MuSEEK, as well as other modules that you might ever need. Simply source the activation file as below.

```
source /idia/projects/meerklass/virtualenv/meerklass/bin/activate
```

If you need any other modules installed in this shared environment, please contact Boom (@piyanatk).

#### 2. Manual installation directly from GitHub

You will need `python>=3.10,<3.13` and `pip` for this.

Then you can simply do

```
pip install git+https://github.com/meerklass/museek.git
```

This will install the latest version of MuSEEK that has been released on the GitHub repository.

You will probably want to do this inside a Python virtual environment. Read the next section for details.

### Development Setup

If you need to develop new features, plugins or pipelines, or make modification to MuSEEK, you will have to setup your own Python virtual environment and manually install MuSEEK, preferably in editing mode. Please follow the guide below and also check [Contributing](#contributing) section.

#### 1. Setup a Python Virtual Environment

On Ilifu, the `virtualenv` command can be used although you may also use other tools (e.g. `conda` or `venv`).

```bash
module load python/3.10.16
virtualenv /path/to/virtualenv/museek
```

This will create a virtual environment named "museek" at the specified path. The environment can then be activated with,

```bash
source /path/to/virtualenv/museek/bin/activate
```

You will see that your command prompt is being prepended with the name of the virtual environment,

```bash
(museek) $
```

It is a good idea to check that you are using Python in the virtual environment at this point

```python
which python
# Should return /path/to/virtualenv/museek/bin/python
```

Installing MuSEEK required `pip`, so it is a good time to upgrade it now.

```bash
python -m pip install --upgrade pip
```

You are now ready to install MuSEEK.

#### 2. Manually install MuSEEK in editable mode

First, clone the package,

```bash
git clone git+https://github.com/meerklass/museek.git
```

MuSEEK can then be installed with `pip`. It is recommended that that package is installed with `--editable` (or short-handed as `-e`) flag when developing new features as all changes will be reflected without having to re-install the package.

```bash
cd museek
python -m pip install -e .[test,dev]
```

The `[test,dev]` option tells `pip` to install all optional dependencies, including for unit testing (`test`) and development tools (`dev`). This will also install `pre-commit` and `ruff` to help with code formatting.

### Using MuSEEK in a Jupyter Notebook

If you want to use MuSEEK on one of the Jupyter nodes on Ilifu (or local Jupyter installation) or run the data inspection notebooks with the `museek_run_notebook` script, you will have to additionally install MuSEEK as Jupyter kernel. After completing the standard installation above, do:

```bash
python -m pip install ipykernel
python -m ipykernel install --name "museek_kernel" --user
```

 `museek_kernel` should now be selectable from Jupyter (may need refreshing).

 ## Available Commands

 Installing MuSEEK will install the `museek` command and console scripts `museek_run_process_uhf_band` and `museek_run_notebook`.

```bash
which museek
# should return /path/to/virtualenv/museek/bin/museek

which museek_run_process_uhf_band
# should return /path/to/virtualenv/museek/bin/museek_run_process_uhf_band

which museek_run_notebook
# should return /path/to/virtualenv/museek/bin/museek_run_notebook
```

The following sections describe how they work.

## Running A Pipeline

A MuSEEK pipeline usually consits of several plugins defined in the [museek/plugin](museek/plugin/).

Running a pipeline requires a configuration file, which define the order of the plugins to run and their parameters.

The configuration files must be in [museek/config](museek/config/) path of this package. These configuration files will likely need to be edited (and thus the reason that the package should be installed with `--editable` flag).

Several pipeline have been defined. Notably, there is one for a demo purpose, and ones for L band and UHF band data processing.

Pipelines are run with the [Ivory](https://github.com/meerklass/ivory) workflow engine, which should have been installed along side MuSEEK and symlinked to the `museek` command.

### Running Locally

To run a pipeline on your local machine or a compute/Jupyter node on ilifu, the `museek` command can be simply executed, providing a relative path to the configuration file within MuSEEK directory.

For example, to run the Demo pipeline defined in `museek/museek/config/demo.py`, execute the following command,

```python
museek museek.config.demo
```

Note that the path to the configuration file must be provided in the Python import style format, i.e. replacing `/` with `.` and discarding `.py` extension. This is due to the restriction in the Ivory workflow engine, which we hope to fix in future releases.

### Changing Pipeline Parameters

To make change to a large numbers of plugins' parameters within a pipeline, it is recommended that the configuration file is copied and edited.

Alternatively, parameters can be overridden by passing extra flags to the `museek` command. For example, the following command will run the Demo pipeline, overriding the output folder to `./demo_results`.

```python
mkdir demo_results
museek --DemoLoadPlugin-context-folder=./demo_results museek.config.demo 
```

### Running as a SLURM Job

For actual data processing on Ilifu, you will want to submit a slurm job to run the pipeline.

You can use the `sbatch` command to schedule a job:

```bash
sbatch demo.sh
```

You can find an `sbatch` script to run the Demo pipeline as an example below, but remember to change `/path/to/virtualenv` to your own environment. The allocated ressources in this script are minimal and for demonstration only, see below for a brief guideline on ressource usage.

```bash
#!/bin/bash

#SBATCH --job-name='MuSEEK-demo'
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --output=museek-stdout.log
#SBATCH --error=museek-stderr.log
#SBATCH --time=00:05:00

source /path/to/virtualenv/museek/bin/activate
echo "Using a Python virtual environment: $(which)"

echo "Creating output directory"
mkdir -p demo_results

echo "Submitting Slurm job"
museek --DemoLoadPlugin-context-folder=demo_results museek.config.demo
```

Once the job is finished, you can check the results of the demo pipeline in your working directory and in `demo_results`.

To adopt this script to the real pipeline, you will need to change `museek.config.demo` to the config you want to use, e.g. `museek.config.process_l_band`. You also need to adjust the ressources in the `sbatch` script depending on the config. As a rough estimate, processing an entire MeerKAT observation block may be done with `--cpus-per-task=32`, `--mem=128GB` and `--time=03:00:00`.

### Processing UHF data on Ilifu with `museek_run_process_uhf_band` Script

The `museek_run_process_uhf_band` script provides a streamlined command-line interface for running the process UHF pipeline on Ilifu. It automatically creates and submits SLURM jobs for you.

```
$ museek_run_process_uhf_band --help
MuSEEK UHF Band Processing Script

This script generates and submits a Slurm job to process UHF band data using 
the MuSEEK pipeline.

USAGE:
  museek_run_process_uhf_band --block-name <block_name> --box <box_number> 
                            [--base-context-folder <path>] [--data-folder <path>]
                            [--slurm-options <options>] [--dry-run]

OPTIONS:
  --block-name <block_name>
      (required) Block name or observation ID (e.g., 1675632179)

  --box <box_number>
      (required) Box number of this block name (e.g., 6)

  --base-context-folder <path>
      (optional) Path to the base context/output folder
      The final context folder will be <base-context-folder>/BOX<box>/<block-name>
      Default: /idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline

  --data-folder <path>
      (optional) Path to raw data folder
      Default: /idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/raw

  --slurm-options <options>
      (optional) Additional SLURM options to pass to sbatch
      Pass raw SLURM options exactly as you would in a batch script
      Can be specified multiple times for multiple flags
      Supports flags with values (--mail-user=user@domain.com) and flags without values (--exclusive)
      Examples: 
        Single: --slurm-options --time=72:00:00
        Multiple flags: --slurm-options --mail-user=user@domain.com --slurm-options --mail-type=ALL --slurm-options --exclusive

  --dry-run
      (optional) Show the generated sbatch script without submitting

  --help
      Display this help message

EXAMPLES:
  museek_run_process_uhf_band --block-name 1675632179 --box 6
  museek_run_process_uhf_band --block-name 1675632179 --box 6 --base-context-folder /custom/pipeline
  museek_run_process_uhf_band --block-name 1675632179 --box 6 --dry-run
  museek_run_process_uhf_band --block-name 1675632179 --box 6 --slurm-options --mail-user=user@uni.edu --slurm-options --mail-type=ALL --slurm-options --time=72:00:00
```


### Examining Results

To access results stored by the pipeline as `pickle` files, the class `ContextLoader` can be used.

## Anatomy of Pipeline and Plugins

### Configuration File

A pipeline is defined by its configuration file, which is technically a Python module file, and usually consists of several `ConfigSection()` instances.

One instance should be called `Pipeline`, which defines the entire pipeline, i.e. the order of the plugins to run. Other instances need to be named to the plugins they belong to. The workflow manager will hand over the correct configuration parameters to each plugin.

For example, the configuration file for the Demo pipeline ([museek/config/demo.py](museek/config/demo.py)) looks like the following,

```python
from ivory.utils.config_section import ConfigSection

Pipeline = ConfigSection(
    plugins=[
        "museek.plugin.demo.demo_load_plugin",
        "museek.plugin.demo.demo_flip_plugin",
        "museek.plugin.demo.demo_plot_plugin",
        "museek.plugin.demo.demo_joblib_plugin",
    ],
)

DemoLoadPlugin = ConfigSection(
    url="https://cdn.openai.com/dall-e-2/demos/text2im/astronaut/horse/photo/9.jpg",
    context_file_name="context.pickle",
    context_folder="./"
)

DemoPlotPlugin = ConfigSection(do_show=False, do_save=True)

DemoFlipPlugin = ConfigSection(do_flip_right_left=True, do_flip_top_bottom=True)

DemoJoblibPlugin = ConfigSection(n_iter=10, n_jobs=2, verbose=0)
```

Here, the `DemoLoadPlugin`, `DemoFlipPlugin`, `DemoPlotPlugin`, and `DemoJoblibPlugin` will be run in that order.

### Plugins

Plugins can be implemented by creating a class inheriting from the Ivory's `AbstractPlugin` abstract class. The inherited class must override the
`run()` and `set_requirements()` methods.

In addition, take note of the following requirements:

1. Only one plugin per file is allowed. One plugin can not import another plugin.

2. Naming: CamelCase ending on "Plugin", example: "GainCalibrationPlugin".

3. To have the plugin included in the pipeline, the config file's "Pipeline" entry needs to include the plugin under "plugins".

4. If the plugin requires configuration (most do), the config file needs to contain a section with the same name as the plugin.

5. Plugins need to define their requirements in `self.set_requirements()`. The workflow engine will compare these to the set of results that are already produced when the plugin starts and hands them over to the `run()` method. The requirements are encapsulated as `Requirement` objects, which are mere `NamedTuples`. See the docstring of the `Requirement` class for more information.

6. Plugins need to define a `self.run()` method, which is executed by the workflow engine.

7. Plugins need to define and run `self.set_result()` to hand their results back to the workflow engine for storage, which will be passed to the next plugin in the pipeline. Plugin results need to be defined as `Result` objects. See the `Result` class doc for more information.

### Available Plugins

More information on these are included in their class documentations.

1. Demonstration plugins: `DemoFlipPlugin`, `DemoLoadPlugin` & `DemoPlotPlugin`
2. `InPlugin`
3. `OutPlugin`
4. `NoiseDiodeFlaggerPlugin`
5. `KnownRfiPlugin`
6. `RawdataFlaggerPlugin`
7. `ScanTrackSplitPlugin`
8. `PointSourceFlaggerPlugin`
9. `AoflaggerPlugin`
10. `AoflaggerSecondRunPlugin`
11. `AntennaFlaggerPlugin`
12. `NoiseDiodePlugin`
13. `GainCalibrationPlugin`
14. `AoflaggerPostCalibrationPlugin`
15. `SanityCheckObservationPlugin`
16. other plugins for 'calibrator', 'zebra', and 'standing wave', but they are not finished


## Notebooks

MuSEEK now come with a few Jupyter notebook templates for data inspection, which are now run after `process_uhf_band` pipeline has been run on the data. 

### Running the Notebook with Jupyter

The notebooks can be copy and run on Jupyter, with the Jupyter Hub on Ilifu (jupyter.ilifu.ac.za), for example. This is only recommend for experimenting with the notebooks.

### Running the Notebook with Papermill

Since version `0.3.0`, we have adopted [papermill](https://papermill.readthedocs.io/en/latest/) as an engine for executing the notebook templates through a command-line interface. This allow the notebook to be run outside Jupyter while modifying the required parameters dynamically, which is more suitable for running the post-calibration notebooks on hundreads of data blocks, for example. The `papermill` command should be automatically installed when you install MuSEEK as it is one of the requirements.

To execute the notebook, first make sure that you have [installed a Jupyter kernel](#using-museek-in-a-jupyter-notebook). For example, if you are using the shared `meerklass` environment on Ilifu, running the two command below will install a Jupyter kernel named `meerklass` for using with `papermill` (and jupyter.ilifu.ac.za)

```
source /idia/projects/meerklass/virtualenv/meerklass/bin/activate
python -m ipykernel install --user --name meerklass
```

Then to execute a notebook, simply do, for example,

```
papermill -k meerklass -p block_name 12345678 notebooks/calibrated_data_check-postcali.ipynb output_notebook.ipynb
```

Here, we tell `papermill` to run the `calibrated_data_check-postcali.ipynb` notebook using the `meerklass` kernel that we just installed, overidding the default `block_name` parameter in the notebook with `12345678` and saved the output notebook as `output_notebook.ipynb`.

Note: As of v0.4.1, MuSEEK ships notebook templates as package data under `museek/notebooks/`. After installing MuSEEK (including via `pip install git+https://github.com/meerklass/museek.git`) you can run templates by name with `museek_run_notebook --notebook <name>` or provide an absolute path to any notebook file. This makes using the templates from shared/non-editable installs straightforward.

To figure out which parameters in the notebook can be passed thorugh `papermill`, use the `--help-notebook` tag. For example,

```bash
$ papermill --help-notebook notebooks/calibrated_data_check-postcali.ipynb 
Usage: papermill [OPTIONS] NOTEBOOK_PATH [OUTPUT_PATH]

Parameters inferred for notebook 'notebooks/calibrated_data_check-postcali.ipynb':
  block_name: str (default "1708972386")
  data_name: str (default "aoflagger_plugin_postcalibration.pickle")
  data_path: str (default "/idia/projects/hi_im/uhf_2024/pipeline/")
```

Check out [papermill documentation](https://papermill.readthedocs.io/en/latest/index.html) and its CLI help text for more information.

```
papermill --help
```

### Running the Notebook with `museek_run_notebook` Command

The `museek_run_notebook` command further streamlines the execution of the notebook via papermill on Ilifu or a compute cluster. It provides a wrapper to the papermill command and dynamically generates and submits SLURM jobs. It will find the notebook "template" in the MuSEEK package currently installed in your Python environment with name matching `--notebook` option, or you may provide an absolute path to any notebook file. For MeerKLASS standard analysis on Ilifu, user only have to run the following commands, providing the block name and box number.

```
source /idia/projects/meerklass/virtualenv/meerklass/bin/activate
museek_run_notebook --notebook calibrated_data_check-postcali --block-name <block-name> --box <box-number>
```

This will create an SBATCH script and submit a SLURM job for you, executing the notebook while overriding the block name and box number, and saving the output to the correct path.

```
$ museek_run_notebook -h
Usage: museek_run_notebook [OPTIONS]

  Generate and submit a Slurm job to execute a MuSEEK Jupyter notebook using papermill.

  This command is desined specifically for post-calibration and observer notebooks, thus requring
  --block-name and --box parameters, but in principle it can be used to run any Jupyter notebook
  on a SLURM cluster. The two required parameters will simply be written at the top of the
  notebook without being used in that case.

  The notebook is searched for in multiple locations (in order):   1. Absolute path to notebook
  file (if provided)   2. Installed package (museek/notebooks/<notebook_name>.ipynb)   3. Current
  working directory (./<notebook_name>.ipynb)

  EXAMPLES:
    # Using notebook name (searches in standard locations):
    # Standard usage for post-calibration notebook with block name
    # and box:
    museek_run_notebook --notebook calibrated_data_check-postcali \
      --block-name 1708972386 --box 6
    # Dry run to show the generated sbatch script without submitting:
    museek_run_notebook --notebook calibrated_data_check-postcali \
      --block-name 1708972386 --box 6 --dry-run
    # Passing additional parameters to the notebook (e.g., data_path):
    museek_run_notebook --notebook calibrated_data_check-postcali \
      --block-name 1708972386 --box 6 --parameters data_path \
      /custom/path/
    # Passing additional SLURM options (e.g., email notification):
    museek_run_notebook --notebook calibrated_data_check-postcali \
      --block-name 1708972386 --box 6 \
      --slurm-options --mail-user=user@uni.edu \
      --slurm-options --mail-type=ALL
    # Using absolute path to notebook:
    museek_run_notebook --notebook /custom/path/my_notebook.ipynb \
      --block-name 1708972386 --box 6

  DEFAULT SLURM PARAMETERS:
    Job name:       MuSEEK-Notebook-<block_name>
    Tasks:          1
    CPUs per task:  32
    Memory:         248GB
    Max time:       1 hour
    Log output:     <notebook_name>-<block_name>.log

Options:
  -n, --notebook TEXT             Name of the notebook to run (e.g., calibrated_data_check-
                                  postcali) or absolute path to notebook file  [required]
  -b, --block-name TEXT           Block name or observation ID (e.g., 1708972386)  [required]
  -x, --box TEXT                  Box number of this block name (e.g., 6)  [required]
  -d, --data-path DIRECTORY       Base path of the data for the notebook. Same as `data_path`
                                  parameter in calibrated_data_check*.ipynb notebooks.  [default:
                                  /idia/projects/meerklass/MEERKLASS-1/uhf_data/XLP2025/pipeline]
  -o, --output-dir DIRECTORY      Directory for notebook output. If not specify, output will be
                                  saved to <data-path>/BOX<box>/<block-name>/
  -v, --venv DIRECTORY            Path to Python virtual environment. Use the default shared
                                  meerklass environment on ilifu if not specified.  [default:
                                  /idia/projects/meerklass/virtualenv/meerklass]
  -k, --kernel TEXT               Jupyter kernel to use for execution. If not specified, auto-
                                  detects from venv.
  -p, --parameters <NAME VALUE>...
                                  Extra notebook parameters to pass papermill. Can be specified
                                  multiple times.
  -s, --slurm-options TEXT        Additional SLURM options to pass to sbatch (e.g., --exclusive,
                                  --mail-user=user@domain.com). Can be specified multiple times.
  --dry-run                       Show the generated sbatch script without submitting
  -h, --help                      Show this message and exit.
```

## Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given. 

You can contribute in many ways:

* __Report Bugs__: Please report potential bugs by [opening an issue](https://github.com/meerklass/museek/issues/new) on MuSEEK GitHub repository, and include:
    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.
* __Fix Bugs, Implement Features, or Write Documentation__: Fork the repository (or ask to be added as a collaborator) and contribute a Pull Request.
* __Submit Feedback__: If you are proposing a feature:
    * Explain in detail how it should work.
    * Keep the scope as narrow as possible, to make it easier to implement.
    * Remember that this is a volunteer-driven project

### Code Formatting Guidelines

Please try to follow standard Python code formatting (PEP8). The easiest way to achieve that is to use `ruff`, which should be available when installing MuSEEK in development mode.

`ruff` can check and automatically format code to align with the recommended Python style. Simple run these two commands in the root directory of the repository.

```
ruff check --statistics .
ruff format .
```

The first will print a summary of violation from the style guide. The second will auto-format the code. Note that the auto-formatting will not work on lines with long string or docstrings. You will have to manually format that.

A pre-commit hook, which is basically a little automated script that is run when `git` makes a new commit, is also provided in the repository to automatically run `ruff` upon making a new commit. To use it, simply install pre-commit hooks:

```
pre-commit install
```

Then everytime you do `git commit`, `ruff` will automatically format and lint your code before each commit.

### Pull Request Guidelines
Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. The docs should be updated or extended accordingly. Add any new plugins to the list in README.md.
3. The pull request should work for Python 3.10.
4. Make sure that the tests pass.
5. Code must be formatted with ``ruff format`` and pass ``ruff check``.


## Maintainers

The current maintainers of MuSEEK are:
- Mario Santos (@mariogrs)
- Wenkai Hu (@wkhu-astro)
- Piyanat Kittiwisit (@piyanatk)
- Geoff Murphy (@GeoffMurphy)