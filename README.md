# MuSEEK: Multi-dish Signal Extraction and Emission Kartographer

MuSEEK [muˈziːk] is a flexible and easy-to-extend data processing pipeline for multi-instrument autocorrelation radio telescopes currently under development by the MeerKLASS collaboration. It takes the observed (or simulated) TOD in the time-frequency domain as an input and processes it into HEALPix maps while applying calibration and automatically masking RFI.

Several Jupyter notebook templates are also provided for data inspection and verification.

The MuSEEK package has been developed at the [Centre for Radio Cosmology](http://www.cosmology.ethz.ch/research/software-lab.html) at UWC and at the Jodrell Bank Centre for Astrophysics at UoM.
It is inspired by SEEK, developed by the [Software Lab of the Cosmology Research Group](http://www.cosmology.ethz.ch/research/software-lab.html) of the [ETH Institute of Astronomy](http://www.astro.ethz.ch).

The development is coordinated on [GitHub](https://github.com/meerklass/museek) and contributions are welcome.

## Table of Contents

- [Installation](#installation)
    - [Setup Python Virtual Environment](#setup-python-virtual-environment)
    - [Install MuSEEK](#install-museek)
    - [Install Jupyter Kernel](#install-jupyter-kernel)
- [Running A Pipeline](#running-a-pipeline)
    - [Running Locally](#running-locally)
    - [Changing Pipeline Parameters](#changing-pipeline-parameters)
    - [Running as a SLURM Job](#running-as-a-slurm-job)
    - [Examining Results](#examining-results)
- [Anatomy of Pipeline and Plugins](#anatomy-of-pipeline-and-plugins)
    - [Configuration File](#configuration-file)
    - [Plugins](#plugins)
    - [Available Plugins](#available-plugins)
- [Maintainers](#maintainers)

## Installation

### Setup Python Virtual Environment

It is recommended to create a Python Virtual Environment to install MuSEEK. The Python interpreter should be >=3.10 (older versions are not tested).

On the Ilifu computing cluster, one can do the following to create a virtual environment although note that you may also use other methods (e.g. with `conda` or `venv`).

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

### Install MuSEEK

MuSEEK should be installed as an editable package.

First, clone the package,

```bash
git clone git+https://github.com/meerklass/museek.git
```

and install via `pip`

```bash
cd museek
python -m pip install --editable .
```

This will also install the `museek` command.

```bash
which museek
# return /path/to/virtualenv/museek/bin/museek
```

### Install Jupyter Kernel

If you want to run the data inspection notebooks on one of the Jupyter nodes on Ilifu, install the virtual environment Python executable as a Jupyer kernel.

```bash
python -m pip install ipykernel
python -m ipykernel install --name "museek_kernel" --user
```

After relaunching the Jupyter node, `museek_kernel` should now be selectable.

## Running A Pipeline

A MuSEEK pipeline usually consits of several plugins defined in the `museek/museek/plugin`.

Running a pipeline requires a configuration file, which define the order of the plugins to run and their parameters.

The configuration files must be in `museek/museek/config` path of the MuSEEK package that you have installed. These configuration files will likely need to be edited (and thus the reason that the package should be installed with `--editable` flag).

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

Alternatively, parameters can be overrided by passing extra flags to the `museek` command. For example, the following command will run the Demo pipeline, overriding the output folder to `./demo_results`.

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

You can find an `sbatch` script to run the Demo pipeline as an example below, and also in `/museek/script/demo.sh`, but remember to change `/path/to/virtualenv` to your own environment. The allocated ressources in this script are minimal and for demonstration only, see below for a brief guideline on ressource usage.

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

### Examining Results

To quickly access results stored by the pipeline as `pickle` files, the class `ContextLoader` can be used.

## Anatomy of Pipeline and Plugins

### Configuration File

A pipeline is defined by its configuration file, which is technically a Python module file, and usually consists of several `ConfigSection()` instances.

One instance should be called `Pipeline`, which defines the entire pipeline, i.e. the order of the plugins to run. Other instances need to be named to the plugins they belong to. The workflow manager will hand over the correct configuration parameters to each plugin.

For example, the configuration file for the Demo pipeline (`museek/museek/config/demo.py`) looks like the following,

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

## Maintainers
The current maintainers of MuSEEK are:
- Mario Santos (@mariogrs)
- Wenkai Hu (@wkhu-astro)
- Piyanat Kittiwisit (@piyanatk)