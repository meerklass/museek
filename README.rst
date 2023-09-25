=============================
MuSEEK: Multi-dish Signal Extraction and Emission Kartographer
=============================

**MuSEEK** [muˈziːk] is a flexible and easy-to-extend data processing pipeline for multi-instrument autocorrelation
radio telescopes. Once finished, it takes the observed (or simulated) TOD in the time-frequency domain as an input and processes it
into **healpix** maps while applying calibration and automatically masking RFI.

The **MuSEEK** package has been developed at the `Centre for Radio Cosmology` at `UWC` and at the `Jodrell Bank Centre for Astrophysics` at `UoM`.
It is inspired by **SEEK**, developed by the `Software Lab of the Cosmology Research Group <http://www.cosmology.ethz.ch/research/software-lab.html>`_ of the `ETH Institute of Astronomy <http://www.astro.ethz.ch>`_.

The development is coordinated on `GitHub <https://github.com/meerklass/museek>`_ and contributions are welcome. The documentation of **MuSEEK** is not yet available at `readthedocs.org <http://museek.readthedocs.io/>`_ .

Install and Run
-----------------------
`MuSEEK` is run via the workflow engine `Ivory`. Consequently the `Ivory` directory needs to be in the `PYTHONPATH` of your python interpreter.
A shorthand for running `Ivory` is directly inside `MuSEEK` and allows running the plugins like (assuming working directory in the repository root)

.. code-block:: python

    museek museek.config.demo

which is a shorthand for

.. code-block:: python

    python cli/main.py museek.config.demo

The following syntax allows to pass extra arguments to `MuSEEK` which override those from the config file (this allows e.g. running the pipeline on different observations with otherwise the same config):

.. code-block:: python

    python cli/main.py --InPlugin-block-name=1634748682 museek.config.process_data

In this specific case, the parameter `block_name` of the plugin `InPlugin` is replaced with the example `1634748682`.

The python interpreter should be >3.10 (older versions are not tested) and needs to have `requirements.txt` installed.
A popular way of doing this is

.. code-block:: bash

    pip install -r museek/requirements.txt

Or you can do

.. code-block:: bash

    cd museek
    pip install --editable .
    cd ..

You can do the same for `ivory` if it's cloned in the same directory.

.. code-block:: bash

    cd ivory
    pip install --editable .
    cd ..


Plugins
-----------------------
Plugins can be implemented by creating a class inheriting from **Ivory**s `AbstractPlugin`. They need to implement the methods
`run()` and `set_requirements()`.

1. Only one plugin per file is allowed. One plugin can not import another plugin.

2. Naming: CamelCase ending on "Plugin", example: "GainCalibrationPlugin".

3. To have the plugin included in the pipeline, the config file's "Pipeline" entry needs to include the plugin under "plugins".

4. If the plugin requires configuration (most do), the config file needs to contain a section with the same name as the plugin. For more information see section config.

5. Plugins need to define their `Requirement`s in `self.set_requirements()`. The workflow engine will compare these to the set of results that are already produced when the plugin starts and hands them over to the `run()` method.

6. Plugins need to define a `run()` method, which is executed by the workflow engine.

7. Plugins need to run `self.set_result()` to hand their results back to the workflow engine for storage.

Configuration
-----------------------
The configuration file is written in python and consists of `ConfigSection()` instances.
There is one general section called `Pipeline`, which defines the entire pipeline, and each other section needs to share
the name of the plugin it belongs to. The workflow manager will then hand over the correct configuration parameters to
each plugin.

A demonstration config is `museek.config.demo`.

Plugin Requirements
-----------------------
Plugin requirements are encapsulated as `Requirement()` objects, which are mere `NamedTuples`. See the `Requirement` class doc for more information.

Plugin Results
-----------------------
Plugin results need to be defined as `Result()` objects. See the `Result` class doc for more information.

Available Plugins
-----------------------
More information on these are included in their class documentations.

1. Demonstration plugins: `DemoFlipPlugin`, `DemoLoadPlugin` & `DemoPlotPlugin`

2. `InPlugin`

3. `OutPlugin`

4. `SanityCheckObservationPlugin`

5. `AoflaggerPlugin`

6. `KnownRfiPlugin`

7. `NoiseDiodeFlaggerPlugin`

8. `AntennaFlaggerPlugin`

9. `PointSourceFlaggerPlugin`

10. `BandpassPlugin`


Ilifu
-----------------------

The computing cluster `Ilifu <https://docs.ilifu.ac.za/#/>`_ makes many python interpreters available using the `module` command.
Using

.. code:: bash

    module avail

displays all available modules that can be loaded. You can follow the Ilifu `documentation <https://docs.ilifu.ac.za/#/tech_docs/software_environments?id=python-virtual-environments>`_
to create a virtual environment with the python interpreter of your choice, e.g. python/3.10.4.

The following compiles all the commands needed to get up and running with `MuSEEK` on slurm.ilifu.ac.za or a similar system.
You first clone the repositories, create a new python environment, install museek and ivory and create a results folder.

.. code:: bash

    git clone git@github.com:meerklass/museek.git
    git clone git@github.com:meerklass/ivory.git

    module load python/3.10.4
    virtualenv ./environment/museek
    source ./environment/museek/bin/activate
    cd ivory
    pip install --editable .
    cd ../museek
    pip install --editable .
    cd ..

    # for jupyter support
    pip install ipykernel
    ipython kernel install --name "museek_kernel" --user

    deactivate

    mkdir museek/results museek/results/demo



If you have not set up `ssh` keys for your github account, it might be easier to clone the repos like

.. code:: bash

    git clone https://github.com/meerklass/museek.git
    git clone https://github.com/meerklass/ivory.git


Now you are ready to run `MuSEEK`! You can use the `sbatch` command to schedule a job:

.. code:: bash

    sbatch example.sh

You can find an `sbatch` script to start with below, but remember to change `/path/to/project` to your own project's
working directory and `/path/to/virtualenv/` to the directory of your new environment. The allocated ressources in this
script are minimal and for demonstration only, see below for a brief guideline on ressource usage.

.. code:: batch

    #!/bin/bash

    #SBATCH --job-name='MuSEEK'
    #SBATCH --cpus-per-task=1
    #SBATCH --ntasks=1
    #SBATCH --mem=4GB
    #SBATCH --output=museek-stdout.log
    #SBATCH --error=museek-stderr.log
    #SBATCH --time=00:05:00

    echo "Submitting Slurm job"
    /path/to/virtualenv/bin/python /path/to/project/museek/cli/main.py museek.config.demo

Once the job is finished, you can check the results of the demo pipeline in your working directory and in `museek/results/demo`.
To go from the demo pipeline to the real one, you will need to change `museek.config.demo` to the config you want to use, an example
is `museek.config.process_data`. You also need to adjust the ressources in the `sbatch` script depending
on the config. As a rough estimate, processing an entire MeerKAT observation block may be done with
`--cpus-per-task=32`, `--mem=128GB` and `--time=03:00:00`.

You can also run `MuSEEK` in `jupyter` on ilifu! If you followed the manual above, you already have a
kernel installed, it's called `museek_kernel` and can be selected to run a `jupyter` notebook.
To quickly access results stored by the pipeline as a `pickle` file from within a notebook, the class `ContextLoader`
can be used.