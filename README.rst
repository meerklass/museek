=============================
MuSEEK: Multi-dish Signal Extraction and Emission Kartographer
=============================

**MuSEEK** [muˈziːk] is a flexible and easy-to-extend data processing pipeline for multi-instrument autocorrelation
radio telescopes. Once finished, it takes the observed (or simulated) TOD in the time-frequency domain as an input and processes it
into **healpix** maps while applying calibration and automatically masking RFI. The data processing is parallelized using **ivory**'s parallelization scheme.

The **MuSEEK** package has been developed at the `Centre for Radio Cosmology` at `UWC` and at the `Jodrell Bank Centre for Astrophysics` at `UoM`.
It is inspired by **SEEK**, developed by the `Software Lab of the Cosmology Research Group <http://www.cosmology.ethz.ch/research/software-lab.html>`_ of the `ETH Institute of Astronomy <http://www.astro.ethz.ch>`_.

The development is coordinated on `GitHub <https://github.com/meerklass/museek>`_ and contributions are welcome. The documentation of **MuSEEK** is not yet available at `readthedocs.org <http://museek.readthedocs.io/>`_ .

Plugins
-----------------------
Plugins can be implementing by creating a class inheriting from **ivory**s `AbstractPlugin`. They need to implement the methods
`run()` and `set_requirements()`.

1. Only one plugin per file is allowed. One plugin can not import another plugin.

2. Naming: CamelCase ending on "Plugin", example: "GainCalibrationPlugin".

3. To have the plugin included in the pipeline, the config file's "Pipeline" entry needs to include the plugin under "plugins".

4. If the plugin requires configuration (most do), the config file needs to contain a section with the same name as the plugin. For more information see section config.

5. Plugins need to define their `Requirement`s in `self.set_requirements()`. The workflow engine will compare these to the set of results that are already produced when the plugin starts and hands them over to the `run()` method.

6. Plugins need to define a `run()` method, which is executed by the workflow engine.

7. Plugins need to run `self.set_result()` to hand their results back to the workflow engine for storage.

Config
-----------------------
The config file is written in python and consists of `ConfigSection()` instances.
There is one general section called `Pipeline`, which defines the entire pipeline, and each other section needs to share
the name of the plugin it belongs to. The workflow manager will then hand over the correct configuration parameters to
each plugin.

Requirement
-----------------------
Plugin requirements are encapsulated as `Requirement()` objects, which are mere `NamedTuples`. See the `Requirement` class doc for more information.

Result
-----------------------
Plugin results need to be defined as `Result()` objects. See the `Result` class doc for more information.