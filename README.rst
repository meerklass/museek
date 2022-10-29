=============================
MuSEEK: Multi-dish Signal Extraction and Emission Kartographer
=============================

.. image:: https://travis-ci.org/cosmo-ethz/seek.png?branch=master
        :target: https://travis-ci.org/cosmo-ethz/seek
        
.. image:: https://coveralls.io/repos/cosmo-ethz/seek/badge.svg
  		:target: https://coveralls.io/r/cosmo-ethz/seek

.. image:: https://readthedocs.org/projects/seek/badge/?version=latest
		:target: http://seek.readthedocs.io/en/latest/?badge=latest
		:alt: Documentation Status
		
.. image:: http://img.shields.io/badge/arXiv-1607.07443-orange.svg?style=flat
        :target: http://arxiv.org/abs/1607.07443

**MuSEEK** [muˈziːk] is a flexible and easy-to-extend data processing pipeline for multi-instrument autocorrelation radio telescopes. It takes the observed (or simulated) TOD in the time-frequency domain as an input and processes it into *healpix*maps while applying calibration and automatically masking RFI. The data processing is parallelized using *ivy's* parallelization scheme.

The **MuSEEK** package has been developed at the `Centre for Radio Cosmology` at UWC and at the `Jodrell Bank Centre for Astrophysics` at `UoM`.
It is based on **SEEK** developed by the `Software Lab of the Cosmology Research Group <http://www.cosmology.ethz.ch/research/software-lab.html>`_ of the `ETH Institute of Astronomy <http://www.astro.ethz.ch>`_.

The development is coordinated on `GitHub <http://github.com/cosmo-ethz/seek>`_ and contributions are welcome. The documentation of **MuSEEK** is not yet available at `readthedocs.org <http://museek.readthedocs.io/>`_ .
