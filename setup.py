from setuptools import setup, find_packages

setup(
    name="museek",
    version="0.3.0",
    description="A flexible and easy-to-extend data processing pipeline for multi-instrument autocorrelation radio experiments",
    author="",
    author_email="",
    packages=find_packages(),
    python_requires='>=3.10, <4',
    entry_points={
        "console_scripts": [
            "museek = ivory.cli.main:run",
        ]
    },
    scripts=[
        "scripts/museek_process_uhf_band.sh",
        "scripts/museek_run_notebook.sh",
    ],
    install_requires=[
        "numpy~=1.23.3",
        "scipy",
        "Cython",
        "h5py",
        "mock",
        "numba",
        "pyephem",
        "six",
        "healpy",
        "astropy",
        "scikit-learn",
        "katdal",
        "sphinx",
        "matplotlib",
        "papermill",
        "ivory @ git+https://github.com/meerklass/ivory.git",
    ],
    classifiers=[
        "GPLv3",
        "Programming Language :: Python :: 3.10",
    ],
)
