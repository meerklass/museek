from setuptools import setup, find_packages

setup(
    name="museek",
    version="0.2.1",
    description="A flexible and easy-to-extend data processing pipeline for multi-instrument autocorrelation radio experiments",
    author="",
    author_email="",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "museek = ivory.cli.main:run",
        ]
    },
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
        "ivory @ git+https://github.com/meerklass/ivory.git",
    ],
    classifiers=[
        "GPLv3",
        "Programming Language :: Python :: 3.10",
    ],
)
