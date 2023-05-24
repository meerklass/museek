from setuptools import setup, find_packages

setup(
    name='museek',
    version='0.0.1',
    description='A flexible and easy-to-extend data processing pipeline for multi-instrument autocorrelation radio experiments',
    author='',
    author_email='',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.3',
        'scipy',
        'Cython',
        'h5py',
        'mock',
        'numba',
        'pyephem',
        'six',
        'healpy',
        'astropy',
        'scikit-learn',
        'katdal',
    ],
    classifiers=[
        'GPLv3',
        'Programming Language :: Python :: 3.10',
    ],
)
