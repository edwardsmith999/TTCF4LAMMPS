#!/usr/bin/env python

from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup( name = "TTCF4LAMMPS",
       version = "1.0.0",
       author = ["Luca Maffioli","Edward Smith"],
       author_email = "lmaffioli@swin.edu.au",
       url = "https://github.com/edwardsmith999/TTCF4LAMMPS",
       classifiers=['Development Status :: 3 - Alpha',
                     'Programming Language :: Python :: 3.11'],
       packages=find_packages(exclude=['contrib', 'docs', 'tests']),
       keywords='Molecular Dynamics TTCF Transient Time Correlation Function',
       license = "GPL",
       install_requires=['numpy', 'scipy', 'matplotlib', 'mpi4py', 'lammps' ],
       description = "Code to run TTCF function on LAMMPS",
       long_description = long_description,
       long_description_content_type='text/markdown',
       entry_points={
            'console_scripts': [
                'TTCF=run_TTCF:main',
            ],
       },
)
