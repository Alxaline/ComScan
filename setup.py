# -*- coding: utf-8 -*-
"""
Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
Created on: Jan 13, 2021
"""

import pip
from setuptools import setup, find_packages

pip.main(['numpy'])  # call pip to install numpy needed by nipy

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='ComScan',
    version='0.0.1',
    description="ComBat, AutoCombat and ImageCombat ComScan scikit-learn compatible",
    long_description=readme,
    author='Alexandre CARRE',
    author_email='alexandre.carre@gustaveroussy.fr',
    url='https://github.com/Alxaline/ComScan',
    license=license,
    packages=find_packages(exclude=['docs']),
    python_requires='>=3.6',
    keywords="Perform radiomics ComScan based on ComBat scikit-learn compatible.",
)

setup(setup_requires=['numpy'],
      install_requires=['numpy',
                        'simpleitk',
                        'pandas',
                        'kneed',
                        'scipy',
                        'scikit-learn',
                        'neurocombat',
                        'tqdm',
                        'nipy @ git+https://github.com/nipy/nipy.git#egg=nipy',
                        'umap-learn',
                        'k-means-constrained'
                        ], **args)
