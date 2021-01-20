# -*- coding: utf-8 -*-
"""
Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
Created on: Jan 13, 2021
"""

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='ComScan',
    version='0.0.1',
    description="ComBat, AutoCombat and ImageCombat harmonization scikit-learn compatible",
    long_description=readme,
    author='Alexandre CARRE',
    author_email='alexandre.carre@gustaveroussy.fr',
    url='https://github.com/Alxaline/ComScan',
    license=license,
    packages=find_packages(exclude=['docs']),
    python_requires='>=3.6',
    keywords="Perform radiomics harmonization based on ComBat scikit-learn compatible.",
)

setup(install_requires=['simpleitk',
                        'numpy',
                        'pandas',
                        'kneed',
                        'scipy',
                        'scikit-learn',
                        'neurocombat',
                        'tqdm',
                        'nipy @ git+https://github.com/nipy/nipy.git#egg=nipy',
                        'umap'
                        ], **args)
