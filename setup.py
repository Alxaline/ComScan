# -*- coding: utf-8 -*-
"""
Author: Alexandre CARRE 
Created on: Jan 13, 2021
"""

from setuptools import setup, find_packages


def import_or_install(package: str) -> None:
    """
    import or install package
    :param package: package name
    """
    try:
        __import__(package)
    except Exception:
        import sys
        import subprocess

        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        __import__(package)


import_or_install("numpy")  # install numpy needed by nipy

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
    url='https://github.com/Alxaline/ComScan',
    license=license,
    packages=find_packages(exclude=['docs']),
    python_requires='>=3.6',
    keywords="Perform radiomics ComScan based on ComBat scikit-learn compatible.",
)

setup(setup_requires=['numpy'],
      install_requires=['numpy',
                        'yellowbrick @ git+https://github.com/Alxaline/yellowbrick.git#egg=yellowbrick',
                        # no use original because need no restriction on numpy
                        'simpleitk',
                        'pandas',
                        'kneed',
                        'scipy',
                        'scikit-learn',
                        'neurocombat',
                        'tqdm',
                        'nipy @ git+https://github.com/nipy/nipy.git#egg=nipy',
                        'umap-learn',
                        'k-means-constrained',
                        ], **args)
