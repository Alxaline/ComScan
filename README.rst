ComScan: ComBat the Scanner effect
==================================

This repository implement ComBat and AutoComBat describe in `Carré, et
al. (2022) <https://www.nature.com/articles/s41598-022-16609-1>`__.

If you face any problem, please feel free to open an issue.

Introduction
------------

Current harmonization/normalization methods such as ComBat use a Bayes
parametric empirical framework to robustly adjust the data to site /
scanner effects. This method requires a representative statistical
sample and is therefore not suitable for radiomics machine learning
models for clinical translation, where the emphasis is on evaluating
individual scans from previously unseen scanners. In addition, it may
not always be obvious to define a batch effect that would be linked to
the site or scanners, as a change in a machine parameter may be more
appropriate for another scanner type or site. AutoComBat has thus been
implemented, and it allows to associate a sample to a given site /
scanner by a clustering method. Thus, the site/scanner can be defined by
dicom tags defining the machine (i.e. magnetic field, TI, TR …) or
metrics of image quality.

This repository has been coded to be compatible with
`scikit-learn <https://scikit-learn.org/stable/>`__ and thus facilitate
machine learning projects.

ImageComBat is under development and allows to normalize the image
directly (using Combat or AutoCombat) based on
`neuroHarmonize <https://github.com/rpomponio/neuroHarmonize>`__.

Installation
------------

1. Create a `conda <https://docs.conda.io/en/latest/>`__ environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   ENVNAME="ComScan"
   conda create -n $ENVNAME python==3.7.7 -y
   conda activate $ENVNAME

2. Install repository
~~~~~~~~~~~~~~~~~~~~~

Method 1: Github Master Branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   pip install git+https://github.com/Alxaline/ComScan.git

Method 2: Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   git clone https://github.com/Alxaline/ComScan.git
   cd ComScan
   pip install -e .

Documentation
-------------

https://comscan.readthedocs.io/en/latest/

How to cite ?
-------------

If you find this repository useful for your research, please cite our
work:

Carré, A., Battistella, E., Niyoteka, S. et al. `AutoComBat: a generic
method for harmonizing MRI-based radiomic
features. <https://www.nature.com/articles/s41598-022-16609-1>`__ Sci
Rep 12, 12762 (2022). https://doi.org/10.1038/s41598-022-16609-1

BibTeX:

::

   @article{carreAutoComBatGenericMethod2022,
           title = {AutoComBat: a generic method for harmonizing MRI-based radiomic features},
           volume = {12},
           issn = {2045-2322},
           url = {https://www.nature.com/articles/s41598-022-16609-1},
           doi = {10.1038/s41598-022-16609-1},
           language = {en},
           number = {1},
           urldate = {2022-07-27},
           journal = {Scientific Reports},
           author = {Carré, Alexandre and Battistella, Enzo and Niyoteka, Stephane and Sun, Roger and Deutsch, Eric and Robert, Charlotte},
           year = {2022},
           keywords = {Cancer imaging, Computational science, Tumour biomarkers},
           pages = {12762}}

Disclaimer
~~~~~~~~~~

Based on:
`ComBatHarmonization <https://github.com/Jfortin1/ComBatHarmonization>`__