# ComScan

Implementation of ComBat, AutoCombat and ImageCombat.

Current harmonization/normalization methods such as Combat use a Bayes parametric empirical framework to robustly
 adjust the data to site / scanner effects. 
This method requires a representative statistical sample and is therefore not suitable for radiomics machine learning 
models for clinical translation, where the emphasis is on evaluating individual scans from previously unseen scanners. 
 In addition, it may not always be obvious to define a batch effect that would be linked to the site or scanners, as 
 a change in a machine parameter may be more appropriate for another scanner type or site. AutoCombat has thus been 
 implemented and it allows to associate a sample to a given site / scanner by a clustering method. 
 Thus the site/scanner can be defined by dicom tags defining the machine (i.e. magnetic field, TI, TR ...) 
 or metrics of image quality. 
ImageCombat has also been implemented and allows to normalize the image directly (using Combat or AutoCombat).

This repository has been coded to be compatible with [scikit-learn](https://scikit-learn.org/stable/) and thus facilitate machine learning projects.

## Installation
### 1. Create a [conda](https://docs.conda.io/en/latest/) environment (recommended)
```
ENVNAME="ComScan"
conda create -n $ENVNAME python==3.7.7 -y
conda activate $ENVNAME
```
### 2. Install repository
#### Method 1: Github Master Branch
```
pip install git+https://github.com/Alxaline/ComScan.git
```
#### Method 2: Development Installation
```
git clone https://github.com/Alxaline/ComScan.git
cd ComScan
pip install -e .
```

### Disclaimer
Based on:
[ComBatHarmonization](https://github.com/Jfortin1/ComBatHarmonization)


