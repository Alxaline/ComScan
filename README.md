# MMI-PROB Dataset

This is a pipeline to do preprocessing on brain MR images of **MMI-PROB** dataset. Also models to do inference on MRI to predict the three labels (Edema, NCR & NET, ET) is available.

In our dataset 3 MRI sequences (T1, T1ce, flair) and CT scan are included.

 ## 1) Prerequisite

### 1. Install ANTs   

Various pre-processing steps of the dataset required ANTs (Advanced Normalization Tools). ANTs is popularly considered a state-of-the-art medical image registration and segmentation toolkit.
As ANTs is coded in C++, a python wrapper is then used ! 

Compile **ANTs** from source code in [Linux and macOS](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS), or in [Windows 10](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Windows-10).

### 2. Create virtual environment (of your choice: conda or venv)

Example with conda:
```
conda create -n yourenvname python==3.7.7
```

then:
```
source activate yourenvname
```

### 3. Install repository

```
git clone https://github.com/U1030/DS2nii.git
cd DS2nii
pip install -r requirements.txt
pip install -e . # developpement mode OR
pip install . # non dev mode
```

### 4. Install HD-BET

To perform skull-stripping in the preprocessing, pipeline is performed with [HD-BET](https://github.com/MIC-DKFZ/HD-BET).
```
git clone https://github.com/MIC-DKFZ/HD-BET
cd HD-BET
pip install -r requirements.txt
pip install -e . # developpement mode OR
pip install . # non dev mode
```
_if you used HD-BET which is in the preprocessing script please also cite:_

```
Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A, Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P. Automated brain extraction of multi-sequence MRI using artificial neural networks. Hum Brain Mapp. 2019; 1–13. https://doi.org/10.1002/hbm.24750
```

 ## 2) Preprocessing
 
Preprocessing is based on [HD-BET](https://github.com/MIC-DKFZ/HD-BET) and [ANTs](https://github.com/ANTsX/ANTs)

The folder structure was as follows on the data: 

SainteAnne folder:

    INPUT_DIR_SainteAnne/
    ├── SainteAnne1
    │   ├── 1-5T
    │   │   ├── SainteAnne1_1-5T_flair.nii.gz
    │   │   └── SainteAnne1_1-5T_t1ce.nii.gz
    │   └── 3T
    │       ├── SainteAnne1_3T_flair.nii.gz
    │       └── SainteAnne1_3T_t1ce.nii.gz
    ├── SainteAnne2
    │   ├── 1-5T
    │   │   ├── SainteAnne2_1-5T_flair.nii.gz
    │   │   └── SainteAnne2_1-5T_t1ce.nii.gz
    │   └── 3T
    │       ├── SainteAnne2_3T_flair.nii.gz
    │       └── SainteAnne2_3T_t1ce.nii.gz
    ├── ...     
 
Leeds folder:

    INPUT_DIR_Leeds/
    ├── Leeds1
    │   ├── TPearly
    │   │   ├── Leeds1_TPearly_flair.nii.gz
    │   │   ├── Leeds1_TPearly_t1ce.nii.gz
    │   │   ├── Leeds1_TPearly_t1.nii.gz
    │   │   └── Leeds1_TPearly_t2.nii.gz
    │   └── TPlate
    │       ├── Leeds1_TPlate_flair.nii.gz
    │       ├── Leeds1_TPlate_t1ce.nii.gz
    │       ├── Leeds1_TPlate_t1.nii.gz
    │       └── Leeds1_TPlate_t2.nii.gz
    ├── Leeds2
    │   ├── TPearly
    │   │   ├── Leeds2_TPearly_flair.nii.gz
    │   │   ├── Leeds2_TPearly_t1ce.nii.gz
    │   │   ├── Leeds2_TPearly_t1.nii.gz
    │   │   └── Leeds2_TPearly_t2.nii.gz
    │   └── TPlate
    │       ├── Leeds2_TPlate_flair.nii.gz
    │       ├── Leeds2_TPlate_t1ce.nii.gz
    │       ├── Leeds2_TPlate_t1.nii.gz
    │       └── Leeds2_TPlate_t2.nii.gz
    ├── ... 

```  
python -m exec.preprocess -i /media/acarre/Data/data_stock/ComBat/SainteAnne/nifti -o /media/acarre/Data/data_stock/ComBat/SainteAnne/nifti_preprocess --dataset SainteAnne -vv
```  
```  
python -m exec.create_labelsmap -i /media/acarre/Data/data_stock/ComBat/Leeds/nifti_preprocess/ -s /media/acarre/Data/data_stock/ComBat/Leeds/wt_segmentations/ -o /media/acarre/Data/data_stock/ComBat/Leeds/labelsmap --dataset Leeds -vv
```  
Script will check that you will have the correct folder structure.
To do inference of models, CT scan is optional.

The preprocessing will be as follows for a patient:
 1) N4 bias field correction of MRI
 2) Resample to 1mm x 1mm x 1mm 
 3) Registration of T1 to SRI24 Atlas (lps), then registration of others modalities to T1 registered
 4) Generate brain mask of each MRI 
 5) Create a final brain mask from the generated brain masks
 6) Skull-stripped images from the final brain mask

```
python -m preprocessing.pre-rt.processing_pre-rt -i /data/MMI-PROB/pre_rt/pre_rt_NIFTI_rename/ -o /data/MMI-PROB/pre_rt/pre_rt_NIFTI_preprocess -vv --log-file /data/MMI-PROB/pre_rt/pre_rt_NIFTI_preprocess/log_file.log
```

Dataset need to include 3 MRI sequences (T1, T1ce, flair). CT is also available in this dataset.



