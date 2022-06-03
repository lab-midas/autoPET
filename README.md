# autoPET challenge
Repository for codes associated with autoPET machine learning challenge: <br/> 
[autopet.grand-challenge.org](https://autopet.grand-challenge.org/) 

If you use the data associated to this challenge, please cite: <a href="https://doi.org/10.7937/gkr0-xv29"><img src="https://img.shields.io/badge/DOI-10.7937%2Fgkr0--xv29-blue"></a>

```
Gatidis S, Kuestner T. A whole-body FDG-PET/CT dataset with manually annotated tumor lesions (FDG-PET-CT-Lesions) 
[Dataset]. The Cancer Imaging Archive, 2022. DOI: 10.7937/gkr0-xv29
```

## Data conversion
Scripts for converting the database between DICOM, NiFTI, HDF5 and MHA formats.

## nnUNet baseline
Baseline model for lesion segmentation: In this baseline model, the nnUNet framework (https://github.com/MIC-DKFZ/nnUNet) was used for training using the 3D fullres configuration with 16 GB of VRAM. PET (SUV) and resampled CT volumes were used as model input. The number of epochs was set to 1,000; the initial learning rate to 1e-4. Training was performed with 5-fold cross validation.  

## MONAI uNet baseline
Baseline model for lesion segmentation: In this proof-of-concept model a standard 3D uNet model as provided within the MONAI framework (https://monai.io) was adapted to dual-channel input (PET (SUV) and resampled CT volumes). Input patches were of size (128, 128, 32), batch size was set to 12, learning rate 1e-4 using Adam, maximum number of epochs set to 800. 

## References
Challenge: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6362493.svg)](https://doi.org/10.5281/zenodo.6362493) <br/>
Database: <a href="https://doi.org/10.7937/gkr0-xv29"><img src="https://img.shields.io/badge/DOI-10.7937%2Fgkr0--xv29-blue"></a>
