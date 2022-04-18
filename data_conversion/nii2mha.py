# batch converts the entire dataset from the .nii.gz format to the .mha format
#(the .mha format is required by grand-challenge.org as input and ouput data of algorithms)

# inputs:
# input 1: path_to_nii_data : path to nifti data e.g.  .../nifti/FDG-PET-CT-Lesions/
# input 2: path_to_mha_data : output path for mha data ... /mha/FDG-PET-CT-Lesions/ (will be created if non existing)

import SimpleITK as sitk
import pathlib as plb
import tqdm
import os
import sys


def find_studies(path_to_data):
    # returns a list of unique study paths within the dataset
    dicom_root = plb.Path(path_to_data)
    patient_dirs = list(dicom_root.glob('*'))

    study_dirs = []
    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        study_dirs.extend(sub_dirs)
    return study_dirs


def nii_to_mha(nii_path, mha_out_path):
    # conversion for a single file
    # nii_path:     path to nii file which should be converted
    # mha_out_path: path to mha file which should be created
    img = sitk.ReadImage(nii_path)
    sitk.WriteImage(img, mha_out_path, True)


def convert_nii_to_mha(study_dirs,path_to_mha_data):
    # batch conversion of all studies
    # study_dirs:       list of all study directories
    # path_to_mha_data: path to mha data which should be created
        
    for study_dir in tqdm.tqdm(study_dirs):

        patient = study_dir.parent.name
        study   = study_dir.name

        suv_nii    = str(study_dir/'SUV.nii.gz')
        ctres_nii  = str(study_dir/'CTres.nii.gz')
        ct_nii     = str(study_dir/'CT.nii.gz')
        pet_nii    = str(study_dir/'PET.nii.gz')
        seg_nii    = str(study_dir/'SEG.nii.gz')

        suv_mha_dir    = os.path.join(path_to_mha_data, patient, study)
        ctres_mha_dir  = os.path.join(path_to_mha_data, patient, study)
        ct_mha_dir     = os.path.join(path_to_mha_data, patient, study)
        pet_mha_dir    = os.path.join(path_to_mha_data, patient, study)
        seg_mha_dir    = os.path.join(path_to_mha_data, patient, study)

        os.makedirs(suv_mha_dir  , exist_ok=True)
        os.makedirs(ctres_mha_dir, exist_ok=True)
        os.makedirs(ct_mha_dir   , exist_ok=True)
        os.makedirs(pet_mha_dir  , exist_ok=True)
        os.makedirs(seg_mha_dir  , exist_ok=True)

        nii_to_mha(suv_nii,   os.path.join(suv_mha_dir,'SUV.mha'))
        nii_to_mha(ctres_nii, os.path.join(ctres_mha_dir,'CTres.mha'))
        nii_to_mha(ct_nii,    os.path.join(ct_mha_dir,'CT.mha'))
        nii_to_mha(pet_nii,   os.path.join(pet_mha_dir,'PET.mha'))
        nii_to_mha(seg_nii,   os.path.join(seg_mha_dir,'SEG.mha') )     


if __name__ == "__main__":

    path_to_nii_data = sys.argv[0]  # path to nifti data e.g. .../nifti/FDG-PET-CT-Lesions/
    path_to_mha_data = sys.argv[1]  # output path for mha data ... /mha/FDG-PET-CT-Lesions/ (will be created if non existing)
    study_dirs = find_studies(path_to_nii_data)

    convert_nii_to_mha(study_dirs, path_to_mha_data)
