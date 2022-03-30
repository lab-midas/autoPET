import SimpleITK as sitk
import pathlib as plb
import tqdm
import os


def find_studies(path_to_data):
    # find all studies
    dicom_root = plb.Path(path_to_data)
    patient_dirs = list(dicom_root.glob('*'))

    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        #print(sub_dirs)
        study_dirs.extend(sub_dirs)
        
        #dicom_dirs = dicom_dirs.append(dir.glob('*'))
    return study_dirs


def nii_to_mha(nii_path, mha_out_path):
    # conversion for a single file
    # nii_path:     path to nii file which should be converted
    # mha_out_path: path to mha file which should be created
    img = sitk.ReadImage(nii_path)
    sitk.WriteImage(img, mha_out_path, True)


def convert_nii_to_mha(study_dirs, path_to_mha_data):
    # batch conversion of all patients
    # study_dirs:       list of all study directories
    # path_to_mha_data: path to mha data which should be created

    for study_dir in tqdm.tqdm(study_dirs[0:1]):

        patient = study_dir.parent.name
        study   = study_dir.name

        suv_nii    = str(study_dir/'SUV.nii.gz')
        ctres_nii  = str(study_dir/'CTres.nii.gz')
        ct_nii     = str(study_dir/'CT.nii.gz')
        pet_nii    = str(study_dir/'PET.nii.gz')
        seg_nii    = str(study_dir/'SEG.nii.gz')

        suv_mha    = os.path.join(path_to_mha_data, patient, study)
        ctres_mha  = os.path.join(path_to_mha_data, patient, study)
        ct_mha     = os.path.join(path_to_mha_data, patient, study)
        pet_mha    = os.path.join(path_to_mha_data, patient, study)
        seg_mha    = os.path.join(path_to_mha_data, patient, study)

        os.makedirs(suv_mha  , exist_ok=True)
        os.makedirs(ctres_mha, exist_ok=True)
        os.makedirs(ct_mha   , exist_ok=True)
        os.makedirs(pet_mha  , exist_ok=True)
        os.makedirs(seg_mha  , exist_ok=True)

        nii_to_mha(suv_nii,   os.path.join(suv_mha,'SUV.mha'))
        nii_to_mha(ctres_nii, os.path.join(ctres_mha,'CTres.mha'))
        nii_to_mha(ct_nii,    os.path.join(ct_mha,'CT.mha'))
        nii_to_mha(pet_nii,   os.path.join(pet_mha,'PET.mha'))
        nii_to_mha(seg_nii,   os.path.join(seg_mha,'SEG.mha') )     


if __name__ == "__main__":
    path_to_nii_data = '/mnt/DataFast/ragatis1/TCIA/tcia_nifti/FDG-PET-CT-Lesions/' 
    path_to_mha_data = '/mnt/DataFast/ragatis1/TCIA/tcia_mha/FDG-PET-CT-Lesions/'
    study_dirs = find_studies(path_to_nii_data)

    convert_nii_to_mha(study_dirs, path_to_mha_data)
