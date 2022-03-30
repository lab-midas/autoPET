import SimpleITK
import numpy as np
import os

class Autopet():  # SegmentationAlgorithm is not inherited in this class anymore

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        self.nii_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs'
        self.result_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result'
        self.nii_seg_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result/TCIA_001.nii.gz'
        pass

    def convert_mha_to_nii(mha_input_path, nii_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(nii_input_path, mha_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        pass

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.copyfile(self.nii_seg_path, os.path.join('/output/images/automated-petct-lesion-segmentation/', uuid+".nii.gz"))
    
    def predict(self):
        """
        Your algorithm goes here
        """  
        os.system(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m 3d_fullres')

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample

        ct_mha = os.listdir('/input/images/ct/')[0]
        pet_mha = os.listdir('/input/images/pet/')[0]
        uuid = os.path.split(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join('/input/images/pet/', pet_mha), os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
        self.convert_mha_to_nii(os.path.join('/input/images/ct/', ct_mha), os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))

        self.predict()
        self.write_outputs(uuid)


if __name__ == "__main__":
    Autopet().process()