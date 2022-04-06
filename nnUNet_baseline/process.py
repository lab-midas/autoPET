import SimpleITK
import numpy as np
import os
import subprocess
import shutil

class Autopet_baseline():  # SegmentationAlgorithm is not inherited in this class anymore

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        self.input_path = '/input/'
        self.nii_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs'
        self.result_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result'
        self.nii_seg_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result/TCIA_001.nii.gz'

        #self.input_path = '/home/rakuest1/Documents/autoPET/nnUNet_baseline/test/input/'
        #self.nii_path = '/home/rakuest1/nnUNet_baseline/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs'
        #self.result_path = '/home/rakuest1/nnUNet_baseline/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result'
        #self.nii_seg_path = '/home/rakuest1/nnUNet_baseline/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result/TCIA_001.nii.gz'
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  #nnUNet specific
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
        shutil.copyfile(self.nii_seg_path, os.path.join('/output/images/automated-petct-lesion-segmentation/', uuid+".nii.gz"))

    def predict(self):
        """
        Your algorithm goes here
        """
        #os.environ['nnUNet_raw_data_base'] = '/home/rakuest1/nnUNet_baseline/nnUNet_raw_data_base'
        #os.environ['RESULTS_FOLDER'] = '/home/rakuest1/nnUNet_baseline/checkpoints'  # /home/rakuest1/miniconda3/envs/autoPET/bin/
        cproc = subprocess.run(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m 3d_fullres', shell=True, check=True)
        print(cproc)  # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being printed.
        print('Prediction done')
        #os.system(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m 3d_fullres')

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        #ct_mha = os.listdir('/input/images/ct/')[0]
        #pet_mha = os.listdir('/input/images/pet/')[0]
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha), os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha), os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))

        self.predict()
        self.write_outputs(uuid)


if __name__ == "__main__":
    Autopet_baseline().process()