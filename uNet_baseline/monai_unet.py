# based on https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d_lightning.ipynb
from email.header import make_header
import sys
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    RandAffined,
    ToTensord
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, compute_meandice
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import pytorch_lightning
import time

import os
import glob
import numpy as np


import nibabel as nib

import SimpleITK as sitk


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            dimensions=3,
            in_channels=2,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(to_onehot_y=True,softmax=True,include_background=False,batch=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
        self.post_label = AsDiscrete(to_onehot=True, n_classes=2)
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # set up the correct data path
        data_dir = '/input/images/'
        images_pt = sorted(glob.glob(os.path.join(data_dir, "SUV*")))
        images_ct = sorted(glob.glob(os.path.join(data_dir, "CTres*")))
       
        data_dicts = [
            {"image_pt": image_name_pt, "image_ct": image_name_ct}
            for image_name_pt, image_name_ct in zip(images_pt, images_ct)
        ]
        val_files = data_dicts
                
        val_transforms = Compose(
            [
                LoadImaged(keys=["image_pt", "image_ct"]),
                AddChanneld(keys=["image_pt", "image_ct"]),
                
                #Spacingd(
                  #  keys=["image_pt", "image_ct"],
                   # pixdim=(2, 2, 3),
                    #mode=("bilinear", "bilinear"),
                #),
                #Orientationd(keys=["image_pt", "image_ct"], axcodes="LAS"),
                
                ScaleIntensityRanged(
                    keys=["image_ct"], a_min=-100, a_max=250,
                    b_min=0.0, b_max=1.0, clip=False,
                ),
                ScaleIntensityRanged(
                    keys=["image_pt"], a_min=0, a_max=15,
                    b_min=0.0, b_max=1.0, clip=False,
                ),                            
                ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),# concatenate pet and ct channels
                                              
                ToTensord(keys=["image_petct"]),
            ]
        )

        
        self.val_ds = Dataset(
             data=val_files, transform=val_transforms)

    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=4, collate_fn = list_data_collate)
        return val_loader

def segment_PETCT(ckpt_path,data_dir,export_dir):

    #ckpt_path='/home/ragatis1/Projects/tcia-lesions/notebooks/epoch=777-step=64573.ckpt'
    print("starting")

    net = Net.load_from_checkpoint(ckpt_path)
    net.eval()

    device = torch.device("cuda:1")
    net.to(device)
    net.prepare_data()

    with torch.no_grad():
        for i, val_data in enumerate(net.val_dataloader()):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            
            mask_out = sliding_window_inference(val_data["image_petct"].to(device), roi_size, sw_batch_size, net)
            mask_out = torch.argmax(mask_out, dim=1).detach().cpu().numpy().squeeze()
            mask_out = mask_out.astype(np.uint8)               
            print("done inference")

            
            PT = nib.load(os.path.join(data_dir,"SUV.nii.gz")) #needs to be loaded to recover nifti header and export mask
            pet_affine = PT.affine
            PT = PT.get_fdata()
            mask_export = nib.Nifti1Image(mask_out, pet_affine)
            nib.save(mask_export, os.path.join(export_dir,"PRED.nii.gz"))
            print("done")


def convert(mha_in_path,nii_out_path):
    #mha_path = '/home/ragatis1/Projects/tcia-lesions/temp/SUV_0c13e4df10.mha'
    #nii_path = '/home/ragatis1/Projects/tcia-lesions/temp/SUV_0c13e4df10.nii.gz'

    #input_image = sitk.ReadImage(mha_in_path)
    input_image = sitk.ReadImage(mha_in_path)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(nii_out_path)
    writer.Execute(input_image)
    print(nii_out_path,'converted')


def run_inference():
    
    data_dir = '/opt/algorithm/' # attention: is hard-coded also in the Net Class above
    #time.sleep(3600)
    print(os.listdir('/input'))

    ct_path  =  '/input/images/ct/af3b6605-c2b9-4067-8af5-8b85aafb2ae3.mha'
    pet_path =  '/input/images/pet/e260efef-0a29-4c68-972e-9e573c740de5.mha'

    print(ct_path)

    convert(ct_path, os.path.join(data_dir,'CTres.nii.gz'))
    convert(pet_path, os.path.join(data_dir,'SUV.nii.gz'))

    ckpt_path= '/opt/algorithm/epoch=777-step=64573.ckpt'
    export_dir = '/output/'
    
    segment_PETCT(ckpt_path,data_dir,export_dir)

if __name__ == '__main__':
    
    run_inference()

