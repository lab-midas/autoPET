import SimpleITK as sitk


def mha_to_nii(mha_path, nii_out_path):
    # conversion for a single file
    # mha_path:     path to mha file which should be converted
    # nii_out_path: path to output nifti file
    '''
    image = SimpleITK.ReadImage(mha_input_path)
    writer = SimpleITK.ImageFileWriter()
    writer.SetFileName(nii_out_path)
    writer.Execute(image)
    '''
    img = sitk.ReadImage(mha_path)
    sitk.WriteImage(img, nii_out_path, True)