import SimpleITK as sitk
import os

def readimage(filename):
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    return reader.Execute()

def ref_info_for_write(ref_img):
    ref_volume = readimage(os.path.abspath(ref_img))
    volume_size = [ref_volume.GetDepth(),ref_volume.GetHeight(),ref_volume.GetWidth()]
    ref_origin = ref_volume.GetOrigin()
    ref_spacing = ref_volume.GetSpacing()
    ref_direction = ref_volume.GetDirection()
    return volume_size, ref_origin, ref_spacing, ref_direction

def write_seg_result(seg_result, ref_origin, ref_spacing, ref_direction, filename):
    result_img = sitk.GetImageFromArray(seg_result)
    result_img.SetOrigin(ref_origin)
    result_img.SetSpacing(ref_spacing)
    result_img.SetDirection(ref_direction)
    sitk.WriteImage(result_img, filename)
    print('Successfully write!')
