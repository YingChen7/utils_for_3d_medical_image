import os
import SimpleITK as sitk
import numpy as np
from read_write_image import readimage, write_seg_result

def generate_bin(raw_root, bin_to, shresh):
    raw = readimage(os.path.abspath(raw_root))
    Origin = raw.GetOrigin()
    Spacing = raw.GetSpacing()
    Direction = raw.GetDirection()
    raw_arr = sitk.GetArrayFromImage(raw)
    bin_arr = np.zeros_like(raw_arr)
    bin_arr[raw_arr==shresh]=1
    
    write_seg_result(bin_arr, Origin, Spacing, Direction, bin_to)


def crop_accord_seg(raw, seg, crop_raw_root, crop_seg_root, Origin, Spacing, Direction):

    sum_x = np.sum(np.sum(seg, axis = -1), axis =-1)
    sum_y = np.sum(np.sum(seg, axis = 0), axis = -1)
    sum_z = np.sum(np.sum(seg, axis = 0), axis = 0)
    
    nonzero_x = np.nonzero(sum_x)
    nonzero_y = np.nonzero(sum_y)
    nonzero_z = np.nonzero(sum_z)
    
    lower_x = nonzero_x[0][0]
    upper_x = nonzero_x[0][-1]
    lower_y = nonzero_y[0][0]
    upper_y = nonzero_y[0][-1]
    lower_z = nonzero_z[0][0]
    upper_z = nonzero_z[0][-1]
    
    start_x = lower_x - 15
    end_x = start_x + 128
    
    start_y = lower_y - 20
    end_y = start_y + 128
    
    
    start_z = lower_z - 25
    end_z = start_z + 128
    
    crop_raw = raw[start_x:end_x, start_y:end_y, start_z:end_z]
    write_seg_result(crop_raw, Origin, Spacing, Direction, crop_raw_root)
    crop_seg = seg[start_x:end_x, start_y:end_y, start_z:end_z]
    write_seg_result(crop_seg, Origin, Spacing, Direction, crop_seg_root)

def combine_left_right(left_root, left_seg_root, right_root, raw_root):
    raw_left = readimage(os.path.abspath(left_root))
    Origin = raw_left.GetOrigin()
    Spacing = raw_left.GetSpacing()
    Direction = raw_left.GetDirection()
    raw_left_arr = sitk.GetArrayFromImage(raw_left)
    
    seg_left = readimage(os.path.abspath(left_seg_root))
    seg_left_arr = sitk.GetArrayFromImage(seg_left)
    
    raw_right = readimage(os.path.abspath(right_root))
    raw_right_arr = sitk.GetArrayFromImage(raw_right)
    
    seg_pos = np.nonzero(seg_left_arr)
    
    
    for i in zip(*seg_pos):
        raw_left_arr[i[0], i[1], i[2]] = raw_right_arr[i[0], i[1], i[2]]
    
    write_seg_result(raw_left_arr, Origin, Spacing, Direction, raw_root)

   
def crop_fixsize(raw_path, seg_path, crop_raw_path, crop_seg_path):
    crop_size = [100, 416, 352] 
    raw = readimage(os.path.abspath(raw_path))
    seg = readimage(os.path.abspath(seg_path))
    Origin = raw.GetOrigin()
    Spacing = raw.GetSpacing()
    Direction = raw.GetDirection()
    raw_arr = sitk.GetArrayFromImage(raw)   
    seg_arr = sitk.GetArrayFromImage(seg) 
 
    start_h =30
    end_h = start_h + crop_size[1]
    
    start_w = 80
    end_w = start_w + crop_size[2]
    crop_raw = raw_arr[:, start_h:end_h, start_w:end_w]
    crop_seg = seg_arr[:, start_h:end_h, start_w:end_w]
    
    write_seg_result(crop_raw, Origin, Spacing, Direction, crop_raw_path)
    write_seg_result(crop_seg, Origin, Spacing, Direction, crop_seg_path)


def norm(im):
    im = im.astype(np.float32)
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v)
    return im

def crop_accord_max(raw_path):
    
    
    raw_img = readimage(raw_path)
    Origin = raw_img.GetOrigin()
    Spacing = raw_img.GetSpacing()
    Direction = raw_img.GetDirection()
    raw_arr = sitk.GetArrayFromImage(raw_img)
    
    
    seg_path = raw_path.replace('MRA', 'MRA_Seg')
    if not os.path.isfile(seg_path):
        return 
    seg_img = readimage(seg_path)
    seg_arr = sitk.GetArrayFromImage(seg_img)
    '''
    max_y = np.sum(np.sum(raw_arr, axis = 0), axis = -1)
    max_z = np.sum(np.sum(raw_arr, axis = 0), axis = 0)
    
    max_y[max_y<150]=0
    max_z[max_z<150]=0
    nonzero_y = np.nonzero(max_y)
    nonzero_z = np.nonzero(max_z)
    '''
    lower_y = 6
    upper_y = 479
    lower_z = 92
    upper_z = 432
    
    crop_raw = raw_arr[:, lower_y:upper_y, lower_z:upper_z]
    crop_seg = seg_arr[:, lower_y:upper_y, lower_z:upper_z]
    
    crop_raw_root = raw_path.replace('MRA', 'Cr-MRA')
    write_seg_result(crop_raw, Origin, Spacing, Direction, crop_raw_root)
    crop_seg_root = seg_path.replace('MRA', 'Cr-MRA')
    write_seg_result(crop_seg, Origin, Spacing, Direction, crop_seg_root)  

def crop_volume_find_bbox(raw_path, crop_raw_path,  seg_path = None, crop_seg_path = None, normalize=True, thresh=0.1):
    
    raw = readimage(os.path.abspath(raw_path))
    
    Origin = raw.GetOrigin()
    Spacing = raw.GetSpacing()
    Direction = raw.GetDirection()
    raw_arr = sitk.GetArrayFromImage(raw)
    
    st_x, en_x, st_y, en_y, st_z, en_z = 0, 0, 0, 0, 0, 0

    if normalize:
        raw = norm(raw_arr)

    for x in range(raw.shape[0]):
        if np.any(raw[x, :, :] > thresh):
            st_x = x
            break
    for x in range(raw.shape[0] - 1, -1, -1):
        if np.any(raw[x, :, :] > thresh):
            en_x = x
            break
    for y in range(raw.shape[1]):
        if np.any(raw[:, y, :] > thresh):
            st_y = y
            break
    for y in range(raw.shape[1] - 1, -1, -1):
        if np.any(raw[:, y, :] > thresh):
            en_y = y
            break
    for z in range(raw.shape[2]):
        if np.any(raw[:, :, z] > thresh):
            st_z = z
            break
    for z in range(raw.shape[2] - 1, -1, -1):
        if np.any(raw[:, :, z] > thresh):
            en_z = z
            break
      
    crop_raw  = raw_arr[st_x:en_x+1, st_y:en_y+1, st_z:en_z+1]
    write_seg_result(crop_raw, Origin, Spacing, Direction, crop_raw_path)
    
    if seg_path is not None:
        seg = readimage(os.path.abspath(seg_path))
        seg_arr = sitk.GetArrayFromImage(seg)
        crop_seg  = seg_arr[st_x:en_x+1, st_y:en_y+1, st_z:en_z+1]
        write_seg_result(crop_seg, Origin, Spacing, Direction, crop_seg_path)
        
    nbbox = np.array([st_x, en_x, st_y, en_y, st_z, en_z]).astype(int)
    
    print(nbbox)

    return nbbox
    
def resize_image_itk(ori_image, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    
    originSize = ori_image.GetSize()  
    originSize = np.array(originSize,float)
    newSize = np.array(newSize,float)
    if np.any(originSize != newSize): 
       
        resampler = sitk.ResampleImageFilter()
        originSpacing = ori_image.GetSpacing()
        factor = originSize / newSize
        newSpacing = originSpacing * factor
        newSize = newSize.astype(np.int) 
        resampler.SetReferenceImage(ori_image)  
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(ori_image)  
        itkimgResampled = sitk.Cast(itkimgResampled,sitk.sitkInt16)  
    else:
        itkimgResampled = sitk.Cast(ori_image,sitk.sitkInt16)   
    return itkimgResampled   
 
# newSize = [426, 496, 100]
# originSize = ori_image.GetSize()  # 原来的体素块尺寸
# originSize = np.array(originSize,float)
# newSize = np.array(newSize,float)
# resamplemethod=sitk.sitkNearestNeighbor
# if np.any(originSize != newSize): 
   
#     resampler = sitk.ResampleImageFilter()
#     originSpacing = ori_image.GetSpacing()
#     factor = originSize / newSize
#     newSpacing = originSpacing * factor
#     newSize = newSize.astype(np.int) 
#     resampler.SetReferenceImage(ori_image)  # 需要重新采样的目标图像
#     resampler.SetSize(newSize.tolist())
#     resampler.SetOutputSpacing(newSpacing.tolist())
#     resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
#     resampler.SetInterpolator(resamplemethod)
#     itkimgResampled = resampler.Execute(ori_image)  # 得到重新采样后的图像
# else:
#     itkimgResampled = ori_image
    
# sitk.WriteImage(itkimgResampled, filename_to)

'''
# for raw images, resamplemethod = sitk.sitkBSpline 
def resize_image_itk(ori_image_path, newSize, res_image_path, resamplemethod=sitk.sitkNearestNeighbor):
    ori_image = readimage(os.path.abspath(ori_image_path))
    resampler = sitk.ResampleImageFilter()
    originSize = ori_image.GetSize()  
    originSpacing = ori_image.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) 
    resampler.SetReferenceImage(ori_image)  
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(ori_image)  
    sitk.WriteImage(itkimgResampled, res_image_path)
    
volumes_list = glob.glob(join(r'H:\Cerebrovascular_datasets\IOP-dataset\batch-3','*-Cr-Up-MRA-Seg.nii.gz'))

for volume_path in volumes_list:  
    resize_image_itk(volume_path, [256, 256, 128], volume_path.replace('-Cr-Up-MRA-Seg', '-Cr-Down-MRA-Seg'))    
''' 
   
def resample_volumes(volumes, target_spacing = [0.46875, 0.46875, 0.8]):
    
    # target_spacing = compute_medSpacing(volumes_list)
    
    target_spacing = [0.46875,    0.46875,   0.80000001]
    print(target_spacing)
    
    for volume in volumes:
        print(volume)
        #path_gt = volume.replace('_0000.nii.gz', '.nii.gz')
        img = readimage(volume)
        #gt_img = readimage(path_gt)
        ori_spacing = img.GetSpacing()
        #gt_ori_spacing = gt_img.GetSpacing()
        #assert np.any(ori_spacing==gt_ori_spacing)
        
        ori_origin = img.GetOrigin()
        ori_direction = img.GetDirection()             
        shape = np.array(img.GetSize())
        print(shape)
        #gt_shape = np.array(gt_img.GetSize())
        #assert  np.any(shape==gt_shape)
        
        new_shape = np.round((np.array( ori_spacing)/np.array(target_spacing)).astype(float) * shape).astype(int)
        print(new_shape)
        
        img_reshaped = resize_image_itk(img, new_shape, sitk.sitkBSpline)
        #gt_img_reshaped = resize_image_itk(gt_img, new_shape, sitk.sitkNearestNeighbor)
        
        img_reshaped.SetOrigin(ori_origin)
        img_reshaped.SetDirection(ori_direction)
        
        #gt_img_reshaped.SetOrigin(ori_origin)
        #gt_img_reshaped.SetDirection(ori_direction)
        
        sitk.WriteImage(img_reshaped, volume.replace('Raw.nii', 'Res-MRA.nii'))
        #sitk.WriteImage(gt_img_reshaped, path_gt.replace('.nii', '_Res.nii'))
  
'''   
volume_result_path = 'C:\\Users\\chenying\\Desktop\\transfer\\VNet\\588_VNet.nii.gz'
volume_gt_path = 'D:\\Cerebrovascular datasets\\MRA\\IXI588-IOP-1158-MRA_GT.nii.gz'
ASD = ASD(volume_result_path,volume_gt_path)
print(ASD)

img = cv2.imread('D:\\result\\Figure\\FAB\\c_min3.png',2)
img1 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
cv2.imwrite('D:\\result\\Figure\\FAB\\jetc_min3.png', img1)
plt.imshow(img1)

img = readimage('H:\\Cerebrovascular_datasets\\train_gt\\293\\90.nii.gz')
arr = sitk.GetArrayFromImage(img)
volume_size, ref_origin, ref_spacing, ref_direction = ref_info_for_write('H:\\Cerebrovascular_datasets\\train_gt\\293\\90.nii.gz')
dis_arr = ndimage.distance_transform_bf(arr,metric = 'taxicab',return_indices=False)
print(np.max(dis_arr))
distance_norm = dis_arr / (3* np.max(dis_arr)+0.05)
distance_rev = (1.0- distance_norm)* arr
dis_img = sitk.GetImageFromArray(distance_rev)
dis_img.SetOrigin(ref_origin)
dis_img.SetSpacing(ref_spacing)    
dis_img.SetDirection(ref_direction)   
sitk.WriteImage(dis_img, 'H:\\Cerebrovascular_datasets\\train_gt\\293\\90_dist.nii.gz')
'''
