import os
import SimpleITK as sitk
from read_write_image import readimage, write_seg_result
import numpy as np

def connected_domain(segmentation_root,segmentation_to):
    segmentation = readimage(os.path.abspath(segmentation_root))
    Origin = segmentation.GetOrigin()
    Spacing = segmentation.GetSpacing()
    Direction = segmentation.GetDirection()
    #seg = sitk.GetArrayFromImage(segmentation)
    segmentation = sitk.Cast(segmentation, sitk.sitkFloat32) 
    segmentation = sitk.BinaryThreshold(segmentation, 1, 1, 1, 0)
    
    
    Input = segmentation
    CCF = sitk.ConnectedComponentImageFilter()
    CCF.SetFullyConnected(True)
    
    output = CCF.Execute(Input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output)
    num_label = CCF.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label +1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1] 
    final_label_list = []
    
    for idx, i in enumerate(num_list_sorted):
        if area_list[i-1] > 50000:
            final_label_list.append(i)
        else:
            break
    output = sitk.GetArrayFromImage(output)
    
    for one_label in num_list:
        if  one_label in final_label_list:
            continue
        else:
            mask = (output== one_label)
            output[mask] = 0
        '''
        x, y, z, w, h, d = stats.GetBoundingBox(one_label)
        one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
        output[z: z + d, y: y + h, x: x + w] *= one_mask
        '''
    output = (output > 0).astype(np.uint8)
    write_seg_result(output, Origin, Spacing, Direction, segmentation_to)
    return output
