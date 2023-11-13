import os
from read_write_image import readimage
import numpy as np
import SimpleITK as sitk
import scipy
import seg_metrics.seg_metrics as sg
'''
set colors for TP, FN, FP
'''
def volume_postprocessing(volume_result_path, volume_gt_path, path_to_save):
    volume_result = readimage(os.path.abspath(volume_result_path))
    volume_gt = readimage(os.path.abspath(volume_gt_path))
    Origin = volume_gt.GetOrigin()
    Spacing = volume_gt.GetSpacing()
    Direction = volume_gt.GetDirection()
    volume_shape = [volume_gt.GetDepth(),volume_gt.GetHeight(),volume_gt.GetWidth()]
    volume_color = np.zeros(shape = (volume_shape[0], volume_shape[1], volume_shape[2]))
    volume_result = sitk.GetArrayFromImage(volume_result)
    volume_gt = sitk.GetArrayFromImage(volume_gt)
    volume_color[(volume_result == 1) * (volume_gt== 1)] = 1
    volume_color[(volume_result == 0) * (volume_gt== 1)] = 2
    volume_color[(volume_result == 1) * (volume_gt== 0)] = 3
    volume_color = sitk.GetImageFromArray(volume_color)
    volume_color.SetOrigin(Origin)
    volume_color.SetSpacing(Spacing)
    volume_color.SetDirection(Direction)
    sitk.WriteImage(volume_color, path_to_save)   
    print('Successfully save the visualization:', volume_result_path)
    
'''
metrics
'''
'''
def distance_transform(img):
    dt = sitk.SignedMaurerDistanceMapImageFilter()
    dt.SetBackgroundValue(0)
    dt.SetInsideIsPositive(True)
    dt.SetUseImageSpacing(True)
    return dt.Execute(sitk.Cast(img, sitk.sitkUInt16))

def distance_A2B(binaryA, binaryB):
    dtA = distance_transform(binaryA)
    dtB = distance_transform(binaryB)
    spacing = dtA.GetSpacing()
    
    arr_A = sitk.GetArrayFromImage(dtA)
    arr_B = sitk.GetArrayFromImage(dtB)
    
    boundary_A = np.where(abs(arr_A)<0.5*spacing[0])
    
    union = arr_A + arr_B
    return np.abs(union[boundary_A])

def compute_SHD95(file_gt,file_seg):
    img_gt = readimage(os.path.abspath(file_gt))
    img_seg = readimage(os.path.abspath(file_seg))
    dA2B = distance_A2B(img_gt,img_seg)
    sorted_A2B = np.sort(dA2B)
    dB2A = distance_A2B(img_seg,img_gt)
    sorted_B2A = np.sort(dB2A)
    return 0.5 * (sorted_A2B[int(len(sorted_A2B)*0.95)]+sorted_B2A[int(len(sorted_B2A)*0.95)])

def compute_SASD(file_gt, file_seg):
    img_gt = readimage(os.path.abspath(file_gt))
    img_seg = readimage(os.path.abspath(file_seg))
    return (np.mean(distance_A2B(img_gt,img_seg))+ np.mean(distance_A2B(img_seg,img_gt))) * 0.5

def compute_SHD(inimg,outimg):
    dA2B = distance_A2B(inimg,outimg)
    sorted_A2B = np.sort(dA2B)
    dB2A = distance_A2B(outimg,inimg)
    sorted_B2A = np.sort(dB2A)
    return 0.5 * (sorted_A2B[int(len(sorted_A2B))-1]+sorted_B2A[int(len(sorted_B2A))-1])
'''
def compute_IOU_DSC(file_gt, file_pred):
    pred = readimage(os.path.abspath(file_pred))
    gt = readimage(os.path.abspath(file_gt))
    pred_arr = sitk.GetArrayFromImage(pred)
    gt_arr = sitk.GetArrayFromImage(gt)
    inter = np.sum(pred_arr * gt_arr)
    union = np.sum(gt_arr) + np.sum(pred_arr) - inter
    iou = float(inter)/float(union)
    
    l = np.sum(gt_arr)
    r = np.sum(pred_arr)
    dice = (2 * float(inter)) / float(l + r)
    
    return iou, dice

def compute_distance_voxel(file_gt, file_seg):
    seg = readimage(os.path.abspath(file_seg))
    gt = readimage(os.path.abspath(file_gt))
    gt_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(sitk.Cast(gt, sitk.sitkUInt16), squaredDistance=False, useImageSpacing=True))
    gt_surface = sitk.LabelContour(gt)
    
    seg_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(sitk.Cast(seg, sitk.sitkUInt16), squaredDistance=False, useImageSpacing=True))
    seg_surface = sitk.LabelContour(sitk.Cast(seg, sitk.sitkUInt16))
           
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    
    seg2gt_distance_map_arr = sitk.GetArrayFromImage(gt_distance_map)*sitk.GetArrayFromImage(sitk.Cast(seg_surface, sitk.sitkFloat32))
    gt2seg_distance_map_arr = sitk.GetArrayFromImage(seg_distance_map)*sitk.GetArrayFromImage(sitk.Cast(gt_surface, sitk.sitkFloat32))
        
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(seg_surface)
    num_seg_surface_pixels = int(statistics_image_filter.GetSum())
    
    statistics_image_filter.Execute(gt_surface)
    num_gt_surface_pixels = int(statistics_image_filter.GetSum()) 
    
    # Get all non-zero distances and then add zero distances if required.
    seg2gt_distances = list(seg2gt_distance_map_arr[seg2gt_distance_map_arr!=0]) 
    #seg2gt_distances = seg2gt_distances + list(np.zeros(num_seg_surface_pixels - len(seg2gt_distances)))
    gt2seg_distances = list(gt2seg_distance_map_arr[gt2seg_distance_map_arr!=0]) 
    #gt2seg_distances = gt2seg_distances + list(np.zeros(num_gt_surface_pixels - len(gt2seg_distances)))
        
    all_surface_distances = seg2gt_distances + gt2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In 
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two 
    # segmentations, though in our case it is. More on this below.
    asd = np.mean(all_surface_distances)
    sorted_seg2gt = np.sort(seg2gt_distances)
    sorted_gt2seg = np.sort(gt2seg_distances)
    hd95 = max(np.percentile(sorted_seg2gt, 95), np.percentile(sorted_gt2seg, 95))
    return asd, hd95

def HD95(file_gt, file_pred):
    pred_img = readimage(os.path.abspath(file_pred))
    gt_img = readimage(os.path.abspath(file_gt))
    pred_img  = sitk.BinaryThreshold(pred_img, 1, 1, 1, 0)
    gt_img = sitk.BinaryThreshold(gt_img, 1, 1, 1, 0)

    predS = sitk.LabelContour(pred_img)
    gtS = sitk.LabelContour(gt_img)
    
    predS_arr   = sitk.GetArrayFromImage(predS)
    gtS_arr = sitk.GetArrayFromImage(gtS) 
    
    pred_Coordinates   = [pred_img.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose(np.flipud(np.nonzero(predS_arr)))]
    gt_Coordinates = [gt_img.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose(np.flipud(np.nonzero(gtS_arr)))]
                
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]
        
     # Compute distances  
    dPredToGt = getDistancesFromAtoB(pred_Coordinates, gt_Coordinates)
    dGtToPred = getDistancesFromAtoB(gt_Coordinates, pred_Coordinates)
    hd95_value = np.percentile(np.sort(np.append(dPredToGt,dGtToPred)), 95)
    
    return hd95_value

def ASD(file_gt, file_pred):
    pred_img = readimage(os.path.abspath(file_pred))
    gt_img = readimage(os.path.abspath(file_gt))
    pred_img  = sitk.BinaryThreshold(pred_img, 1, 1, 1, 0)
    gt_img = sitk.BinaryThreshold(gt_img, 1, 1, 1, 0)

    predS = sitk.LabelContour(pred_img)
    gtS = sitk.LabelContour(gt_img)
    
    predS_arr   = sitk.GetArrayFromImage(predS)
    gtS_arr = sitk.GetArrayFromImage(gtS) 
    
    pred_Coordinates   = [pred_img.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(predS_arr) ))]
    gt_Coordinates = [gt_img.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(gtS_arr) ))]
                
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]
    dPredToGt = getDistancesFromAtoB(pred_Coordinates, gt_Coordinates)
    dGtToPred = getDistancesFromAtoB(gt_Coordinates, pred_Coordinates)

    asd_value = (np.sum(dPredToGt)+ np.sum(dGtToPred))/(len(pred_Coordinates)+len(gt_Coordinates))
       
    return asd_value

def compute_distance_physical(file_gt, file_seg):
    asd_value = ASD(file_gt=file_gt, file_pred=file_seg)
    hd95_value = HD95(file_gt=file_gt, file_pred=file_seg)
    return asd_value, hd95_value


def compute_distance_physical_bylib(file_gt, file_seg):
    labels= [1]
    metrics = sg.write_metrics(labels, file_gt, file_seg, metrics=['msd', 'hd95'])
    return metrics['msd'], metrics['hd95']

    


