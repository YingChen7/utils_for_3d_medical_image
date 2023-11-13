from crop_volume import  readimage, crop_volumes_into_tiles
from os.path import join
import glob
from eval_visual import compute_IOU_DSC, compute_distance_physical, compute_distance_physical_bylib
import pandas as pd
'''
1. test croping volume to tiles
'''
# tile_size = [40, 160, 128]
# overlap_size = [20, 80, 64]
# volumes_list = glob.glob(join('/home2/chenying_b/cerebrovascular2/asian-vs-white','*-Cr-MRA.nii.gz')) 

# convert_volumes_into_tiles(volumes_list, tile_size, overlap_size)

'''
2. test stitching tiles to original volume
'''

'''
tiles_seg_result = np.zeros(shape = [324,32,128,128])
frompath = 'D:\\test_gt\\8'
filelist=os.listdir(frompath)
files_count=len(filelist)
for index in range(files_count):
    tile = readimage(os.path.abspath(frompath + '/' + str(index) + '.nii.gz'))
    tile_array = sitk.GetArrayFromImage(tile)
    tiles_seg_result[index,:,:,:] = tile_array
    
volume_size, ref_origin, ref_spacing, ref_direction = ref_info_for_write(training_volumes[0])
volume_seg_result = stitch_tiles(tiles_seg_result, volume_size, tile_size, overlap_size)
write_seg_result(volume_seg_result, ref_origin, ref_spacing, ref_direction, 'D:\\test_gt\\8.nii.gz')
'''

'''
3. connect component analysis
'''
# generate_bin('E:\\nasal_dataset\\wwb_left.nii.gz', 'E:\\nasal_dataset\\wwb_left_bin.nii.gz', shresh =2561)
# connected_domain('E:\\nasal_dataset\\wwb_left_bin.nii.gz', 'E:\\nasal_dataset\\\wwb_left_bin.nii.gz')  
# generate_bin('E:\\nasal_dataset\\wwb_right.nii.gz', 'E:\\nasal_dataset\\wwb_right_bin.nii.gz', shresh = 2254)
# connected_domain('E:\\nasal_dataset\\wwb_right_bin.nii.gz', 'E:\\nasal_dataset\\wwb_right_bin.nii.gz')  

'''
4. Evaluate and visualize segmentation
'''
# Normaldata_path = r'/home2/chenying_data/Cerebrovascular2/dataHealth/train/'
# ABdata_path =  r'/home2/chenying_data/Cerebrovascular2/dataAB/train/'
# Normaldata_test_path = r'/home2/chenying_data/Cerebrovascular2/dataHealth/test/'
# ABdata_test_path =  r'/home2/chenying_data/Cerebrovascular2/dataAB/test/'
# volumes_Health = glob.glob(join(Normaldata_path, '*Res-MRA.nii.gz'))
# volumes_AB = glob.glob(join(ABdata_path, '*Res-MRA.nii.gz'))
# volumes_Health = np.sort(volumes_Health)
# volumes_AB = np.sort(volumes_AB)

# volumes_HH = [name for name in volumes_Health if 'HH' in name]
# volumes_Guys = [name for name in volumes_Health if 'Guys' in name]
# volumes_IOP = [name for name in volumes_Health if 'IOP' in name]
# volumes_AB  = [name for name in volumes_AB]

infer_path = r''
gt_path  = r''

infer_volumes= sorted(glob.glob(join(infer_path, '*.nii.gz')))

columns     = ["volume_id", "IoU", "DSC", "ASD", "HD95"]
pd_scores   = pd.DataFrame(columns=columns)

for index in range(len(infer_volumes)):
    val_volume_pred = infer_volumes[index]
    val_volume_GT = val_volume_pred.replace(infer_path, gt_path).replace('_Pred.nii.gz', '.nii.gz')
    IOU, DICE = compute_IOU_DSC(val_volume_GT, val_volume_pred)
    SASD, SHD95 = compute_distance_physical_bylib(file_gt=val_volume_GT,file_seg=val_volume_pred)
    #volume_postprocessing(val_volume_pred, val_volume_GT, val_volume_pred_c)
    pd_item = pd.DataFrame({"volume_id":[val_volume_pred.replace(infer_path, '')],\
                            "IoU":[IOU],\
                            "DSC":[DICE],\
                            "ASD":[SASD],\
                            "HD95":[SHD95],\
                                        })
    pd_scores = pd.concat([pd_scores, pd_item])

path_scores = infer_path + "metrics.csv"
pd_scores.to_csv(path_scores)


     