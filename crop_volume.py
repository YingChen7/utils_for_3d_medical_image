import SimpleITK as sitk
import numpy as np
from math import ceil 
import os
from read_write_image import readimage


def volume_to_tiles(volume, tile_size, overlap_size):
    volume_shape = [volume.GetDepth(),volume.GetHeight(),volume.GetWidth()]
    print('volume shape:')
    print(volume_shape)
    tiles_count = get_tiles_count(volume_shape,tile_size, overlap_size) 
    
    tiles = np.zeros(shape = (np.prod(tiles_count), tile_size[0],tile_size[1],tile_size[2]))
    
    padded_volume = pad_volume(volume,tiles_count,tile_size,overlap_size)
    padded_volume = sitk.GetArrayFromImage(padded_volume)
    
    mean_value = np.mean(padded_volume)
    std_value = np.std(padded_volume) 
    
    padded_volume = (padded_volume-mean_value)/std_value
    for d in range(tiles_count[0]):
        for h in range(tiles_count[1]):
            for w in range(tiles_count[2]):
                tiles[d*tiles_count[1]*tiles_count[2]+h*tiles_count[2]+w,:,:,:] = padded_volume[int(d*(tile_size[0]-overlap_size[0])):int(d*(tile_size[0]-overlap_size[0])+tile_size[0]),
                                                                                                int(h*(tile_size[1]-overlap_size[1])):int(h*(tile_size[1]-overlap_size[1])+tile_size[1]),
                                                                                                int(w*(tile_size[2]-overlap_size[2])):int(w*(tile_size[2]-overlap_size[2])+tile_size[2])]
    return tiles
        
def get_tiles_count(volume_size, tile_size, overlap_size):   
    count_depth = int(ceil(float(volume_size[0] - overlap_size[0]) / float(tile_size[0] - overlap_size[0])))
    count_height = int(ceil(float(volume_size[1] - overlap_size[1]) / float(tile_size[1] - overlap_size[1])))
    count_width = int(ceil(float(volume_size[2] - overlap_size[2]) / float(tile_size[2] - overlap_size[2])))
    return [count_depth,count_height,count_width]

def pad_volume(volume, tiles_count, tile_size, overlap_size):
    padder = sitk.ConstantPadImageFilter()
    padder.SetConstant(0)
    
    depth_pad_l,depth_pad_u = get_pad_value(volume.GetDepth(), tiles_count[0], tile_size[0], overlap_size[0])
    height_pad_l,height_pad_u = get_pad_value(volume.GetHeight(), tiles_count[1], tile_size[1], overlap_size[1])
    width_pad_l,width_pad_u = get_pad_value(volume.GetWidth(), tiles_count[2], tile_size[2], overlap_size[2])
    print('pad volume l:')
    print([depth_pad_l, height_pad_l, width_pad_l])
    print('pad volume u:')
    print([depth_pad_u, height_pad_u, width_pad_u])
    
    low_pad = sitk.VectorUInt32()
    low_pad.append(width_pad_l)
    low_pad.append(height_pad_l)
    low_pad.append(depth_pad_l)

    up_pad = sitk.VectorUInt32()
    up_pad.append(width_pad_u)
    up_pad.append(height_pad_u)
    up_pad.append(depth_pad_u)

    padder.SetPadLowerBound(low_pad)
    padder.SetPadUpperBound(up_pad)
    return padder.Execute(volume)

def get_pad_value(volume_size, tiles_count, tile_size, overlap_size):
    total = tile_size * tiles_count - overlap_size * (tiles_count-1)  - volume_size
    lowerpad = int(total / 2)
    upperpad = int(total - lowerpad)
    return lowerpad, upperpad

def standardize_info(volume_path, tile_size, overlap_size):
    volume = readimage(os.path.abspath(volume_path))
    volume_shape = [volume.GetDepth(),volume.GetHeight(),volume.GetWidth()]
    tiles_count = get_tiles_count(volume_shape,tile_size, overlap_size)     
    padded_volume = pad_volume(volume,tiles_count,tile_size,overlap_size)
    padded_volume = sitk.GetArrayFromImage(padded_volume)
    mean_value = np.mean(padded_volume)
    std_value = np.std(padded_volume) 
    return mean_value, std_value

def crop_volumes_into_tiles(volumes_list, tile_size, overlap_size):
    for index in range(len(volumes_list)):
        print('Volume:' + volumes_list[index])
        volume = readimage(os.path.abspath(volumes_list[index]))
        Origin = volume.GetOrigin()
        Spacing = volume.GetSpacing()
        Direction = volume.GetDirection()
        tiles = volume_to_tiles(volume, tile_size, overlap_size)
        write_path = volumes_list[index].split('.nii')[0]
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        for count in range(tiles.shape[0]):
            tile_to_write = sitk.GetImageFromArray(tiles[count,:,:,:])
            tile_to_write.SetOrigin(Origin)
            tile_to_write.SetSpacing(Spacing)
            tile_to_write.SetDirection(Direction)
            sitk.WriteImage(tile_to_write,write_path+'/'+str(count)+'.nii.gz')
        print('Successfully convert into tiles')


