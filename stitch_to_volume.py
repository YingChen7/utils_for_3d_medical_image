import numpy as np
from crop_volume import readimage, get_tiles_count, get_pad_value

def stitch_tiles(tiles_seg_result, volume_size, tile_size, overlap_size):
    tiles_count = get_tiles_count(volume_size, tile_size, overlap_size)
    
    depth_pad_l,depth_pad_u = get_pad_value(volume_size[0], tiles_count[0], tile_size[0], overlap_size[0])
    height_pad_l,height_pad_u = get_pad_value(volume_size[1], tiles_count[1], tile_size[1], overlap_size[1])
    width_pad_l,width_pad_u = get_pad_value(volume_size[2], tiles_count[2], tile_size[2], overlap_size[2])
    padded_volume_size = [volume_size[0] + depth_pad_l + depth_pad_u, volume_size[1] + height_pad_l + height_pad_u, volume_size[2] + width_pad_l + width_pad_u]
    
    padded_volume_seg_result = np.zeros(shape = padded_volume_size)
    for d in range(tiles_count[0]):
        for h in range(tiles_count[1]):
            for w in range(tiles_count[2]):
                padded_volume_seg_result[int(d*(tile_size[0]-overlap_size[0])):int(d*(tile_size[0]-overlap_size[0])+tile_size[0]),
                                         int(h*(tile_size[1]-overlap_size[1])):int(h*(tile_size[1]-overlap_size[1])+tile_size[1]),
                                         int(w*(tile_size[2]-overlap_size[2])):int(w*(tile_size[2]-overlap_size[2])+tile_size[2])] = tiles_seg_result[d*tiles_count[1]*tiles_count[2]+h*tiles_count[2]+w,:,:,:]
    
    volume_seg_result = np.zeros(shape = volume_size)
    volume_seg_result = padded_volume_seg_result[depth_pad_l:depth_pad_l + volume_size[0], height_pad_l:height_pad_l + volume_size[1], width_pad_l:width_pad_l + volume_size[2]]   
    return volume_seg_result

