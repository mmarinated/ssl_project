import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def convert_map_to_lane_map(ego_map, binary_lane):
    mask = (ego_map[0,:,:] == ego_map[1,:,:]) * (ego_map[1,:,:] == ego_map[2,:,:]) + (ego_map[0,:,:] == 250 / 255)

    if binary_lane:
        return (~ mask)
    return ego_map * (~ mask.view(1, ego_map.shape[1], ego_map.shape[2]))

def convert_map_to_road_map(ego_map):
    mask = (ego_map[0,:,:] == 1) * (ego_map[1,:,:] == 1) * (ego_map[2,:,:] == 1)

    return (~mask)

def collate_fn(batch):
    return tuple(zip(*batch))

def draw_box(ax, corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    
    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)





