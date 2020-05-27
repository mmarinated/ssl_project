import torch
import numpy as np
import random
import cv2
from .utils import convert_from_bb_space, convert_to_bb_space

###
# Baseline for cars
###

def get_baseline_raw_bbs(in_bb_space=False):
    """ returns grid_bbs_k24"""
    grid_bbs = []

    len_x, len_y = 50., 25.
    offset_x, from_y, to_y = 20., 420., 440 
    for x_start in np.arange(offset_x, 800 - offset_x, len_x):
        for y_start in np.arange(from_y, to_y, len_y):
            bb_24 = torch.Tensor([
                [x_start + len_x, x_start + len_x, x_start, x_start],
                [y_start + len_y, y_start, y_start + len_y, y_start],
            ])
            grid_bbs.append(bb_24)

    grid_bbs_k24 = torch.stack(grid_bbs)
    if in_bb_space:
        grid_bbs_k24 = convert_to_bb_space(grid_bbs_k24, axis=-2)
    return grid_bbs_k24
    
###
# Generate data for experiments
###

def _generate_random_bbs_of_fixed_car_size(k, size_xy=(40, 20)):
    """
    returns: random_bbs_k24
    """
    random_k2 = torch.Tensor(np.random.randint(0, 800, size=(k, 2)).astype(float))
    random_bbs_k24 = torch.zeros(k, 2, 4)
    random_bbs_k24 += random_k2[..., None]

    # shift_24 = torch.Tensor(np.random.randint(0, 40, size=(2, 4)).astype(float))
    shift_24 = torch.Tensor([
        [0, 0, -size_xy[0], -size_xy[0]],
        [0, -size_xy[1], 0, -size_xy[1]]
    ]).float()
    random_bbs_k24 += shift_24[None]
    return random_bbs_k24

def generate_random_bbs(k):
    """
    returns: random_bbs_k24
    """
    random_bbs_k24 = []
    for _ in range(k):
        x, y = random.randint(2, 200), random.randint(2, 20)
        random_bbs_k24.append(_generate_random_bbs_of_fixed_car_size(1, (x, y)))
    return torch.cat(random_bbs_k24)