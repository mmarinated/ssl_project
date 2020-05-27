import torch
import numpy as np
import random
import cv2


def convert_to_bb_space(points_a2, axis=-1):
    """
    a: 
        any shape
    axis: 
        axis of dim=2 -- (x, y) 
    """
    assert points_a2.shape[axis] == 2, f"axis={axis} should have size 2: (x, y)"
    points_a2 = _my_transpose(_my_copy(points_a2), axis, -1)
    
    # TODO FIXME change 800 to constants. ...  
    points_a2[..., 1] = 800 - points_a2[..., 1]
    ans_a2 = (points_a2 - 400) / 10
    
    return _my_transpose(ans_a2, axis, -1)

def convert_from_bb_space(points_a2, axis=-1):
    """
    a: 
        any shape
    axis: 
        axis of dim=2 -- (x, y) 
    """
    assert points_a2.shape[axis] == 2, f"axis={axis} should have size 2: (x, y)"
    points_a2 = _my_transpose(_my_copy(points_a2), axis, -1)
        
    points_a2 = points_a2 * 10 + 400
    points_a2[..., 1] = 800 - points_a2[..., 1]
    
    return _my_transpose(points_a2, axis, -1)


###
## pytorch / numpy agnostic
###

def _my_transpose(arr, axis_1, axis_2):
    """works both for pytorch and numpy"""
    try:
        return arr.transpose(axis_1, axis_2)
    except:
        return arr.swapaxes(axis_1, axis_2)
    
def _my_copy(arr):
    """works both for pytorch and numpy"""
    try:
        return arr.clone()
    except:
        return arr.copy()
