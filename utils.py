import numpy as np
from matplotlib.path import Path
import cv2
import torch

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from shapely.geometry import Polygon

def to_np(x):
    return x.detach().data.squeeze().numpy()


def bounding_boxes_to_segmentation(full_width, full_height, scale, bounding_boxes, categories):
    out = torch.zeros((full_height, full_width), dtype=torch.long)
    
    ind = np.indices((full_height, full_width))
    ind[0] = ind[0] - full_height / 2
    ind[1] = ind[1] - full_width / 2
    ind = np.moveaxis(ind, 0, 2)
    for i, b in enumerate(bounding_boxes.detach().data.numpy()):
        p = Path([b[:,0]*scale,b[:,1]*scale,b[:,3]*scale,b[:,2]*scale])
        g = p.contains_points(ind.reshape(full_width*full_height,2))
        g = np.flip(g.reshape(full_height, full_width), axis=1).T
        g = g.copy()
        out[g] = 1

    return out

def get_bounding_boxes_from_seg(segment_tensor, scale, full_height, full_width):
    _, contours, _ = cv2.findContours(to_np(segment_tensor).astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        
    def convert(value, dim):
        if dim == "x":
            out = ( value - full_width / 2 ) / scale
        elif dim == "y":
             out = -( value - full_height / 2 ) / scale
        return out
    for i in range(len(boundRect)):
        boundRect[i] = [[convert(boundRect[i][0] + boundRect[i][2], "x"), convert(boundRect[i][1], "y")], 
                        [convert(boundRect[i][0] + boundRect[i][2], "x"), convert(boundRect[i][1] +boundRect[i][3], "y")],
                        [convert(boundRect[i][0], "x"), convert(boundRect[i][1], "y")],
                        [convert(boundRect[i][0], "x"), convert(boundRect[i][1] + boundRect[i][3], "y")]
                        ]
        
    return torch.FloatTensor(boundRect).permute(0,2,1)


def compute_ats_bounding_boxes(boxes1, boxes2):
    num_boxes1 = boxes1.size(0)
    num_boxes2 = boxes2.size(0)

    boxes1_max_x = boxes1[:, 0].max(dim=1)[0]
    boxes1_min_x = boxes1[:, 0].min(dim=1)[0]
    boxes1_max_y = boxes1[:, 1].max(dim=1)[0]
    boxes1_min_y = boxes1[:, 1].min(dim=1)[0]

    boxes2_max_x = boxes2[:, 0].max(dim=1)[0]
    boxes2_min_x = boxes2[:, 0].min(dim=1)[0]
    boxes2_max_y = boxes2[:, 1].max(dim=1)[0]
    boxes2_min_y = boxes2[:, 1].min(dim=1)[0]

    condition1_matrix = (boxes1_max_x.unsqueeze(1) > boxes2_min_x.unsqueeze(0))
    condition2_matrix = (boxes1_min_x.unsqueeze(1) < boxes2_max_x.unsqueeze(0))
    condition3_matrix = (boxes1_max_y.unsqueeze(1) > boxes2_min_y.unsqueeze(0))
    condition4_matrix = (boxes1_min_y.unsqueeze(1) < boxes2_max_y.unsqueeze(0))
    condition_matrix = condition1_matrix * condition2_matrix * condition3_matrix * condition4_matrix

    iou_matrix = torch.zeros(num_boxes1, num_boxes2)
    for i in range(num_boxes1):
        for j in range(num_boxes2):
            if condition_matrix[i][j]:
                iou_matrix[i][j] = compute_iou(boxes1[i], boxes2[j])

    iou_max = iou_matrix.max(dim=0)[0]

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    total_threat_score = 0
    total_weight = 0
    for threshold in iou_thresholds:
        tp = (iou_max > threshold).sum()
        threat_score = tp * 1.0 / (num_boxes1 + num_boxes2 - tp)
        total_threat_score += 1.0 / threshold * threat_score
        total_weight += 1.0 / threshold

    average_threat_score = total_threat_score / total_weight
    
    return average_threat_score

def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()

    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)

def compute_iou(box1, box2):
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull
    
    return a.intersection(b).area / a.union(b).area
