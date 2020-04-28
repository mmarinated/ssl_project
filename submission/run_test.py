import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data_helper import LabeledDataset
from helper import compute_ats_bounding_boxes, compute_ts_road_map

from model_loader import get_transform, ModelLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--testset', action='store_true')
parser.add_argument('--verbose', action='store_true')
opt = parser.parse_args()

image_folder = opt.data_dir
annotation_csv = f'{opt.data_dir}/annotation.csv'

if opt.testset:
    labeled_scene_index = np.arange(134, 148)
else:
    labeled_scene_index = np.arange(120, 134)

labeled_trainset = LabeledDataset(
    image_folder=image_folder,
    annotation_file=annotation_csv,
    scene_index=labeled_scene_index,
    transform=get_transform(),
    extra_info=False
    )
dataloader = torch.utils.data.DataLoader(
    labeled_trainset,
    batch_size=1,
    shuffle=False,
    num_workers=4
    )

model_loader = ModelLoader()

total = 0
total_ats_bounding_boxes = 0
total_ts_road_map = 0
for i, data in enumerate(dataloader):
    total += 1
    sample, target, road_image = data
    sample = sample.cuda()

    predicted_bounding_boxes = model_loader.get_bounding_boxes(sample)[0].cpu()
    predicted_road_map = model_loader.get_binary_road_map(sample).cpu()

    ats_bounding_boxes = compute_ats_bounding_boxes(predicted_bounding_boxes, target['bounding_box'][0])
    ts_road_map = compute_ts_road_map(predicted_road_map, road_image)

    total_ats_bounding_boxes += ats_bounding_boxes
    total_ts_road_map += ts_road_map

    if opt.verbose:
        print(f'{i} - Bounding Box Score: {ats_bounding_boxes:.4} - Road Map Score: {ts_road_map:.4}')

print(f'{model_loader.team_name} - {model_loader.round_number} - Bounding Box Score: {total_ats_bounding_boxes / total:.4} - Road Map Score: {total_ts_road_map / total:.4}')
    









