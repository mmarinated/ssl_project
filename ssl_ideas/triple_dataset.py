from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

from ssl_project.utils import TRANSFORM
from ssl_project.constants import CAM_NAMES, NUM_SAMPLE_PER_SCENE
from ssl_project.paths import PATH_TO_DATA

def load_image(scene_id, cam_name, sample_id, transform=TRANSFORM):
    path = os.path.join(PATH_TO_DATA, f'scene_{scene_id}', f'sample_{sample_id}',  f"{cam_name}.jpeg") 
    image = Image.open(path)
    return transform(image)


def TripleDataset(cam_names=CAM_NAMES, scene_ids=None):
    """it is a function which returns an instance of TripleDataset"""
    idces_offset_with_label = [
         ([0, 1, 2], 1),    # positive_1
         ([0, 3, 2], 0),
         ([1, 0, 3], 0),
    ]
#     idces_offset_with_label = [
#          ([0, 1], 1), # forward or backward
#          ([1, 0], 0),
#     ]
    my_datasets = []
    for cam_name in cam_names: # maybe you want just one CAM_NAME
        for scene_id in scene_ids:
            for idces_offset, label in idces_offset_with_label:
                cur_dataset = _HelperForTripleDataset(cam_name, scene_id, idces_offset, label)
                my_datasets.append(cur_dataset)

    super_dataset = torch.utils.data.ConcatDataset(my_datasets)
    
    return super_dataset


class _HelperForTripleDataset(torch.utils.data.Dataset):
    def __init__(self, cam_name, scene_id, idces_offset, label):
        self.cam_name = cam_name
        self.scene_id = scene_id
        self.idces_offset = idces_offset
        self.label = label
        
    def __len__(self):
        return NUM_SAMPLE_PER_SCENE - max(self.idces_offset)
    
    def __getitem__(self, idx):
        """
        return 
            images_o3hw, label
        """
        # where n_o = num offsets
        images_o3hw = torch.stack([
            load_image(self.scene_id, self.cam_name, idx + offset)
            for offset in self.idces_offset
        ])
        
        return images_o3hw, self.cam_name, self.label