import numpy as np


PHOTO_H = 256
PHOTO_W = 306
EGO_IMAGE_SIZE = 800

CAM_NAMES = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
CAM_ANGLE = [60,                    0,             -60,           120,            180,          -120]
IDX_TO_ANGLE = dict(enumerate(CAM_ANGLE))

VIEW_ANGLE = 70

CATEGORY_TO_IDX = {
    'other_vehicle': 0,
    'bicycle': 1,
    'car': 2,
    'pedestrian': 3,
    'truck': 4,
    'bus': 5,
    'motorcycle': 6,
    'emergency_vehicle': 7,
    'animal': 8,
}

IDX_TO_CATEGORY = {v:k for k, v in CATEGORY_TO_IDX.items()}

CATEGORY_TO_HEIGHT = {
    'other_vehicle': 1.8,
    'bicycle': 1.5,
    'car': 1.8,
    'pedestrian': 1.7,
    'truck': 3.4,
    'bus': 4,
    'motorcycle': 1.5,
    'emergency_vehicle': 3,
    'animal': 1.5,
}

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6

UNLABELED_SCENE_INDEX = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
LABELED_SCENE_INDEX = np.arange(106, 134)


