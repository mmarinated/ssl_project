import numpy as np


PHOTO_H = 256
PHOTO_W = 306

CAM_NAMES = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
CAM_ANGLE = [60,                    0,             -60,           120,            180,          -120]
IDX_TO_ANGLE = dict(enumerate(CAM_ANGLE))

VIEW_ANGLE = 70

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6

UNLABELED_SCENE_INDEX = np.arange(106)
# The scenes from 106 - 133 are labeled
# You should devide the labeled_scene_index into two subsets (training and validation)
LABELED_SCENE_INDEX = np.arange(106, 134)


