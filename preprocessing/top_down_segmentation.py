"""module that creates segmented 'top-down' view"""

import numpy as np
from scipy.spatial import Delaunay
from ssl_project.constants import CATEGORY_TO_IDX, EGO_IMAGE_SIZE
from joblib import Parallel, delayed

import cv2
from ssl_project.data_loaders.data_helper import UnlabeledDataset, LabeledDataset

LINE_VALUE = 0.1111
EGO_TYPES = np.array([LINE_VALUE, 0.5019608, 0.827451 ,  0.98039216, 1.       ], dtype=np.float32)
EGO_NAMES = np.array(["line",    "sidewalk",    "road", "crosswalk", "ground"])
EGO_CLASS = np.arange(len(EGO_TYPES)) + max(CATEGORY_TO_IDX.values()) + 1

VALUE_TO_CATEGORY = {
    idx : cat for cat, idx in CATEGORY_TO_IDX.items()
}
VALUE_TO_CATEGORY.update(zip(EGO_CLASS, EGO_NAMES))


x, y = np.meshgrid(np.arange(EGO_IMAGE_SIZE), np.arange(EGO_IMAGE_SIZE))
xy_N2 = np.vstack((x.ravel(), y.ravel())).T

def _get_hull_from_bb(bb):
    bb_42 = bb.numpy().copy().T
    # many costyls
    bb_42 *= 10
    bb_42 += 400
    bb_42[:, 1] = 800 - bb_42[:, 1]
    hull = Delaunay(bb_42)
    return hull

def _create_mask(extra, target):
    ego_hw3 = extra['ego_image'].numpy().transpose(1, 2, 0).copy()
    lane_mask_hw = extra["lane_image"].numpy()
    np.place(ego_hw3, np.broadcast_to(lane_mask_hw[..., None], ego_hw3.shape), LINE_VALUE)
    # assert np.allclose(np.diff(ego_hw3, axis=-1), 0) # not true for crosswalks...
    ego_hw = ego_hw3[..., 0]
    
    assert np.isin(np.unique(ego_hw), EGO_TYPES).all()
    
    for typ, ego_class in zip(EGO_TYPES, EGO_CLASS):
        np.place(ego_hw, ego_hw == typ, ego_class)
    
    for bb, cat in zip(target['bounding_box'], target['category']):
        hull = _get_hull_from_bb(bb)
        mask_hw = (hull.find_simplex(xy_N2) >= 0).reshape(EGO_IMAGE_SIZE, EGO_IMAGE_SIZE)
        ego_hw[mask_hw] = cat

    return ego_hw


def _run(idx):
    labeled_trainset = LabeledDataset()
 
    scene_id, sample_id, path = labeled_trainset._get_ids_and_path(idx)
    target, road_image, ego_image, data_entries = labeled_trainset._get_target_road_ego_image(scene_id, sample_id, path)
    extra = labeled_trainset._get_extra(data_entries, ego_image)
    
    mask_hw = _create_mask(extra, target)
    cv2.imwrite(f"{path}/top_down_segm.png", mask_hw)


def create_data(n_jobs=8, slc=slice(None), debug=False):
    labeled_trainset = LabeledDataset()
    
    if debug:
        for idx in range(len(labeled_trainset))[slc]:
            _run(idx)
    else:
        Parallel(n_jobs=n_jobs)(
            delayed(_run)(idx) 
            for idx in range(len(labeled_trainset))[slc]
        )
